"""Nodes in the one-action Rogue agent graph."""

from dataclasses import asdict
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel

from game_runner.budget import BudgetStop, TrialBudget, within_budget
from game_runner.clients.rogue import (
    RogueAPIError, RogueClient, RogueTransportError,
)
from game_runner.context.budget import ContextLimitError, RETRY_RESERVE, count_messages
from game_runner.context.planning import ContextPlanner, ContextSettings
from game_runner.context.progress import Progress
from game_runner.decision import parse_decision
from game_runner.graph.state import StepState
from game_runner.persistence.repository import GameRepository


class DecisionValidationError(RuntimeError):
    pass


class SessionInterrupted(RuntimeError):
    pass


def make_nodes(
    model: BaseChatModel, client: RogueClient, repository: GameRepository,
    *, context_settings: ContextSettings, max_model_calls: int, summary_model=None,
    budget: TrialBudget | None = None,
):
    planner = ContextPlanner(model, repository, context_settings, summary_model, budget)

    async def context(state: StepState) -> dict:
        observation, event_id = state["observation"], state["observation_event_id"]
        progress = state.get("progress") or Progress.initial(event_id, observation)
        prepared = await planner.prepare(
            state["session_id"], observation, event_id, state["observation_outcome"], progress,
        )
        return {
            "messages": prepared.messages, "base_messages": prepared.messages,
            "memory_revision_id": prepared.revision_id, "progress": progress,
            "attempts": 0,
            "action": None,
        }

    async def invoke_model(state: StepState) -> dict:
        messages: list[BaseMessage] = state["messages"]
        count = count_messages(model, messages)
        if count.count + context_settings.actor_output_tokens > context_settings.token_budget:
            raise ContextLimitError("actor request and reserved output exceed context budget")
        if budget:
            budget.start_call()
        input_messages = [{"role": message.type, "content": message.content} for message in messages]
        call_id = repository.start_model_call(
            state["session_id"], input_messages, observation_event_id=state["observation_event_id"],
            memory_revision_id=state["memory_revision_id"],
            input_token_count=count.count, input_token_count_method=count.method,
        )
        started = time.monotonic()
        try:
            response = await within_budget(budget, lambda: model.ainvoke(messages))
        except Exception as exc:
            repository.finish_model_call(
                call_id, "error", validation_error=(
                    type(exc).__name__ if budget and not isinstance(exc, BudgetStop) else str(exc)
                ),
                duration_ms=round((time.monotonic() - started) * 1000),
            )
            raise
        if not isinstance(response, AIMessage):
            repository.finish_model_call(call_id, "error", validation_error="model did not return AIMessage")
            raise TypeError("model did not return AIMessage")
        return {
            "response": response,
            "call_id": call_id,
            "duration_ms": round((time.monotonic() - started) * 1000),
            "attempts": state["attempts"] + 1,
        }

    def validate(state: StepState) -> dict:
        response = state["response"]
        raw = response.model_dump(mode="json")
        error: str | None = None
        decision = None
        try:
            decision = parse_decision(response.content, state["observation"].mode)
        except ValueError as exc:
            error = str(exc)
        repository.finish_model_call(
            state["call_id"], "invalid" if error else "valid",
            raw_response=raw,
            parsed_action=decision.action.model_dump(mode="json", exclude_none=True) if decision and not error else None,
            decision_rationale=decision.rationale if decision and not error else None,
            validation_error=error,
            response_metadata=raw.get("response_metadata"),
            usage_metadata=raw.get("usage_metadata"),
            duration_ms=state["duration_ms"],
        )
        if error:
            if state["attempts"] >= max_model_calls:
                raise DecisionValidationError(f"model response invalid after {max_model_calls} attempts: {error}")
            # Keep retry feedback bounded. The complete rejected reply remains
            # in the audit record and cannot grow the next request.
            base = state["base_messages"]
            feedback = "The previous response failed validation. Return one valid JSON object only."
            if _fits_retry(model, base, feedback + " " + error):
                feedback += " " + error
            retry = [*base[:-1], HumanMessage(content=base[-1].content + "\n" + feedback)]
            if not _fits_retry(model, base, feedback):
                raise ContextLimitError("actor correction exceeds retry reserve")
            return {"action": None, "messages": retry}
        assert decision is not None
        return {"action": decision.action, "rationale": decision.rationale}

    async def perform_action(state: StepState) -> dict:
        action = state["action"]
        assert action is not None
        action_payload = action.model_dump(mode="json", exclude_none=True)
        if budget:
            budget.start_action()
        event_id = repository.start_game_event(
            state["session_id"], "action", model_call_id=state["call_id"],
            action_request=action_payload,
        )
        before = state["observation"]
        try:
            observed = await within_budget(budget, lambda: client.action(before.episode_id, action))
        except RogueAPIError as exc:
            repository.finish_game_event(event_id, "failed", http_status=exc.status_code, error=exc.detail)
            raise
        except RogueTransportError as exc:
            try:
                recovered = await within_budget(budget, client.observe)
            except BudgetStop as stop:
                repository.finish_game_event(event_id, "uncertain", error=str(stop))
                raise
            except (RogueAPIError, RogueTransportError, ValueError) as recovery_error:
                repository.finish_game_event(event_id, "uncertain", error=f"{exc}; reconciliation failed: {recovery_error}")
                raise SessionInterrupted("cannot reconcile timed-out action") from recovery_error
            if recovered.episode_id != before.episode_id or recovered.api_step != before.api_step + 1:
                repository.finish_game_event(
                    event_id, "uncertain", api_step=recovered.api_step,
                    error=f"{exc}; episode or step mismatch during reconciliation",
                )
                raise SessionInterrupted("action outcome cannot be confirmed") from exc
            progress = state["progress"].advance(event_id, before, action_payload, recovered, recovered=True)
            repository.finish_game_event(
                event_id, "uncertain", api_step=recovered.api_step,
                observation=recovered.model_dump(mode="json"),
                progress=asdict(progress),
                error=f"{exc}; state recovered but action output was lost",
            )
            return {
                "next_observation": recovered, "next_observation_event_id": event_id,
                "next_observation_outcome": "uncertain",
                "next_progress": progress,
            }
        except Exception as exc:
            repository.finish_game_event(event_id, "uncertain", error=str(exc))
            raise
        progress = state["progress"].advance(event_id, before, action_payload, observed, recovered=False)
        repository.finish_game_event(
            event_id, "succeeded", api_step=observed.api_step,
            http_status=200, observation=observed.model_dump(mode="json"),
            progress=asdict(progress),
        )
        return {
            "next_observation": observed, "next_observation_event_id": event_id,
            "next_observation_outcome": "succeeded",
            "next_progress": progress,
        }

    return context, invoke_model, validate, perform_action


def _fits_retry(model, base: list[BaseMessage], feedback: str) -> bool:
    corrected = [*base[:-1], HumanMessage(content=base[-1].content + "\n" + feedback)]
    return count_messages(model, corrected).count - count_messages(model, base).count <= RETRY_RESERVE
