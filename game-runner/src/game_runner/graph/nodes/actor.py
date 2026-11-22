"""Prepare, invoke, and validate the actor's decision."""

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from game_runner.budget import BudgetStop, TrialBudget, within_budget
from game_runner.context.budget import ContextLimitError, RETRY_RESERVE, count_messages
from game_runner.context.planning import ContextPlanner, ContextSettings
from game_runner.context.progress import Progress
from game_runner.decision import ManagedDecision, parse_decision
from game_runner.graph.state import StepState
from game_runner.persistence.repository import GameRepository
from game_runner.prompts.decision_retry import DECISION_RETRY_INSTRUCTION


class DecisionValidationError(RuntimeError):
    pass


def make_actor_nodes(
    model: BaseChatModel, repository: GameRepository,
    *, context_settings: ContextSettings, max_model_calls: int, summary_model=None,
    budget: TrialBudget | None = None,
):
    planner = ContextPlanner(model, repository, context_settings, summary_model, budget)

    async def context(state: StepState) -> dict:
        observation, event_id = state["observation"], state["observation_event_id"]
        progress = state.get("progress") or Progress.initial(event_id, observation)
        loop = state.get("goal_loop")
        prepared = await planner.prepare(
            state["session_id"], observation, event_id, state["observation_outcome"], progress,
            state.get("working_state"), loop.actor_record() if loop is not None else None,
        )
        return {
            "messages": prepared.messages, "base_messages": prepared.messages,
            "memory_revision_id": prepared.revision_id, "progress": progress,
            "history_window": prepared.history_window,
            "attempts": 0,
            "action": None,
        }

    def actor_metadata(state: StepState) -> dict | None:
        loop = state.get("goal_loop")
        if loop is None:
            return None
        goal = loop.goal
        return {"active_goal_id": goal.id if goal is not None and goal.status == "active" else None}

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
            input_token_count=count.count, input_token_count_method=count.method, history_window=state["history_window"],
            decision_metadata=actor_metadata(state),
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
            decision = parse_decision(response, state["observation"].mode, managed=state.get("goal_loop") is not None)
        except ValueError as exc:
            error = str(exc)
        metadata = None
        if isinstance(decision, ManagedDecision):
            metadata = actor_metadata(state)
            assert metadata is not None
            metadata["request_goal_review"] = decision.request_goal_review
        repository.finish_model_call(
            state["call_id"], "invalid" if error else "valid",
            raw_response=raw,
            parsed_action=decision.action.model_dump(mode="json", exclude_none=True) if decision and not error else None,
            decision_rationale=decision.rationale if decision and not error else None,
            validation_error=error,
            response_metadata=raw.get("response_metadata"),
            usage_metadata=raw.get("usage_metadata"),
            duration_ms=state["duration_ms"], decision_metadata=metadata,
        )
        if error:
            if state["attempts"] >= max_model_calls:
                raise DecisionValidationError(f"model response invalid after {max_model_calls} attempts: {error}")
            # Keep retry feedback bounded. The complete rejected reply remains
            # in the audit record and cannot grow the next request.
            base = state["base_messages"]
            feedback = DECISION_RETRY_INSTRUCTION
            if _fits_retry(model, base, feedback + " " + error):
                feedback += " " + error
            retry = [*base[:-1], HumanMessage(content=base[-1].content + "\n" + feedback)]
            if not _fits_retry(model, base, feedback):
                raise ContextLimitError("actor correction exceeds retry reserve")
            return {"action": None, "messages": retry}
        assert decision is not None
        return {
            "action": decision.action, "rationale": decision.rationale,
            "request_goal_review": isinstance(decision, ManagedDecision) and decision.request_goal_review,
        }

    return context, invoke_model, validate


def _fits_retry(model, base: list[BaseMessage], feedback: str) -> bool:
    corrected = [*base[:-1], HumanMessage(content=base[-1].content + "\n" + feedback)]
    return count_messages(model, corrected).count - count_messages(model, base).count <= RETRY_RESERVE
