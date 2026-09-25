"""Nodes in the one-action Rogue agent graph."""

import json
import time
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from game_runner.clients.rogue import (
    ActionRequest, KeyActionRequest, RogueAPIError, RogueClient, RogueTransportError,
)
from game_runner.context.messages import build_messages
from game_runner.graph.state import StepState
from game_runner.persistence.repository import GameRepository


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Annotated[ActionRequest | KeyActionRequest, Field(discriminator="type")]
    rationale: str = Field(min_length=1, max_length=300)


class DecisionValidationError(RuntimeError):
    pass


class SessionInterrupted(RuntimeError):
    pass


def validate_mode(decision: Decision, mode: str) -> None:
    action = decision.action
    if mode == "confirm":
        if not isinstance(action, ActionRequest) or action.action not in {"CONFIRM", "CANCEL"}:
            raise ValueError("confirm mode requires CONFIRM or CANCEL")
    elif mode in {"select_item", "select_direction", "select_hand"}:
        if isinstance(action, ActionRequest) and action.action != "CANCEL":
            raise ValueError(f"{mode} requires a key action or CANCEL")
        if isinstance(action, KeyActionRequest):
            allowed = {
                "select_item": set("abcdefghijklmnopqrstuvwxyz*"),
                "select_direction": set("hjklyubn"),
                "select_hand": {"l", "r"},
            }[mode]
            if action.key not in allowed:
                raise ValueError(f"{mode} received an invalid key")
    elif mode in {"game_over", "more"}:
        raise ValueError(f"cannot choose an action in {mode} mode")


def make_nodes(
    model: BaseChatModel, client: RogueClient, repository: GameRepository,
    *, context_token_budget: int, max_model_calls: int,
):
    def context(state: StepState) -> dict:
        history = repository.recent_actions(state["session_id"])
        return {
            "messages": build_messages(state["observation"], history, context_token_budget),
            "attempts": 0,
            "action": None,
        }

    async def invoke_model(state: StepState) -> dict:
        messages: list[BaseMessage] = state["messages"]
        input_messages = [{"role": message.type, "content": message.content} for message in messages]
        call_id = repository.start_model_call(state["session_id"], input_messages)
        started = time.monotonic()
        try:
            response = await model.ainvoke(messages)
        except Exception as exc:
            repository.finish_model_call(
                call_id, "error", validation_error=str(exc),
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
        decision: Decision | None = None
        try:
            if not isinstance(response.content, str):
                raise ValueError("model content must be a JSON string")
            decision = Decision.model_validate(json.loads(response.content))
            validate_mode(decision, state["observation"].mode)
        except (ValueError, ValidationError) as exc:
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
            return {
                "action": None,
                "messages": [
                    *state["messages"], response,
                    HumanMessage(content=f"Invalid action response: {error}. Return one valid JSON object only."),
                ],
            }
        assert decision is not None
        return {"action": decision.action, "rationale": decision.rationale}

    async def perform_action(state: StepState) -> dict:
        action = state["action"]
        assert action is not None
        action_payload = action.model_dump(mode="json", exclude_none=True)
        event_id = repository.start_game_event(
            state["session_id"], "action", model_call_id=state["call_id"],
            action_request=action_payload,
        )
        before = state["observation"]
        try:
            observed = await client.action(before.episode_id, action)
        except RogueAPIError as exc:
            repository.finish_game_event(event_id, "failed", http_status=exc.status_code, error=exc.detail)
            raise
        except RogueTransportError as exc:
            try:
                recovered = await client.observe()
            except (RogueAPIError, RogueTransportError, ValueError) as recovery_error:
                repository.finish_game_event(event_id, "uncertain", error=f"{exc}; reconciliation failed: {recovery_error}")
                raise SessionInterrupted("cannot reconcile timed-out action") from recovery_error
            if recovered.episode_id != before.episode_id or recovered.api_step != before.api_step + 1:
                repository.finish_game_event(
                    event_id, "uncertain", api_step=recovered.api_step,
                    error=f"{exc}; episode or step mismatch during reconciliation",
                )
                raise SessionInterrupted("action outcome cannot be confirmed") from exc
            repository.finish_game_event(
                event_id, "uncertain", api_step=recovered.api_step,
                observation=recovered.model_dump(mode="json"),
                error=f"{exc}; state recovered but action output was lost",
            )
            return {"next_observation": recovered}
        except Exception as exc:
            repository.finish_game_event(event_id, "uncertain", error=str(exc))
            raise
        repository.finish_game_event(
            event_id, "succeeded", api_step=observed.api_step,
            http_status=200, observation=observed.model_dump(mode="json"),
        )
        return {"next_observation": observed}

    return context, invoke_model, validate, perform_action
