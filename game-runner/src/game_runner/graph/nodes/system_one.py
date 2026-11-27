"""Offer Rogue commands as a typed choice while retaining the graph's action contract."""

import json
import string
import time

import httpx
from langchain_core.messages import BaseMessage

from game_runner.budget import BudgetStop, TrialBudget, within_budget
from game_runner.clients.decisions import DecisionsClient
from game_runner.clients.rogue import ActionRequest, KeyActionRequest, Observation
from game_runner.context.budget import ContextLimitError
from game_runner.context.planning import ContextPlanner, ContextSettings
from game_runner.context.progress import DIRECTIONS, SLOT, Progress
from game_runner.decision import Decision, validate_mode
from game_runner.graph.nodes.actor import DecisionValidationError, prepare_actor_context
from game_runner.graph.state import StepState
from game_runner.persistence.repository import GameRepository
from game_runner.prompts.game_rules import GAME_RULES

SYSTEM_ONE_INSTRUCTION = (
    "Choose the best available command for the current Rogue 5.4 observation. "
    "The last human message is the latest observation. Earlier human and assistant messages show "
    "observations and selected actions. Use recorded map and inventory facts within their scope "
    "and check their freshness. A recovered observation leaves the preceding action effect uncertain. "
    "Choose item commands to open Rogue's prompts, then answer each prompt when it appears. "
    "Choose quit only when continuing is impossible.\n\n" + GAME_RULES
)
QUESTION_INSTRUCTION = "Which single available action best advances the game objective from the latest observation?"
QUESTION_RETRY = " The previous answer was invalid. Select one of the offered choice IDs."

SIMPLE_COMMANDS = {
    "rest": "Rest for one turn.",
    "search": "Search nearby for hidden terrain or traps.",
    "ascend": "Use the upstairs command.",
    "descend": "Use the downstairs command.",
    "pick_up": "Pick up the item on the current tile.",
    "inventory": "Inspect the inventory.",
    "take_off_armor": "Take off the equipped armor.",
    "quit": "Open the quit confirmation prompt.",
}
ITEM_COMMANDS = {
    "eat": ("e", "Choose a food item to eat."),
    "quaff": ("q", "Choose a potion to drink."),
    "read": ("r", "Choose a scroll to read."),
    "wield": ("w", "Choose a weapon to wield."),
    "wear": ("W", "Choose armor to wear."),
    "drop": ("d", "Choose an item to drop."),
    "throw": ("t", "Choose a direction and an item to throw."),
    "zap": ("z", "Choose a direction and a wand or staff to zap."),
    "put_on_ring": ("P", "Choose a ring to put on."),
    "remove_ring": ("R", "Remove an equipped ring, choosing a hand if prompted."),
}

type ChoiceMenu = dict[str, tuple[str, ActionRequest | KeyActionRequest]]


def choice_menu(observation: Observation, progress: Progress) -> ChoiceMenu:
    mode = observation.mode
    if observation.terminated or mode in {"more", "game_over"}:
        raise DecisionValidationError(f"cannot choose an action in {mode} mode")
    if mode == "confirm":
        return {
            "confirm": ("Confirm the current prompt.", ActionRequest(action="CONFIRM")),
            "cancel": ("Cancel the current prompt.", ActionRequest(action="CANCEL")),
        }
    view = [*observation.messages, *observation.screen]
    if mode in {"normal", "inventory", "unknown"} and any(
        "press return to continue" in line.lower() for line in view
    ):
        return {"enter": ("Continue past the prompt.", KeyActionRequest(key="ENTER"))}
    if mode == "inventory":
        return {
            "enter": ("Continue past the inventory.", KeyActionRequest(key="ENTER")),
            "escape": ("Close the inventory.", KeyActionRequest(key="ESCAPE")),
        }
    cancel = ("Cancel the current prompt.", ActionRequest(action="CANCEL"))
    if mode == "select_direction":
        return {
            **{
                f"direction_{direction.lower()}": (
                    f"Aim {direction.lower().replace('_', ' ')}.", KeyActionRequest(key=key)
                )
                for key, direction in DIRECTIONS.items()
            },
            "cancel": cancel,
        }
    if mode == "select_hand":
        return {
            "hand_left": ("Select the left hand.", KeyActionRequest(key="l")),
            "hand_right": ("Select the right hand.", KeyActionRequest(key="r")),
            "cancel": cancel,
        }
    if mode == "select_item":
        slots = {match[1]: match[2] for line in view if (match := SLOT.fullmatch(line))}
        if not slots and progress.inventory is not None and not progress.inventory.stale:
            slots = dict(progress.inventory.slots)
        return {
            **{
                f"item_{key}": (
                    f"Select inventory slot {key} ({slots[key]})." if key in slots else f"Select inventory slot {key}.",
                    KeyActionRequest(key=key),
                )
                for key in sorted(slots) or string.ascii_lowercase
            },
            "list_items": ("Ask Rogue to show selectable items.", KeyActionRequest(key="*")),
            "cancel": cancel,
        }
    if mode != "normal":
        raise DecisionValidationError(f"cannot choose an action in {mode} mode")
    return {
        **{
            f"move_{direction.lower()}": (
                f"Move or attack {direction.lower().replace('_', ' ')}.",
                ActionRequest(action=f"MOVE_{direction}"),
            )
            for direction in DIRECTIONS.values()
        },
        **{name: (description, ActionRequest(action=name.upper())) for name, description in SIMPLE_COMMANDS.items()},
        **{name: (description, KeyActionRequest(key=key)) for name, (key, description) in ITEM_COMMANDS.items()},
    }


def _request(model_name: str, messages: list[BaseMessage], options: ChoiceMenu, *, retry: bool = False) -> dict:
    return {
        "model": model_name,
        "state": {"messages": [
            {"role": message.type, "content": SYSTEM_ONE_INSTRUCTION if i == 0 else message.content}
            for i, message in enumerate(messages)
        ]},
        "questions": {"next_action": {
            "type": "choice",
            "instructions": QUESTION_INSTRUCTION + (QUESTION_RETRY if retry else ""),
            "criteria": {name: description for name, (description, _) in options.items()},
        }},
    }


def _input_size(payload: dict) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def make_system_one_nodes(
    model: DecisionsClient, repository: GameRepository, *, context_settings: ContextSettings,
    max_model_calls: int, budget: TrialBudget | None = None,
):
    planner = ContextPlanner(model, repository, context_settings)

    async def context(state: StepState) -> dict:
        observation = state["observation"]
        progress = state.get("progress") or Progress.initial(state["observation_event_id"], observation)
        options = choice_menu(observation, progress)
        prepared = await prepare_actor_context(
            planner, state, progress=progress,
            count_input=lambda messages: _input_size(_request(model.model_name, messages, options)),
        )
        return {**prepared, "choice_options": options}

    async def invoke_model(state: StepState) -> dict:
        options = state["choice_options"]
        forced = len(options) == 1
        payload = _request(model.model_name, state["messages"], options, retry=state["attempts"] > 0)
        input_size = _input_size(payload)
        if input_size + context_settings.actor_output_tokens > context_settings.token_budget:
            raise ContextLimitError("choice request and reserved output exceed context budget")
        if budget and not forced:
            budget.start_call()
        metadata = {
            "decision_source": "forced" if forced else "system_one",
            "rationale_source": "actor_adapter", "question": payload["questions"]["next_action"],
            "option_order": list(options),
        }
        call_id = repository.start_model_call(
            state["session_id"], payload["state"]["messages"],
            observation_event_id=state["observation_event_id"], memory_revision_id=state["memory_revision_id"],
            history_window=state["history_window"], input_token_count=None if forced else input_size,
            input_token_count_method=None if forced else "utf8_byte_estimate",
            decision_metadata=metadata,
        )
        started = time.monotonic()
        try:
            response = {} if forced else await within_budget(budget, lambda: model.decide(payload))
        except (httpx.HTTPError, BudgetStop) as exc:
            error = f"HTTP {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else type(exc).__name__
            repository.finish_model_call(
                call_id, "error", validation_error=error, duration_ms=round((time.monotonic() - started) * 1000),
            )
            raise
        return {
            "choice_response": response, "choice_metadata": metadata, "call_id": call_id,
            "duration_ms": round((time.monotonic() - started) * 1000), "attempts": state["attempts"] + int(not forced),
        }

    def validate(state: StepState) -> dict:
        options = state["choice_options"]
        forced = len(options) == 1
        response = state["choice_response"]
        answers = response.get("answers")
        answer = answers.get("next_action") if isinstance(answers, dict) else None
        choice = answer.get("choice") if isinstance(answer, dict) and answer.get("type") == "choice" else None
        if forced:
            choice = next(iter(options))

        decision = None
        error = None
        if isinstance(choice, str) and choice in options:
            decision = Decision(action=options[choice][1], rationale=f"Selected {choice}.")
            try:
                validate_mode(decision, state["observation"].mode)
            except ValueError as exc:
                error = str(exc)
        else:
            error = "next_action did not select an offered choice"

        usage = response.get("usage")
        usage = usage.copy() if isinstance(usage, dict) else None
        if usage is not None and "total_tokens" not in usage:
            input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
            if type(input_tokens) is int and type(output_tokens) is int and min(input_tokens, output_tokens) >= 0:
                usage["total_tokens"] = input_tokens + output_tokens
        provider = {key: response[key] for key in ("id", "model", "provider") if key in response}
        metadata = {**state["choice_metadata"], **provider}
        if isinstance(answer, dict):
            metadata["answer"] = {
                key: answer[key] for key in ("type", "choice", "probabilities", "confidence") if key in answer
            }
        elif forced:
            metadata["answer"] = {"type": "choice", "choice": choice}
        repository.finish_model_call(
            state["call_id"], "invalid" if error else "valid", raw_response=None if forced else response,
            parsed_action=decision.action.model_dump(mode="json", exclude_none=True) if decision and not error else None,
            decision_rationale=decision.rationale if decision and not error else None, validation_error=error,
            response_metadata=provider or None, usage_metadata=usage, duration_ms=state["duration_ms"],
            decision_metadata=metadata,
        )
        if error:
            if forced or state["attempts"] >= max_model_calls:
                raise DecisionValidationError(error)
            return {"action": None}
        assert decision is not None
        return {"action": decision.action, "rationale": decision.rationale, "request_goal_review": False}

    return context, invoke_model, validate
