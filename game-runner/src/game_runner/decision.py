"""Shared, side-effect-free decision contract and mode validation."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from game_runner.clients.rogue import ActionRequest, KeyActionRequest
from game_runner.responses import decode_json


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Annotated[ActionRequest | KeyActionRequest, Field(discriminator="type")]
    rationale: str = Field(min_length=1, max_length=300)


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


def parse_decision(content: object, mode: str) -> Decision:
    decision = Decision.model_validate(decode_json(content))
    validate_mode(decision, mode)
    return decision
