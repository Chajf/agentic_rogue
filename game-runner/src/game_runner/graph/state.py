"""State passed through one graph invocation."""

from typing import NotRequired, TypedDict
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage

from game_runner.clients.rogue import ActionRequest, KeyActionRequest, Observation


class StepState(TypedDict):
    session_id: UUID
    observation: Observation
    attempts: int
    messages: NotRequired[list[BaseMessage]]
    response: NotRequired[AIMessage]
    call_id: NotRequired[int]
    duration_ms: NotRequired[int]
    action: NotRequired[ActionRequest | KeyActionRequest | None]
    rationale: NotRequired[str]
    next_observation: NotRequired[Observation]
