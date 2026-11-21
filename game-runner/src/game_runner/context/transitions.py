"""Read-only trace types shared by gameplay and offline analysis."""

from dataclasses import dataclass
from typing import Literal, TypedDict

from game_runner.clients.rogue import Observation
from game_runner.config import HistoryFormat

ObservationOutcome = Literal["succeeded", "uncertain"]


class HistoryWindow(TypedDict):
    format: HistoryFormat
    eligible_output_event_ids: list[int]
    delivered_output_event_ids: list[int]


@dataclass(frozen=True)
class DecisionTransition:
    input_event_id: int
    input_observation: Observation
    input_outcome: ObservationOutcome
    model_call_id: int
    action: dict
    rationale: str
    output_event_id: int
    output_observation: Observation
    output_outcome: ObservationOutcome
    progress: dict | None = None
    input_progress: dict | None = None
