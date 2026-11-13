"""Read-only trace types shared by gameplay and offline analysis."""

from dataclasses import dataclass
from typing import Literal

from game_runner.clients.rogue import Observation

ObservationOutcome = Literal["succeeded", "uncertain"]


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
