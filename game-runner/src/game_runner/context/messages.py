"""Render chronological replay and the full current observation."""

from dataclasses import asdict
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from game_runner.clients.rogue import Observation
from game_runner.context.memory import MemoryRevision
from game_runner.context.progress import Progress
from game_runner.context.transitions import DecisionTransition, ObservationOutcome
from game_runner.prompts.system import SYSTEM_INSTRUCTION


def _observation_text(
    observation: Observation, outcome: ObservationOutcome, *, current: bool = False,
) -> str:
    label = "Current API step" if current else "API step"
    text = (
        f"{label}: {observation.api_step}; status: {observation.status}; "
        f"interaction mode: {observation.mode}.\n"
        f"Latest game messages: {json.dumps(observation.messages, ensure_ascii=False)}\n"
        f"Stats: {observation.state.model_dump_json(exclude_none=True)}"
    )
    if outcome == "uncertain":
        text += "\nThis observation was recovered after an action timeout. The action response was lost."
    if current:
        text += "\nCurrent 80x24 screen (spaces are significant):\n" + "\n".join(observation.screen)
    return text


def build_messages(
    observation: Observation, history: list[DecisionTransition], *,
    observation_outcome: ObservationOutcome = "succeeded",
    progress: Progress | None = None, memory: MemoryRevision | None = None,
    uncovered: list[dict] | None = None,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_INSTRUCTION)]
    for transition in history:
        messages.extend((
            HumanMessage(content=_observation_text(transition.input_observation, transition.input_outcome)),
            AIMessage(content=json.dumps(
                {"action": transition.action, "rationale": transition.rationale},
                ensure_ascii=False, separators=(",", ":"),
            )),
        ))

    current = _observation_text(observation, observation_outcome, current=True)
    if progress is not None:
        current += "\nObserved progress (null means unknown): " + json.dumps(asdict(progress), ensure_ascii=False)
    if memory is not None:
        current += (
            f"\nEpisode memory through event {memory.through_event_id} "
            f"(historical evidence, {len(memory.gap_event_ids)} recovery gaps): "
            + memory.content.model_dump_json()
        )
    if uncovered:
        current += "\nUncompacted episode evidence (oldest first): " + json.dumps(
            uncovered, ensure_ascii=False, separators=(",", ":"),
        )
    messages.append(HumanMessage(content=current))
    return messages
