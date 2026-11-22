"""Render chronological replay and the full current observation."""

from dataclasses import asdict
import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from game_runner.clients.rogue import Observation
from game_runner.config import ContextVariant, HistoryFormat
from game_runner.context.compact_history import render_compact_history
from game_runner.context.decision_support import neighbors_suffix
from game_runner.context.memory import MemoryRevision
from game_runner.context.progress import Progress, visible_position
from game_runner.context.transitions import DecisionTransition, ObservationOutcome
from game_runner.prompts.goals import ACTOR_GOAL_GUIDANCE, GOAL_SELECTOR_INSTRUCTION
from game_runner.prompts.system import ACTOR_PROMPTS, PromptProfile


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
    uncovered: list[dict] | None = None, context_variant: ContextVariant = "baseline",
    prompt_profile: PromptProfile = "legacy", working_view: dict | None = None,
    history_format: HistoryFormat = "replay", goal_context: dict | None = None,
) -> list[BaseMessage]:
    messages: list[BaseMessage] = [SystemMessage(content=ACTOR_PROMPTS[prompt_profile].instruction)]
    if history_format == "replay":
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
    if context_variant == "neighbors" and observation_outcome == "succeeded":
        position = visible_position(observation)
        if position is not None and (progress is None or progress.position == position):
            current += neighbors_suffix(observation.screen, position)
    if working_view is not None:
        current += (
            "\nObserved working state. Coordinates are zero-based [row, column]. "
            "Null terrain is unknown. Age is in API steps.\n"
            + json.dumps(working_view, ensure_ascii=False, separators=(",", ":"))
        )
    if goal_context is not None:
        current += "\n" + ACTOR_GOAL_GUIDANCE + "\n" + json.dumps(goal_context, separators=(",", ":"))
    if history_format == "compact":
        current = render_compact_history(history) + current
    messages.append(HumanMessage(content=current))
    return messages


def build_goal_selection_messages(observation: Observation, evidence: dict) -> list[BaseMessage]:
    current = _observation_text(observation, "succeeded", current=True)
    current += "\nGoal evidence:\n" + json.dumps(evidence, ensure_ascii=False, separators=(",", ":"))
    return [SystemMessage(content=GOAL_SELECTOR_INSTRUCTION), HumanMessage(content=current)]
