"""Replay completed decisions and keep the current screen intact."""

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from game_runner.clients.rogue import Observation
from game_runner.persistence.repository import DecisionTransition, ObservationOutcome
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
    observation: Observation, history: list[DecisionTransition], token_budget: int,
    *, observation_outcome: ObservationOutcome = "succeeded",
) -> list[BaseMessage]:
    current = _observation_text(observation, observation_outcome, current=True)
    # Tokenizers differ by local model. Budget history with the existing character
    # estimate while always keeping the system instruction and current screen.
    history_budget = max(0, token_budget * 3 - len(SYSTEM_INSTRUCTION) - len(current))
    selected: list[tuple[HumanMessage, AIMessage]] = []
    for transition in reversed(history):
        before = _observation_text(transition.input_observation, transition.input_outcome)
        decision = json.dumps(
            {"action": transition.action, "rationale": transition.rationale},
            ensure_ascii=False, separators=(",", ":"),
        )
        pair_size = len(before) + len(decision) + 2
        if pair_size > history_budget:
            break
        selected.append((HumanMessage(content=before), AIMessage(content=decision)))
        history_budget -= pair_size

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_INSTRUCTION)]
    for before, decision in reversed(selected):
        messages.extend((before, decision))
    messages.append(HumanMessage(content=current))
    return messages
