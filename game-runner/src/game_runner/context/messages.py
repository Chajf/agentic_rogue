"""Keep the current screen and concise action history in model context."""

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from game_runner.clients.rogue import Observation
from game_runner.prompts.system import SYSTEM_INSTRUCTION


def _history_entry(row: dict) -> str:
    observed = row["observation"]
    action = row["action_request"]
    state = observed.get("state") or {}
    return json.dumps(
        {
            "action": action,
            "messages": observed.get("messages", []),
            "mode": observed.get("mode"),
            "level": state.get("dungeon_level"),
            "hp": state.get("hp"),
            "status": observed.get("status"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_messages(
    observation: Observation, history: list[dict], token_budget: int,
) -> list[BaseMessage]:
    current = (
        f"Current API step: {observation.api_step}; status: {observation.status}; "
        f"interaction mode: {observation.mode}.\n"
        f"Latest game messages: {json.dumps(observation.messages, ensure_ascii=False)}\n"
        f"Stats: {observation.state.model_dump_json(exclude_none=True)}\n"
        "Current 80x24 screen (spaces are significant):\n"
        + "\n".join(observation.screen)
    )
    # Tokenizers differ by local model. Reserve room by using a conservative
    # character estimate; the current observation is always kept intact.
    history_budget = max(0, token_budget * 3 - len(SYSTEM_INSTRUCTION) - len(current))
    selected: list[str] = []
    for row in reversed(history):
        entry = _history_entry(row)
        if len(entry) + 1 > history_budget:
            break
        selected.append(entry)
        history_budget -= len(entry) + 1
    selected.reverse()
    if selected:
        current = "Recent actions and effects (oldest first):\n" + "\n".join(selected) + "\n\n" + current
    return [SystemMessage(content=SYSTEM_INSTRUCTION), HumanMessage(content=current)]
