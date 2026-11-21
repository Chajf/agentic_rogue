"""Render the observed effects of a selected window of Rogue actions."""

import json

from game_runner.context.progress import COMBAT
from game_runner.context.transitions import DecisionTransition
from game_runner.prompts.compact_history import COMPACT_HISTORY_HEADER

FEEDBACK_LIMIT = 120
EFFECT_LABELS = {
    "combat_observed": "combat observed",
    "level_changed": "level changed",
    "observed_relocation": "observed relocation",
    "quiet_failed_move": "no displacement or feedback",
    "uncertain": "action effect uncertain",
}


def _action_text(action: dict) -> str:
    if action["type"] == "key":
        return "key=" + json.dumps(action["key"])
    return " ".join([
        action["action"],
        *(f"{name}={action[name]}" for name in ("item", "direction", "hand") if action.get(name) is not None),
    ])


def _position_text(progress: dict, *, scoped: bool) -> str:
    position = progress.get("position")
    if position is None:
        return "?"
    coords = f"[{position[0]},{position[1]}]"
    return f"map {progress['map_event_id']} {coords}" if scoped else coords


def _effect(delta: dict) -> str:
    effect = (delta.get("attempt") or {}).get("effect")
    if effect == "confirmed_move":
        count = (delta.get("visit") or {}).get("count")
        if count is None:
            return "confirmed move"
        return "confirmed first visit" if count == 1 else "confirmed revisit"
    return EFFECT_LABELS.get(effect, "")


def _feedback(messages: list[str], *, combat: bool) -> str:
    feedback = [" ".join(message.split()) for message in messages if message.strip()]
    if not feedback:
        return ""
    message = next((line for line in feedback if COMBAT.search(line)), feedback[0]) if combat else feedback[0]
    if len(message) > FEEDBACK_LIMIT:
        message = message[:FEEDBACK_LIMIT - 1] + "…"
    return "game " + json.dumps(message, ensure_ascii=False)


def _row(transition: DecisionTransition) -> str:
    after = transition.output_observation
    prefix = f"{after.api_step} {_action_text(transition.action)}"
    if transition.output_outcome == "uncertain":
        return f"{prefix} ? -> ? | action response lost, effect unknown"

    trusted_input = transition.input_outcome == "succeeded"
    before_progress = (transition.input_progress or {}) if trusted_input else {}
    progress = transition.progress or {}
    from_map, to_map = before_progress.get("map_event_id"), progress.get("map_event_id")
    map_changed = from_map is not None and to_map is not None and from_map != to_map
    origin = _position_text(before_progress, scoped=map_changed)
    destination = _position_text(progress, scoped=map_changed)
    effect = _effect(progress.get("working_state_delta") or {}) if trusted_input else "action effect uncertain"
    if map_changed:
        effect = "map changed"

    details = [effect] if effect else []
    if trusted_input:
        for label, key in (("HP", "hp"), ("depth", "level")):
            if (change := progress.get(key)) and change[0] != change[1]:
                details.append(f"{label} {change[0]} -> {change[1]}")
    if after.mode != transition.input_observation.mode:
        details.append(f"mode {after.mode}")
    feedback = _feedback(after.messages, combat=effect == "combat observed")
    if feedback:
        details.append(feedback)
    return f"{prefix} {origin} -> {destination} | " + ", ".join(details or ["response observed"])


def render_compact_history(history: list[DecisionTransition]) -> str:
    if not history:
        return ""
    lines = [COMPACT_HISTORY_HEADER]
    first = history[0]
    if first.input_outcome == "uncertain":
        lines.append(f"History boundary at API step {first.input_observation.api_step}. The prior action response was lost.")
    lines.extend(_row(transition) for transition in history)
    return "\n".join(lines) + "\n\n"
