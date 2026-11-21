"""Render decision cues from observations already available to the actor."""

import json

from .progress import Position


NEIGHBORS_PREFIX = "\nVisible neighbors (null means unknown): "

GROUNDING_INSTRUCTION = (
    "Before choosing a move, check which visible tile the direction targets. "
    "Check the target again after a failed attempt. For DESCEND, check that the player occupies the stair tile."
)
MOVE_OFFSETS = {
    "UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1),
    "UP_LEFT": (-1, -1), "UP_RIGHT": (-1, 1), "DOWN_LEFT": (1, -1), "DOWN_RIGHT": (1, 1),
}


def visible_neighbors(screen: list[str], position: Position) -> dict[str, str | None]:
    row, col = position
    neighbors = {}
    for name, (dy, dx) in MOVE_OFFSETS.items():
        y, x = row + dy, col + dx
        glyph = screen[y][x] if 0 < y < len(screen) - 1 and 0 <= x < len(screen[y]) else " "
        neighbors[name] = glyph if glyph != " " else None
    return neighbors

def neighbors_suffix(screen: list[str], position: Position) -> str:
    neighbors = json.dumps(visible_neighbors(screen, position), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return NEIGHBORS_PREFIX + neighbors


def stair_sighting(position: Position, event_id: int) -> str:
    row, col = position
    return f"Stairs (%) last seen at row {row}, column {col} on this map (observation {event_id})."
