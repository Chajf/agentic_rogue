"""Derive episode-scoped facts from the observations shown by Rogue."""

import re
from dataclasses import dataclass, replace

from game_runner.clients.rogue import Observation

DIRECTIONS = dict(zip("kjhlyubn", (
    "UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT",
), strict=True))
SLOT = re.compile(r"^\s*([a-z])\)\s+(.+?)\s*$")
COMBAT = re.compile(r"\b(hit|hits|miss|misses|defeat|defeated|killed|attack|attacks|damage)\b", re.I)
INVENTORY_SAFE_ACTIONS = {"INVENTORY", "CANCEL", "SEARCH", "REST"}
Position = tuple[int, int]


def visible_position(observation: Observation) -> Position | None:
    if observation.mode != "normal" or observation.terminated or observation.state.dungeon_level is None:
        return None
    if any(SLOT.fullmatch(line) for line in observation.screen):
        return None
    if any("press return to continue" in line.lower() for line in observation.screen):
        return None
    positions = [
        (row, col) for row, line in enumerate(observation.screen[1:-1], 1)
        for col, value in enumerate(line) if value == "@"
    ]
    if len(positions) != 1:
        return None
    row, col = positions[0]
    neighbors = observation.screen[max(1, row - 1):row + 2]
    if any(set(line[max(0, col - 1):col + 2]) & set(".#%+|-") for line in neighbors):
        return row, col
    return None


def movement_direction(action: dict) -> str | None:
    if action["type"] == "key":
        return DIRECTIONS.get(action["key"])
    name = action["action"]
    direction = name.removeprefix("MOVE_")
    if name.startswith("MOVE_") and direction in DIRECTIONS.values():
        return direction
    return None


def changed(before, after):
    return (before, after) if None not in (before, after) else None


@dataclass(frozen=True)
class InventorySnapshot:
    event_id: int
    slots: tuple[tuple[str, str], ...]
    stale: bool = False


@dataclass(frozen=True)
class Progress:
    event_id: int
    map_event_id: int | None
    map_level: int | None
    position: Position | None
    inventory: InventorySnapshot | None = None
    direction: str | None = None
    no_movement_streak: int = 0
    position_change: str | None = None
    level: tuple[int, int] | None = None
    hp: tuple[int, int] | None = None
    hunger: tuple[str, str] | None = None
    messages: tuple[str, ...] | None = None

    @classmethod
    def initial(cls, event_id: int, observation: Observation) -> "Progress":
        position = visible_position(observation)
        return cls(
            event_id, event_id if position is not None else None,
            observation.state.dungeon_level, position,
        )

    def advance(
        self, event_id: int, before: Observation, action: dict, after: Observation, *, recovered: bool,
    ) -> "Progress":
        position = visible_position(after)
        level = after.state.dungeon_level
        # Menus can hide the status line. Carry the last known level but compare
        # positions only between adjacent unambiguous map observations.
        name = action.get("action")
        key = action.get("key")
        explicit_inventory = before.mode == "normal" and (name == "INVENTORY" or key == "i")
        if explicit_inventory:
            position = None
        transition_unknown = level is None and (name in {"ASCEND", "DESCEND"} or key in {"<", ">"})
        previous_level = self.map_level
        map_changed = recovered or transition_unknown or (
            previous_level is not None and level is not None and previous_level != level
        ) or any(
            "you fell into a trap!" in msg.lower() or "trap door" in msg.lower()
            for msg in after.messages
        )
        map_id = None if map_changed else self.map_event_id
        if map_id is None and position is not None:
            map_id = event_id
        if transition_unknown or (recovered and level is None):
            previous_level = None

        messages = None if recovered else tuple(after.messages)
        position_change = None
        if not map_changed and self.position is not None and position is not None and map_id is not None:
            position_change = "same" if self.position == position else "moved"
        stats_delta = (
            changed(before.state.dungeon_level, level),
            changed(before.state.hp, after.state.hp),
            changed(before.state.hunger, after.state.hunger),
        ) if not recovered else (None, None, None)

        direction = movement_direction(action) if before.mode == after.mode == "normal" else None
        # Messages and stat changes can signal combat or another effect even
        # when the player glyph stayed at the same coordinates.
        quiet = messages is not None and not messages and not any(
            delta and delta[0] != delta[1] for delta in stats_delta
        ) and before.state == after.state
        no_movement = direction is not None and position_change == "same" and quiet
        streak = (self.no_movement_streak if self.direction == direction else 0) + 1 if no_movement else 0

        inventory = self.inventory
        slots = {
            match[1]: match[2] for line in [*after.messages, *after.screen]
            if (match := SLOT.fullmatch(line))
        } if explicit_inventory and not recovered else {}
        if slots:
            inventory = InventorySnapshot(event_id, tuple(sorted(slots.items())))
        elif inventory is not None and (
            recovered or explicit_inventory or name not in INVENTORY_SAFE_ACTIONS
            or before.mode != "normal" or any(COMBAT.search(message) for message in after.messages)
        ):
            inventory = replace(inventory, stale=True)

        return Progress(
            event_id, map_id, level if level is not None else previous_level, position, inventory,
            direction if no_movement else None, streak, position_change, *stats_delta, messages,
        )
