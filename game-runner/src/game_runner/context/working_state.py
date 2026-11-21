"""Reduce confirmed Rogue observations into episode-local evidence."""

from collections import deque
from dataclasses import dataclass, field
from uuid import UUID

from game_runner.clients.rogue import Observation
from game_runner.context.decision_support import MOVE_OFFSETS
from game_runner.context.progress import COMBAT, Position, Progress, movement_direction
from game_runner.context.transitions import DecisionTransition, ObservationOutcome

TERRAIN = frozenset(".#+|-^%")
RESOURCES = frozenset("*!?:)]=/,")
MONSTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class UnlinkedObservation(ValueError):
    """An observation does not follow the working state's confirmed trace."""


def coordinates(position: Position | None) -> list[int] | None:
    return list(position) if position is not None else None


@dataclass(frozen=True)
class Evidence:
    event_id: int
    api_step: int


@dataclass(frozen=True)
class Sighting:
    glyph: str
    source: Evidence
    first_seen: Evidence


@dataclass(frozen=True)
class Visits:
    count: int
    last: Evidence


@dataclass(frozen=True)
class Attempt:
    action: dict
    map_instance: int | None
    origin: Position | None
    direction: str | None
    target: Position | None
    destination: Position | None
    effect: str
    input: Evidence
    output: Evidence
    hp: tuple[int, int] | None = None
    level: tuple[int, int] | None = None

    def record(self) -> dict:
        row = {
            "action": self.action.copy(),
            "map_instance": self.map_instance,
            "from": coordinates(self.origin),
            "direction": self.direction,
            "target": coordinates(self.target),
            "to": coordinates(self.destination),
            "effect": self.effect,
            "input_event_id": self.input.event_id,
            "output_event_id": self.output.event_id,
        }
        if self.hp and self.hp[0] != self.hp[1]:
            row["hp"] = list(self.hp)
        if self.level and self.level[0] != self.level[1]:
            row["level"] = list(self.level)
        return row


@dataclass
class WorkingState:
    episode_id: UUID
    through: Evidence
    outcome: ObservationOutcome = "succeeded"
    map_instance: int | None = None
    position: Position | None = None
    terrain: dict[Position, Sighting] = field(default_factory=dict)
    entities: dict[Position, Sighting] = field(default_factory=dict)
    visits: dict[Position, Visits] = field(default_factory=dict)
    edges: dict[tuple[Position, Position], Evidence] = field(default_factory=dict)
    attempts: dict[tuple[Position, str], deque[Attempt]] = field(default_factory=dict)
    _last_position: Position | None = None

    @classmethod
    def from_reset(cls, event_id: int, observation: Observation, progress: Progress) -> tuple["WorkingState", dict]:
        if observation.api_step != 0 or progress.event_id != event_id:
            raise UnlinkedObservation("working state requires the linked reset at API step zero")
        state = cls(observation.episode_id, Evidence(event_id, 0))
        state._enter_map(progress.map_event_id)
        delta = state._delta(None, progress.position)
        state._observe(observation, progress, delta)
        return state, delta

    def advance(self, transition: DecisionTransition, progress: Progress) -> dict:
        """Apply a linked transition and return the JSON-ready delta for its output event."""
        before, after = transition.input_observation, transition.output_observation
        if (
            transition.input_event_id != self.through.event_id
            or transition.input_outcome != self.outcome
            or transition.output_event_id <= transition.input_event_id
            or transition.output_event_id != progress.event_id
            or before.api_step != self.through.api_step
            or after.api_step != before.api_step + 1
            or before.episode_id != self.episode_id
            or after.episode_id != self.episode_id
            or before.seed != after.seed
        ):
            raise UnlinkedObservation("working-state transition is not linked to the current observation")

        output = Evidence(transition.output_event_id, after.api_step)
        gap = transition.output_outcome == "uncertain"
        map_instance = None if gap else progress.map_event_id
        position = None if gap or map_instance is None else progress.position
        same_map = map_instance is not None and map_instance == self.map_instance
        attempt = self._attempt(transition, progress, output, same_map=same_map, position=position)

        self.through = output
        self.outcome = transition.output_outcome
        self._enter_map(map_instance)
        delta = self._delta(transition.input_event_id, position)
        delta["attempt"] = attempt.record()
        if gap:
            delta["gap"] = "action_response_lost"
        else:
            self._observe(after, progress, delta)
        if same_map and attempt.origin is not None:
            action_group = attempt.direction or attempt.action.get("action") or attempt.action["key"]
            self.attempts.setdefault((attempt.origin, action_group), deque(maxlen=2)).append(attempt)
            if attempt.effect == "confirmed_move":
                assert attempt.destination is not None
                edge = (attempt.origin, attempt.destination)
                if edge not in self.edges:
                    self.edges[edge] = output
                    delta["edge"] = [coordinates(attempt.origin), coordinates(attempt.destination)]
        return delta

    def _enter_map(self, map_instance: int | None) -> None:
        if map_instance == self.map_instance:
            return
        self.map_instance = map_instance
        self.position = self._last_position = None
        self.terrain.clear()
        self.entities.clear()
        self.visits.clear()
        self.edges.clear()
        self.attempts.clear()

    def _delta(self, input_event_id: int | None, position: Position | None) -> dict:
        return {
            "schema_version": 1,
            "input_event_id": input_event_id,
            "event_id": self.through.event_id,
            "api_step": self.through.api_step,
            "map_instance": self.map_instance,
            "player": coordinates(position),
        }

    def _observe(self, observation: Observation, progress: Progress, delta: dict) -> None:
        self.position = progress.position if self.map_instance is not None else None
        if self.position is None:
            return
        for row, line in enumerate(observation.screen[1:-1], 1):
            for col, glyph in enumerate(line):
                position = (row, col)
                if glyph in TERRAIN:
                    self._sight(self.terrain, glyph, position, delta, "terrain")
                    if position in self.entities:
                        del self.entities[position]
                        delta.setdefault("entities", []).append({"position": coordinates(position), "glyph": None})
                elif glyph in MONSTERS or glyph in RESOURCES:
                    self._sight(self.entities, glyph, position, delta, "entities")
        if self.position != self._last_position:
            previous = self.visits.get(self.position)
            count = 1 if previous is None else previous.count + 1
            self.visits[self.position] = Visits(count, self.through)
            self._last_position = self.position
            delta["visit"] = {"position": coordinates(self.position), "count": count}

    def _sight(
        self, records: dict[Position, Sighting], glyph: str, position: Position, delta: dict, field_name: str,
    ) -> None:
        previous = records.get(position)
        first_seen = previous.first_seen if previous is not None else self.through
        records[position] = Sighting(glyph, self.through, first_seen)
        if previous is not None and previous.glyph == glyph:
            delta.setdefault(f"{field_name}_seen", []).append(coordinates(position))
        else:
            delta.setdefault(field_name, []).append({"position": coordinates(position), "glyph": glyph})

    def _attempt(
        self, transition: DecisionTransition, progress: Progress, output: Evidence, *,
        same_map: bool, position: Position | None,
    ) -> Attempt:
        action = transition.action.copy()
        origin = self.position
        direction = movement_direction(action) if origin is not None else None
        target = None
        if direction is not None and origin is not None:
            dy, dx = MOVE_OFFSETS[direction]
            target = (origin[0] + dy, origin[1] + dx)

        destination = position if same_map and origin is not None else None
        confirmed = transition.input_outcome == transition.output_outcome == "succeeded"
        if not confirmed:
            effect = "uncertain"
        elif progress.level and progress.level[0] != progress.level[1]:
            effect = "level_changed"
        elif destination is None:
            effect = "uncertain"
        elif destination != origin:
            effect = "confirmed_move" if destination == target else "observed_relocation"
        elif direction and progress.no_movement_streak:
            effect = "quiet_failed_move"
        else:
            monster = self.entities.get(target) if target is not None else None
            combat = (
                monster is not None and monster.source == self.through and monster.glyph in MONSTERS
                and any(COMBAT.search(message) for message in progress.messages or ())
            )
            effect = "combat_observed" if combat else "other"
        return Attempt(
            action, self.map_instance, origin, direction, target, destination, effect, self.through, output,
            hp=progress.hp if confirmed else None, level=progress.level if confirmed else None,
        )


def replay_working_state(
    reset_event_id: int, reset: Observation, transitions: list[DecisionTransition],
) -> tuple[WorkingState, list[dict]]:
    """Reconstruct working state and deltas directly from observations and accepted actions."""
    progress = Progress.initial(reset_event_id, reset)
    state, delta = WorkingState.from_reset(reset_event_id, reset, progress)
    deltas = [delta]
    for transition in transitions:
        progress = progress.advance(
            transition.output_event_id, transition.input_observation, transition.action,
            transition.output_observation, recovered=transition.output_outcome == "uncertain",
        )
        deltas.append(state.advance(transition, progress))
    return state, deltas
