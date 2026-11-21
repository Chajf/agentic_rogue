"""Bounded projections of the active map for the actor and goal selector."""

from collections import deque
from dataclasses import asdict, dataclass
from itertools import product

from game_runner.context.progress import Position
from game_runner.context.working_state import MONSTERS, RESOURCES, Sighting, WorkingState, coordinates

RADIUS = 3
WALKABLE = frozenset(".#+%")
BLOCKED = frozenset("|-^")
CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _distance(first: Position, second: Position) -> int:
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def _source(sighting: Sighting, state: WorkingState) -> dict:
    return {**asdict(sighting.source), "age": state.through.api_step - sighting.source.api_step}


def actor_view(state: WorkingState) -> dict | None:
    """Project observed terrain, visits, stairs, and attempts from the current map."""
    player = state.position
    if player is None:
        return None

    row, col = player
    nearby = list(product(range(row - RADIUS, row + RADIUS + 1), range(col - RADIUS, col + RADIUS + 1)))

    def by_distance(position: Position):
        return _distance(position, player), position

    tiles = []
    for position in sorted(nearby, key=by_distance):
        terrain = state.terrain.get(position)
        visit = state.visits.get(position)
        if terrain is None and visit is None:
            continue
        tiles.append({
            "position": coordinates(position),
            "terrain": terrain.glyph if terrain else None,
            "source": _source(terrain, state) if terrain else None,
            "newly_seen": terrain is not None and terrain.first_seen == state.through,
            "visits": visit.count if visit else 0,
            "last_visit_event_id": visit.last.event_id if visit else None,
        })
        if len(tiles) == 16:
            break

    stairs = sorted((pos for pos, sighting in state.terrain.items() if sighting.glyph == "%"), key=by_distance)
    attempts = sorted(
        (attempt for position in nearby for group in state.attempts.get(position, {}).values() for attempt in group),
        key=lambda attempt: attempt.output.event_id,
    )
    return {
        "map_instance": state.map_instance,
        "player": coordinates(player),
        "through_event_id": state.through.event_id,
        "tiles": tiles,
        "landmarks": [
            {"kind": "stairs", "position": coordinates(pos), "source": _source(state.terrain[pos], state)}
            for pos in stairs[:2]
        ],
        "recent_attempts": [attempt.record() for attempt in attempts[-4:]],
    }


@dataclass(frozen=True)
class RouteStep:
    parent: Position | None
    distance: int
    inferred: bool


def _cardinal_neighbors(position: Position):
    row, col = position
    return ((row + dy, col + dx) for dy, dx in CARDINAL)


def _routes(state: WorkingState, player: Position) -> dict[Position, RouteStep]:
    blocked = {pos for pos, sighting in state.terrain.items() if sighting.glyph in BLOCKED}
    blocked.update(
        pos for pos, entity in state.entities.items() if entity.source == state.through and entity.glyph in MONSTERS
    )
    walkable = {pos for pos, sighting in state.terrain.items() if sighting.glyph in WALKABLE} - blocked
    confirmed: dict[Position, set[Position]] = {}
    for origin, destination in state.edges:
        if destination not in blocked:
            confirmed.setdefault(origin, set()).add(destination)
    routes = {player: RouteStep(None, 0, False)}
    queue = deque([player])
    while queue:
        origin = queue.popleft()
        prior = routes[origin]
        traversed = confirmed.get(origin, set())
        # Infer only cardinal steps between supported cells. The player is an
        # observed starting anchor even when @ hides the underlying terrain.
        inferred = walkable.intersection(_cardinal_neighbors(origin)) if origin in walkable or origin == player else set()
        for destination in sorted(traversed | inferred, key=lambda pos: (pos not in traversed, pos)):
            if destination not in routes:
                routes[destination] = RouteStep(origin, prior.distance + 1, prior.inferred or destination not in traversed)
                queue.append(destination)
    return routes


def _route_record(position: Position, routes: dict[Position, RouteStep]) -> dict:
    step = routes.get(position)
    if step is None:
        return {"reachability": "unknown"}
    record = {"reachability": "inferred" if step.inferred else "confirmed", "distance": step.distance}
    if step.distance <= 12:
        path = []
        cursor: Position | None = position
        while cursor is not None:
            path.append(coordinates(cursor))
            cursor = routes[cursor].parent
        record["route"] = list(reversed(path))
    return record


def _borders_unknown(state: WorkingState, position: Position) -> bool:
    return any(
        1 <= row <= 22 and 0 <= col < 80
        and (row, col) not in state.terrain and (row, col) not in state.visits and (row, col) not in state.entities
        for row, col in _cardinal_neighbors(position)
    )


def goal_view(state: WorkingState) -> dict | None:
    """Project sourced opportunities for a future model-selected goal."""
    player = state.position
    if player is None:
        return None
    routes = _routes(state, player)
    attempted_targets = {
        attempt.target for groups in state.attempts.values() for group in groups.values() for attempt in group
    }
    candidates = []
    for position, terrain in state.terrain.items():
        if terrain.glyph == "%":
            candidates.append((position, "stairs", "stairs_sighting", terrain))
        elif position in routes and terrain.glyph in WALKABLE:
            if terrain.glyph == "+" and position not in state.visits and position not in attempted_targets:
                candidates.append((position, "explore", "untried_door", terrain))
            elif _borders_unknown(state, position):
                candidates.append((position, "explore", "frontier", terrain))
    candidates.extend(
        (position, "collect", "resource_sighting", entity)
        for position, entity in state.entities.items() if entity.glyph in RESOURCES and entity.source == state.through
    )
    candidates.sort(key=lambda item: (
        item[0] not in routes, routes[item[0]].distance if item[0] in routes else 0, item[0], item[1],
    ))
    return {
        "map_instance": state.map_instance,
        "player": coordinates(player),
        "through_event_id": state.through.event_id,
        "opportunities": [
            {
                "kind": kind,
                "basis": basis,
                "position": coordinates(position),
                "glyph": sighting.glyph,
                "source": _source(sighting, state),
                "applicability": {"map_instance": state.map_instance, "current_sighting_required": kind == "collect"},
                **_route_record(position, routes),
            }
            for position, kind, basis, sighting in candidates[:6]
        ],
    }
