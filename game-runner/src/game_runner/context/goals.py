"""Model-selected goals and validation against scoped observations."""

from dataclasses import asdict, dataclass, replace
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from game_runner.context.progress import Position
from game_runner.context.working_state import RESOURCES, WorkingState
from game_runner.context.working_views import WALKABLE, supported_route
from game_runner.responses import decode_json

GoalKind = Literal["explore", "reach", "descend", "collect", "retreat"]
GoalStatus = Literal["active", "suspended", "complete", "invalid", "superseded"]
Coordinate = tuple[Annotated[int, Field(strict=True, ge=1, le=22)], Annotated[int, Field(strict=True, ge=0, le=79)]]
EventID = Annotated[int, Field(strict=True, gt=0)]


class SelectorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectCandidate(SelectorResponse):
    choice: Literal["select"]
    candidate_id: Annotated[int, Field(strict=True, gt=0)]


class ProposeGoal(SelectorResponse):
    choice: Literal["propose"]
    kind: GoalKind
    target: Coordinate | None = None
    source_event_id: EventID | None = None


class GoalDisposition(SelectorResponse):
    choice: Literal["keep", "suspend", "defer"]


GoalReply = Annotated[SelectCandidate | ProposeGoal | GoalDisposition, Field(discriminator="choice")]
REPLY_ADAPTER = TypeAdapter(GoalReply)


@dataclass(frozen=True)
class Goal:
    id: int
    kind: GoalKind
    observation_event_id: int
    map_instance: int
    completion: str
    target: Position | None = None
    source_event_id: int | None = None
    resource_glyph: str | None = None
    status: GoalStatus = "active"
    stage: str = "pursuing"
    route: tuple[Position, ...] = ()
    route_index: int = 0
    reached_stairs: bool = False

    def record(self) -> dict:
        record = asdict(self)
        record["target"] = list(self.target) if self.target is not None else None
        record["route"] = [list(position) for position in self.route]
        return record

    def actor_record(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "status": self.status, "target": self.target,
            "completion": self.completion, "stage": self.stage,
        }


def parse_goal_reply(content: object) -> GoalReply:
    return REPLY_ADAPTER.validate_python(decode_json(content))


def target_supported(goal: Goal, state: WorkingState) -> bool:
    if goal.map_instance != state.map_instance:
        return False
    if goal.target is None:
        return True
    terrain = state.terrain.get(goal.target)
    if goal.kind == "descend":
        return terrain is not None and terrain.glyph == "%"
    if goal.kind == "collect":
        entity = state.entities.get(goal.target)
        return entity is not None and entity.glyph == goal.resource_glyph
    return terrain.glyph in WALKABLE if terrain else goal.target in state.visits


def select_goal(
    reply: GoalReply, previous: Goal | None, state: WorkingState, candidates: list[dict], next_id: int,
) -> Goal | None:
    if isinstance(reply, GoalDisposition):
        valid = previous is not None and previous.status in {"active", "suspended"} and target_supported(previous, state)
        if reply.choice == "defer":
            if valid and previous.status == "active":
                raise ValueError("defer requires no active goal")
            return previous
        if not valid:
            raise ValueError("no valid goal to keep or suspend")
        if reply.choice == "suspend" and previous.status != "active":
            raise ValueError("suspend requires an active goal")
        return replace(previous, status="active" if reply.choice == "keep" else "suspended")

    if isinstance(reply, SelectCandidate):
        candidate = next((entry for entry in candidates if entry["candidate_id"] == reply.candidate_id), None)
        if candidate is None:
            raise ValueError("candidate was not offered in this request")
        kind = "descend" if candidate["kind"] == "stairs" else candidate["kind"]
        reply = ProposeGoal(
            choice="propose", kind=kind, target=candidate["position"], source_event_id=candidate["source"]["event_id"],
        )
    return _new_goal(reply, state, next_id)


def _new_goal(reply: ProposeGoal, state: WorkingState, goal_id: int) -> Goal:
    kind, target, source = reply.kind, reply.target, reply.source_event_id
    assert state.map_instance is not None and state.position is not None
    if target is None:
        if kind != "explore" or source not in {None, state.through.event_id}:
            raise ValueError("only exploration can use the current observation with no target")
        return Goal(goal_id, kind, state.through.event_id, state.map_instance, "discover_terrain")

    terrain = state.terrain.get(target)
    visit = state.visits.get(target)
    entity = state.entities.get(target)
    if kind == "descend":
        if terrain is None or terrain.glyph != "%" or source != terrain.source.event_id:
            raise ValueError("descent requires a sourced stair tile on the current map")
        completion = "descend_at_target"
    elif kind == "collect":
        if entity is None or entity.glyph not in RESOURCES or entity.source != state.through or source != entity.source.event_id:
            raise ValueError("collection requires a currently visible, sourced resource")
        completion = "collect_at_target"
    else:
        sources = set()
        if terrain is not None and terrain.glyph in WALKABLE:
            sources.add(terrain.source.event_id)
        if visit is not None and (terrain is None or terrain.glyph in WALKABLE):
            sources.add(visit.last.event_id)
        if source is None or source not in sources or target == state.position:
            raise ValueError("target must be a sourced walkable destination on the current map")
        completion = "reach_target"
    return Goal(
        goal_id, kind, state.through.event_id, state.map_instance, completion,
        target=target, source_event_id=source, resource_glyph=entity.glyph if kind == "collect" else None,
        stage="at_stairs" if kind == "descend" and target == state.position else "pursuing",
        route=supported_route(state, target), reached_stairs=kind == "descend" and target == state.position,
    )
