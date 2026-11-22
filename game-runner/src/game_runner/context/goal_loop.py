"""Review and verify model-selected goals using only linked observations."""

from dataclasses import dataclass, field, replace

from game_runner.clients.rogue import Observation
from game_runner.context.goals import Goal, GoalReply, select_goal, target_supported
from game_runner.context.progress import Position
from game_runner.context.transitions import DecisionTransition
from game_runner.context.working_state import WorkingState
from game_runner.context.working_views import goal_view

COMPARABLE_MOVES = {"confirmed_move", "quiet_failed_move"}


@dataclass(frozen=True)
class GoalReview:
    reasons: dict[str, int]
    view: dict

    @property
    def candidates(self) -> list[dict]:
        return self.view["opportunities"]


@dataclass
class GoalLoop:
    same_edge_returns: int = 2
    movement_stall_actions: int = 6
    goal: Goal | None = None
    pending: dict[str, int] = field(default_factory=dict)
    last_review_map: int | None = None
    last_review_event: int | None = None
    _last_signature: tuple | None = None
    _last_candidates: set[tuple] = field(default_factory=set)
    _next_id: int = 1
    _comparable_moves: int = 0
    _edge_traversals: int = 0
    _last_edge: tuple[Position, Position] | None = None

    def assess(self, observation: Observation, state: WorkingState) -> GoalReview | None:
        if state.position is None or state.outcome != "succeeded" or observation.mode != "normal" or observation.terminated:
            return None
        reasons = self.pending.copy()
        if self.last_review_map != state.map_instance:
            reasons["actionable_map"] = state.through.event_id
        if "actor_requested" in reasons and self._last_signature == self._signature(observation, state):
            del reasons["actor_requested"]
        active = self.goal is not None and self.goal.status == "active"
        if not reasons and active:
            return None

        view = goal_view(state)
        assert view is not None
        entries = (
            entry for entry in view["opportunities"]
            if entry["kind"] != "explore" or tuple(entry["position"]) != state.position
        )
        candidates = [{**entry, "candidate_id": index} for index, entry in enumerate(entries, start=1)]
        view["opportunities"] = candidates
        if not active and self.last_review_event is not None and self.last_review_map == state.map_instance:
            for candidate in candidates:
                target = tuple(candidate["position"])
                resource = candidate["kind"] == "collect"
                evidence = state.entities[target] if resource else state.terrain[target]
                if _identity(candidate) not in self._last_candidates and (
                    resource or evidence.first_seen.event_id > self.last_review_event
                ):
                    reasons["new_opportunity"] = evidence.source.event_id
                    break
        return GoalReview(reasons, view) if reasons else None

    def choose(self, reply: GoalReply, state: WorkingState, review: GoalReview) -> dict:
        previous = self.goal
        chosen = select_goal(reply, previous, state, review.candidates, self._next_id)
        superseded = None
        if chosen is not None and chosen.id == self._next_id:
            self._next_id += 1
            if previous is not None and previous.status in {"active", "suspended"}:
                superseded = replace(previous, status="superseded").record()
        self.goal = chosen
        return {"choice": reply.choice, "goal": chosen.record() if chosen else None, "superseded": superseded}

    def record_review(self, observation: Observation, state: WorkingState, review: GoalReview) -> None:
        self.last_review_map = state.map_instance
        self.last_review_event = state.through.event_id
        self._last_signature = self._signature(observation, state)
        self._last_candidates = {_identity(candidate) for candidate in review.candidates}
        self.pending.clear()
        self._reset_stall()

    def actor_record(self) -> dict:
        return {"goal": self.goal.actor_record() if self.goal else None}

    def advance(
        self, transition: DecisionTransition, state: WorkingState, delta: dict, *, review_requested: bool,
    ) -> dict | None:
        event_id = transition.output_event_id
        if review_requested:
            self.pending["actor_requested"] = event_id
        attempt = delta["attempt"]
        previous = self.goal
        verdict = None
        if previous is not None and previous.status in {"active", "suspended"}:
            if previous.status == "active" and _completed(previous, transition, state, delta):
                self.goal = replace(previous, status="complete", stage="complete")
                self.pending["goal_complete"] = event_id
                verdict = "complete"
            elif not target_supported(previous, state) or _pickup_disproved(previous, transition, state):
                self.goal = replace(previous, status="invalid", stage="invalid")
                self.pending["goal_invalid"] = event_id
                verdict = "invalid"
            elif previous.status == "active" and _confirmed(transition):
                self.goal = _progress(previous, state, attempt)
                if self.goal.route_index > previous.route_index or (self.goal.reached_stairs and not previous.reached_stairs):
                    verdict = "progress"

        if verdict or state.map_instance != attempt["map_instance"]:
            self._reset_stall()
        elif attempt["effect"] in COMPARABLE_MOVES:
            self._track_movement(attempt, event_id)

        if self.goal is None and not self.pending:
            return None
        return {
            "goal_id": self.goal.id if self.goal else None,
            "status": self.goal.status if self.goal else None,
            "stage": self.goal.stage if self.goal else None,
            "outcome": verdict,
            "comparable_moves_without_progress": self._comparable_moves,
            "same_edge_returns": self._edge_traversals // 2,
            "pending_review": self.pending.copy(),
        }

    def _signature(self, observation: Observation, state: WorkingState) -> tuple:
        goal = (self.goal.id, self.goal.status, self.goal.stage) if self.goal else None
        return state.map_instance, state.position, observation.state, tuple(observation.screen[1:-1]), goal

    def _reset_stall(self) -> None:
        self._last_edge = None
        self._edge_traversals = self._comparable_moves = 0
        self.pending.pop("edge_loop", None)
        self.pending.pop("movement_stall", None)

    def _track_movement(self, attempt: dict, event_id: int) -> None:
        self._comparable_moves += 1
        if attempt["effect"] == "confirmed_move":
            edge = (tuple(attempt["from"]), tuple(attempt["to"]))
            self._edge_traversals = self._edge_traversals + 1 if self._last_edge == edge[::-1] else 1
            self._last_edge = edge
        else:
            self._last_edge = None
            self._edge_traversals = 0
        if self._edge_traversals // 2 >= self.same_edge_returns:
            self.pending.setdefault("edge_loop", event_id)
        if self._comparable_moves >= self.movement_stall_actions:
            self.pending.setdefault("movement_stall", event_id)


def _identity(candidate: dict) -> tuple:
    return candidate["kind"], tuple(candidate["position"]), candidate["glyph"]


def _confirmed(transition: DecisionTransition) -> bool:
    return transition.input_outcome == transition.output_outcome == "succeeded"


def _command(transition: DecisionTransition, action: str, key: str) -> bool:
    return transition.input_observation.mode == "normal" and (
        transition.action.get("action") == action or transition.action.get("key") == key
    )


def _completed(goal: Goal, transition: DecisionTransition, state: WorkingState, delta: dict) -> bool:
    if not _confirmed(transition):
        return False
    attempt = delta["attempt"]
    before, after = transition.input_observation, transition.output_observation
    target = list(goal.target) if goal.target is not None else None
    if goal.kind == "descend":
        depth = before.state.dungeon_level
        return (
            goal.map_instance == attempt["map_instance"] and attempt["from"] == target
            and _command(transition, "DESCEND", ">") and depth is not None and after.state.dungeon_level == depth + 1
        )
    if state.map_instance != goal.map_instance:
        return False
    if goal.kind == "collect":
        if state.position != goal.target:
            return False
        at_pickup = _command(transition, "PICK_UP", ",") and attempt["from"] == target
        if not (at_pickup or attempt["effect"] == "confirmed_move"):
            return False
        if goal.resource_glyph == "*":
            gold = before.state.gold
            return gold is not None and after.state.gold is not None and after.state.gold > gold
        return any(message.lower().startswith("you now have ") for message in after.messages)
    if goal.target is not None:
        return state.position == goal.target
    return any(
        state.terrain[tuple(change["position"])].first_seen.event_id == transition.output_event_id
        for change in delta.get("terrain", ())
    )


def _pickup_disproved(goal: Goal, transition: DecisionTransition, state: WorkingState) -> bool:
    return (
        goal.kind == "collect" and _confirmed(transition) and state.position == goal.target
        and _command(transition, "PICK_UP", ",")
        and any("nothing here" in message.lower() for message in transition.output_observation.messages)
    )


def _progress(goal: Goal, state: WorkingState, attempt: dict) -> Goal:
    position = state.position
    stage = goal.stage
    if goal.kind == "descend" and position is not None:
        stage = "at_stairs" if position == goal.target else "pursuing"
    route_index = goal.route_index
    if attempt["effect"] == "confirmed_move" and position in goal.route:
        route_index = max(route_index, goal.route.index(position))
    return replace(goal, stage=stage, route_index=route_index, reached_stairs=goal.reached_stairs or stage == "at_stairs")
