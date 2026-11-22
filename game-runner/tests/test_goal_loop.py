"""Goal lifecycle fixtures use the linked map observations from the runner."""

import json

import pytest

from game_runner.context.goal_loop import GoalLoop
from game_runner.context.goals import parse_goal_reply
from game_runner.context.progress import Progress
from game_runner.context.working_state import WorkingState

from test_working_state import action, frame, transitions


class GoalTrace:
    def __init__(self, reset, *steps):
        self.progress = Progress.initial(10, reset)
        self.state, _ = WorkingState.from_reset(10, reset, self.progress)
        self.observation = reset
        self.loop = GoalLoop()
        self.steps = iter(transitions(reset, *steps))

    def reply(self, choice, **fields):
        review = self.loop.assess(self.observation, self.state)
        assert review is not None
        decision = parse_goal_reply(json.dumps({"choice": choice, **fields}))
        result = self.loop.choose(decision, self.state, review)
        self.loop.record_review(self.observation, self.state, review)
        return result

    def step(self, *, request=False):
        step = next(self.steps)
        self.observation = step.output_observation
        self.progress = self.progress.advance(
            step.output_event_id, step.input_observation, step.action, self.observation,
            recovered=step.output_outcome == "uncertain",
        )
        delta = self.state.advance(step, self.progress)
        return self.loop.advance(step, self.state, delta, review_requested=request)

    def reasons(self):
        review = self.loop.assess(self.observation, self.state)
        return set(review.reasons) if review else set()


def test_stairs_allow_a_model_proposed_retreat():
    reset = frame("..@%.", ".....")
    trace = GoalTrace(
        reset,
        (action("MOVE_RIGHT"), frame("...@.", ".....")),
        (action("MOVE_LEFT"), reset),
        (action("MOVE_LEFT"), frame(".@.%.")),
    )
    assert trace.reasons() == {"actionable_map"}
    chosen = trace.reply("propose", kind="retreat", target=[1, 1], source_event_id=10)
    assert chosen["goal"]["kind"] == "retreat"
    trace.step()
    assert trace.step()["outcome"] is None
    assert trace.step()["outcome"] == "complete"
    assert trace.reasons() == {"goal_complete"}


def test_selecting_stairs_keeps_the_goal_through_moves_and_credits_descent():
    reset = frame(".@..%", ".....")
    trace = GoalTrace(
        reset,
        (action("MOVE_RIGHT"), frame("..@.%", ".....")),
        (action("MOVE_RIGHT"), frame("...@%", ".....")),
        (action("MOVE_RIGHT"), frame("....@", ".....")),
        ({"type": "key", "key": ">"}, frame(".@..", "....", level=2)),
    )
    review = trace.loop.assess(reset, trace.state)
    stairs = next(item for item in review.candidates if item["kind"] == "stairs")
    goal = trace.reply("select", candidate_id=stairs["candidate_id"])["goal"]
    assert goal["kind"] == "descend"
    assert trace.step()["outcome"] == "progress"
    assert trace.step()["goal_id"] == goal["id"]
    assert trace.step()["stage"] == "at_stairs"
    assert trace.step()["outcome"] == "complete"
    assert trace.reasons() == {"actionable_map", "goal_complete"}


def test_revealing_stairs_can_keep_an_existing_exploration_goal():
    reset = frame(".@.. ", ".....")
    stairs = frame(".@..%", ".....")
    trace = GoalTrace(reset, (action("SEARCH"), stairs))
    first = trace.reply("propose", kind="explore", target=[1, 3], source_event_id=10)
    trace.step(request=True)
    kept = trace.reply("keep")
    assert kept["goal"]["id"] == first["goal"]["id"]
    assert kept["goal"]["kind"] == "explore"


def test_loop_and_quiet_failures_request_review_at_exact_thresholds():
    reset = frame(".@..", "....")
    other = frame("..@.", "....")
    trace = GoalTrace(reset, *((action("MOVE_RIGHT" if i % 2 == 0 else "MOVE_LEFT"), other if i % 2 == 0 else reset)
                              for i in range(4)))
    trace.reply("defer")
    for _ in range(3):
        trace.step()
        assert not trace.reasons()
    assert trace.step()["same_edge_returns"] == 2
    assert trace.reasons() == {"edge_loop"}

    trace = GoalTrace(reset, *((action("MOVE_UP"), reset) for _ in range(6)))
    trace.reply("defer")
    for _ in range(5):
        trace.step()
        assert not trace.reasons()
    assert trace.step()["comparable_moves_without_progress"] == 6
    assert trace.reasons() == {"movement_stall"}


def test_unchanged_actor_requests_and_stale_targets_are_qualified():
    reset = frame(".@%.", "....")
    next_map = frame(".@%.", "....", level=2)
    trace = GoalTrace(reset, (action("REST"), reset), (action("MOVE_RIGHT"), next_map))
    trace.reply("propose", kind="descend", target=[1, 2], source_event_id=10)
    trace.step(request=True)
    assert not trace.reasons()
    assert trace.step()["outcome"] == "invalid"
    with pytest.raises(ValueError, match="current map"):
        trace.reply("propose", kind="descend", target=[1, 2], source_event_id=10)
    assert trace.loop.goal.status == "invalid"


def test_goal_less_review_reopens_on_new_sourced_evidence():
    reset = frame(".@.. ", ".....")
    trace = GoalTrace(reset, (action("REST"), reset), (action("SEARCH"), frame(".@..%", ".....")))
    trace.reply("defer")
    trace.step()
    assert not trace.reasons()
    trace.step()
    assert trace.reasons() == {"new_opportunity"}


def test_goal_review_waits_for_modal_completion_and_ignores_recovered_effect():
    reset = frame(".@%.", "....")
    menu = frame("", "a) Food", mode="select_item")
    trace = GoalTrace(
        reset, (action("INVENTORY"), menu), (action("CANCEL"), frame("..@.", "....")),
        (action("DESCEND"), frame(".@..", "....", level=2), "uncertain"),
        (action("REST"), frame(".@..", "....", level=2)),
    )
    trace.reply("propose", kind="descend", target=[1, 2], source_event_id=10)
    trace.step(request=True)
    assert not trace.reasons()
    assert trace.step()["stage"] == "at_stairs"
    assert trace.reasons() == {"actor_requested"}
    assert trace.step()["outcome"] == "invalid"
    assert not trace.reasons()
    trace.step()
    assert trace.reasons() == {"actionable_map", "actor_requested", "goal_invalid"}


def test_collect_requires_a_confirmed_pickup_and_targetless_exploration_requires_new_terrain():
    reset = frame(".@!.", "....")
    covered = frame("..@.", "....")
    pickup = frame("..@.", "....", messages=("You now have a blue potion (a)",))
    trace = GoalTrace(reset, (action("MOVE_RIGHT"), covered), (action("PICK_UP"), pickup))
    trace.reply("propose", kind="collect", target=[1, 2], source_event_id=10)
    assert trace.step()["status"] == "active"
    assert trace.step()["outcome"] == "complete"

    reset = frame(".@. ", "....")
    trace = GoalTrace(reset, (action("REST"), reset), (action("SEARCH"), frame(".@.%", "....")))
    trace.reply("propose", kind="explore")
    assert trace.step()["outcome"] is None
    assert trace.step()["outcome"] == "complete"


def test_a_suspended_goal_can_be_resumed_or_superseded_by_the_model():
    reset = frame(".@%.", "....")
    trace = GoalTrace(reset, (action("MOVE_RIGHT"), frame("..@.", "....")),
                      (action("MOVE_LEFT"), reset), (action("MOVE_RIGHT"), frame("..@.", "....")))
    selected = trace.reply("propose", kind="descend", target=[1, 2], source_event_id=10)["goal"]
    trace.step(request=True)
    assert trace.reply("suspend")["goal"]["status"] == "suspended"
    trace.step(request=True)
    resumed = trace.reply("keep")["goal"]
    assert resumed["id"] == selected["id"] and resumed["status"] == "active"
    trace.step(request=True)
    replacement = trace.reply("propose", kind="retreat", target=[1, 1], source_event_id=12)
    assert replacement["goal"]["id"] != selected["id"]
    assert replacement["superseded"]["status"] == "superseded"


def test_purposeful_backtracking_resets_stall_when_it_advances_the_goal_route():
    reset = frame(".@...", ".....")
    trace = GoalTrace(
        reset, (action("MOVE_RIGHT"), frame("..@..", ".....")),
        (action("MOVE_LEFT"), reset), (action("MOVE_LEFT"), frame("@....", ".....")),
    )
    trace.reply("defer")
    trace.step(request=True)
    trace.reply("propose", kind="retreat", target=[1, 0], source_event_id=11)
    returned = trace.step()
    assert returned["outcome"] == "progress"
    assert returned["same_edge_returns"] == returned["comparable_moves_without_progress"] == 0
    assert trace.step()["outcome"] == "complete"
