from uuid import UUID

import pytest

from game_runner.context.messages import build_messages
from game_runner.context.planning import ContextPlanner, ContextSettings
from game_runner.context.progress import Progress
from game_runner.context.working_state import UnlinkedObservation, replay_working_state
from game_runner.context.working_views import actor_view, goal_view

from test_agent_graph import FakeModel, FakeRepository
from test_working_state import action, frame, transitions

WORKING_LABEL = "\nObserved working state."


def test_actor_view_adds_only_observed_facts_to_the_final_message():
    reset = frame(".@%+", "....")
    trace = transitions(reset, (action("MOVE_RIGHT"), frame("..@+", "....")))
    state, _ = replay_working_state(10, reset, trace)
    progress = Progress.initial(10, reset).advance(
        11, reset, trace[0].action, trace[0].output_observation, recovered=False,
    )
    kwargs = {"progress": progress, "prompt_profile": "rules-v2"}
    baseline = build_messages(trace[0].output_observation, trace, **kwargs)
    view = actor_view(state)
    augmented = build_messages(trace[0].output_observation, trace, **kwargs, working_view=view)

    assert augmented[:-1] == baseline[:-1]
    assert augmented[-1].content.partition(WORKING_LABEL)[0] == baseline[-1].content
    assert sum(m.content.count("Current 80x24 screen") for m in augmented) == 1
    assert "opportunities" not in view
    assert view["landmarks"] == [
        {"kind": "stairs", "position": [1, 2], "source": {"event_id": 10, "api_step": 0, "age": 1}},
    ]
    assert view["tiles"][0] == {
        "position": [1, 2], "terrain": "%", "source": {"event_id": 10, "api_step": 0, "age": 1},
        "newly_seen": False, "visits": 1, "last_visit_event_id": 11,
    }
    assert view["recent_attempts"][0]["output_event_id"] == 11


def test_initial_map_offers_a_sourced_exploration_candidate_without_known_stairs():
    reset = frame(".@.+ ", "-----")
    state, _ = replay_working_state(10, reset, [])
    goals = goal_view(state)
    door, = (item for item in goals["opportunities"] if item["basis"] == "untried_door")
    assert door == {
        "kind": "explore", "basis": "untried_door", "position": [1, 3], "glyph": "+",
        "source": {"event_id": 10, "api_step": 0, "age": 0},
        "applicability": {"map_instance": 10, "current_sighting_required": False},
        "reachability": "inferred", "distance": 2, "route": [[1, 1], [1, 2], [1, 3]],
    }
    assert all(item["kind"] != "stairs" for item in goals["opportunities"])
    assert actor_view(state)["tiles"][0]["terrain"] is None


def test_attempted_door_is_no_longer_untried_and_limits_hold():
    reset = frame(".......", ".......", "..@+...", ".......", ".......")
    trace = transitions(reset, *((action("MOVE_RIGHT"), reset) for _ in range(5)))
    state, _ = replay_working_state(10, reset, trace)
    view = actor_view(state)
    assert len(view["tiles"]) == 16
    assert [a["output_event_id"] for a in view["recent_attempts"]] == [14, 15]
    assert all(item["basis"] != "untried_door" for item in goal_view(state)["opportunities"])


def test_current_resource_and_known_stairs_can_be_sourced_without_an_invented_route():
    reset = frame(".@..|%*", "-------")
    moved = frame(".@..|  ", "-------")
    state, _ = replay_working_state(10, reset, [])
    candidates = {item["kind"]: item for item in goal_view(state)["opportunities"]}
    assert candidates["collect"]["source"]["event_id"] == 10
    assert candidates["stairs"]["reachability"] == candidates["collect"]["reachability"] == "unknown"
    assert "route" not in candidates["stairs"] and "route" not in candidates["collect"]

    state, _ = replay_working_state(10, reset, transitions(reset, (action("REST"), moved)))
    candidates = {item["kind"]: item for item in goal_view(state)["opportunities"]}
    assert "collect" not in candidates
    assert candidates["stairs"]["source"]["age"] == 1


def test_confirmed_diagonal_can_support_a_route_but_unknown_diagonal_cannot():
    reset = frame("@ ", " %")
    initial, _ = replay_working_state(10, reset, [])
    stairs, = goal_view(initial)["opportunities"]
    assert stairs["reachability"] == "unknown"

    trace = transitions(
        reset, (action("MOVE_DOWN_RIGHT"), frame(". ", " @")), (action("MOVE_UP_LEFT"), reset),
    )
    state, _ = replay_working_state(10, reset, trace)
    stairs, = (item for item in goal_view(state)["opportunities"] if item["kind"] == "stairs")
    assert stairs["reachability"] == "confirmed"
    assert stairs["route"] == [[1, 0], [2, 1]]


@pytest.mark.parametrize(("observed", "outcome"), [
    (frame(".@--", "a) scroll", mode="select_item"), "succeeded"),
    (frame(".@%+", "...."), "uncertain"),
])
def test_projections_are_withheld_during_menus_and_gaps(observed, outcome):
    reset = frame(".@%+", "....")
    trace = transitions(reset, (action("REST"), observed, outcome))
    state, _ = replay_working_state(10, reset, trace)
    assert actor_view(state) is goal_view(state) is None


async def test_context_planner_rejects_working_state_ahead_of_input():
    reset = frame(".@%+", "....")
    state, _ = replay_working_state(10, reset, transitions(reset, (action("REST"), reset)))
    planner = ContextPlanner(FakeModel(), FakeRepository(), ContextSettings(8192, prompt_profile="rules-v2"))
    with pytest.raises(UnlinkedObservation, match="actor's input boundary"):
        await planner.prepare(UUID(int=2), reset, 10, "succeeded", Progress.initial(10, reset), state)
