from dataclasses import replace
from uuid import UUID

import pytest

from game_runner.context.progress import Progress
from game_runner.context.transitions import DecisionTransition
from game_runner.context.working_state import UnlinkedObservation, WorkingState, replay_working_state

from test_agent_graph import observed

EPISODE = UUID(int=1)


def action(name):
    return {"type": "semantic", "action": name}


def frame(*rows, level=1, mode="normal", messages=()):
    observation = observed(EPISODE, screen=["message", *rows, "status"], mode=mode)
    return observation.model_copy(update={
        "state": observation.state.model_copy(update={"dungeon_level": level}),
        "messages": list(messages),
    })


def transitions(reset, *steps, first_event=10):
    before, outcome = reset, "succeeded"
    trace = []
    for step, (request, after, *provenance) in enumerate(steps, 1):
        after = after.model_copy(update={"api_step": step})
        after_outcome = provenance[0] if provenance else "succeeded"
        trace.append(DecisionTransition(
            first_event + step - 1, before, outcome, 100 + step, request, "Fixture action.",
            first_event + step, after, after_outcome,
        ))
        before, outcome = after, after_outcome
    return trace


def test_replay_preserves_stairs_under_player_and_blank_and_counts_backtracking():
    reset = frame(".@%.", "...#")
    trace = transitions(
        reset,
        (action("MOVE_RIGHT"), frame("..@.", "...#")),
        ({"type": "key", "key": "h"}, frame(".@ .", "...#")),
        (action("REST"), frame(".@ .", "...#")),
    )
    state, deltas = replay_working_state(10, reset, trace)

    assert replay_working_state(10, reset, trace) == (state, deltas)
    assert state.visits[(1, 1)].count == 2
    assert state.visits[(1, 2)].count == 1
    assert state.terrain[(1, 2)].glyph == "%"
    assert state.terrain[(1, 2)].source.event_id == 10
    assert set(state.edges) == {((1, 1), (1, 2)), ((1, 2), (1, 1))}
    assert deltas[1]["edge"] == [[1, 1], [1, 2]]
    assert "visit" not in deltas[-1] and "edge" not in deltas[-1]


def test_confirmed_sighting_refreshes_source_and_visible_replacement_clears_old_entity():
    reset = frame(".@%*.", ".....")
    trace = transitions(
        reset,
        (action("REST"), frame(".@%B.", ".....")),
        (action("REST"), frame(".@  .", ".....")),
        (action("REST"), frame(".@..#", ".....")),
    )
    historical, partial = replay_working_state(10, reset, trace[:2])
    assert historical.terrain[(1, 2)].source.event_id == 11
    assert historical.entities[(1, 3)].glyph == "B"
    assert historical.entities[(1, 3)].source.event_id == 11
    assert [1, 2] in partial[1]["terrain_seen"]
    new, deltas = replay_working_state(10, reset, trace)
    assert new.terrain[(1, 2)].glyph == "."
    assert (1, 3) not in new.entities
    assert deltas[-1]["entities"] == [{"position": [1, 3], "glyph": None}]


def test_quiet_door_attempts_and_same_position_combat_are_distinct():
    reset = frame(".@+B", "..B.")
    trace = transitions(
        reset,
        (action("MOVE_RIGHT"), reset),
        (action("MOVE_RIGHT"), reset),
        (action("MOVE_DOWN_RIGHT"), frame(".@+B", "..B.", messages=("You hit the bat.",))),
    )
    state, deltas = replay_working_state(10, reset, trace)
    assert [a.effect for a in state.attempts[((1, 1), "RIGHT")]] == ["quiet_failed_move"] * 2
    assert deltas[-1]["attempt"]["effect"] == "combat_observed"
    assert deltas[-1]["attempt"]["to"] == [1, 1]
    assert state.visits[(1, 1)].count == 1
    assert not state.edges


def test_menu_round_trip_preserves_map_without_counting_a_visit():
    reset = frame(".@%.", "...#")
    overlay = frame("a) scroll", ".@--", mode="select_item", level=None)
    trace = transitions(reset, (action("INVENTORY"), overlay), ({"type": "key", "key": "ENTER"}, reset))
    during, partial = replay_working_state(10, reset, trace[:1])
    assert during.position is None and during.map_instance == 10
    assert during.terrain[(1, 2)].glyph == "%"
    assert "terrain" not in partial[-1]
    after, deltas = replay_working_state(10, reset, trace)
    assert after.position == (1, 1) and after.visits[(1, 1)].count == 1
    assert "visit" not in deltas[-1]


def test_level_change_at_same_coordinates_expires_old_map():
    reset = frame(".@%.", "...#")
    trace = transitions(reset, (action("DESCEND"), frame(".@..", "...#", level=2)))
    state, deltas = replay_working_state(10, reset, trace)
    assert state.map_instance == 11 and state.visits[(1, 1)].count == 1
    assert state.terrain[(1, 2)].glyph == "."
    assert state.terrain[(1, 2)].source.event_id == 11
    assert not state.attempts and not state.edges
    assert deltas[-1]["attempt"]["map_instance"] == 10
    assert deltas[-1]["attempt"]["level"] == [1, 2]
    assert deltas[-1]["attempt"]["to"] is None


def test_recovered_gap_seeds_facts_only_on_the_next_confirmed_frame():
    reset = frame(".@%.", "...#")
    recovered = frame("..@.", "...#")
    confirmed = frame("..@.", "...#")
    trace = transitions(
        reset,
        (action("MOVE_RIGHT"), recovered, "uncertain"),
        (action("REST"), confirmed),
    )
    at_gap, partial = replay_working_state(10, reset, trace[:1])
    assert at_gap.position is None and at_gap.map_instance is None
    assert not at_gap.visits and not at_gap.terrain and not at_gap.edges
    assert partial[-1]["gap"] == "action_response_lost"
    assert partial[-1]["attempt"]["effect"] == "uncertain"
    assert partial[-1]["attempt"]["to"] is None
    after, deltas = replay_working_state(10, reset, trace)
    assert after.position == (1, 2) and after.visits[(1, 2)].count == 1
    assert (1, 2) not in after.terrain and not after.edges
    assert deltas[-1]["attempt"]["effect"] == "uncertain"


def test_unexpected_displacement_does_not_prove_a_traversed_edge():
    reset = frame(".@...", ".....")
    trace = transitions(reset, (action("MOVE_RIGHT"), frame("....@", ".....")))
    state, deltas = replay_working_state(10, reset, trace)
    attempt = deltas[-1]["attempt"]
    assert attempt["effect"] == "observed_relocation"
    assert attempt["target"] == [1, 2] and attempt["to"] == [1, 4]
    assert state.visits[(1, 4)].count == 1 and not state.edges


def test_unlinked_transition_leaves_working_state_at_original_boundary():
    reset = frame(".@%.", "...#")
    transition, = transitions(reset, (action("REST"), reset))
    progress = Progress.initial(10, reset)
    state, _ = WorkingState.from_reset(10, reset, progress)
    after = progress.advance(11, reset, transition.action, transition.output_observation, recovered=False)
    with pytest.raises(UnlinkedObservation, match="not linked"):
        state.advance(replace(transition, input_event_id=9), after)
    assert state.through.event_id == 10 and state.visits[(1, 1)].count == 1
