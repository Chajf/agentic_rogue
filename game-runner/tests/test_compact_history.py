"""Freeze examples of the compact actor message with observed action effects."""

from dataclasses import asdict, replace

from game_runner.context.compact_history import render_compact_history
from game_runner.context.messages import build_messages
from game_runner.context.progress import Progress
from game_runner.context.working_state import replay_working_state
from game_runner.context.working_views import actor_view

from test_working_state import action, frame, transitions

HEADER = (
    "Recent completed attempts (oldest first). Each line describes a past attempt. "
    "The latest screen below is the current frame. The working-state block contains sourced observations."
)


def trace_with_progress(reset, raw_trace, reset_event=10):
    state, deltas = replay_working_state(reset_event, reset, raw_trace)
    progress = Progress.initial(reset_event, reset)
    trace = []
    for transition, delta in zip(raw_trace, deltas[1:], strict=True):
        previous = progress
        progress = progress.advance(
            transition.output_event_id, transition.input_observation, transition.action,
            transition.output_observation, recovered=transition.output_outcome == "uncertain",
        )
        trace.append(replace(
            transition, input_progress=asdict(previous), progress={**asdict(progress), "working_state_delta": delta},
        ))
    return state, progress, trace


def test_empty_compact_window_is_identical_to_replay():
    reset = frame(".@+.", "....")
    context = {"progress": Progress.initial(10, reset), "prompt_profile": "rules-v2"}
    assert build_messages(reset, [], history_format="compact", **context) == build_messages(reset, [], **context)


def test_frozen_compact_rows_for_first_visit_revisit_combat_and_map_change():
    reset = frame(".@%B", "....")
    combat = frame(".@B.", "....", messages=("There is a strange sound.", "You hit the bat."))
    combat = combat.model_copy(update={"state": combat.state.model_copy(update={"hp": 10})})
    trace = transitions(
        reset,
        (action("MOVE_RIGHT"), frame("..@B", "....")),
        ({"type": "key", "key": "h"}, frame(".@B.", "....")),
        (action("MOVE_RIGHT"), combat),
        (action("DESCEND"), frame(".@..", "....", level=2)),
    )
    state, progress, history = trace_with_progress(reset, trace)
    current = history[-1].output_observation
    options = {"progress": progress, "prompt_profile": "rules-v2", "working_view": actor_view(state)}
    baseline = build_messages(current, history, **options)
    compact = build_messages(current, history, history_format="compact", **options)
    expected = "\n".join([
        HEADER,
        '1 MOVE_RIGHT [1,1] -> [1,2] | confirmed first visit',
        '2 key="h" [1,2] -> [1,1] | confirmed revisit',
        '3 MOVE_RIGHT [1,1] -> [1,1] | combat observed, HP 12 -> 10, game "You hit the bat."',
        '4 DESCEND map 10 [1,1] -> map 14 [1,1] | map changed, HP 10 -> 12, depth 1 -> 2',
        "",
        "",
    ])
    assert [message.type for message in compact] == ["system", "human"]
    assert compact[0].content == baseline[0].content
    assert compact[-1].content == expected + baseline[-1].content
    for messages in (baseline, compact):
        assert messages[-1].type == "human"
        assert sum(message.content.count("Current 80x24 screen") for message in messages) == 1


def test_compact_row_preserves_action_parameters_and_prefers_bounded_combat_feedback():
    reset = frame(".@B.", "....")
    hit = frame(".@B.", "....", messages=("An ordinary message.", "You hit\nthe bat. " + "x" * 140))
    _, _, trace = trace_with_progress(reset, transitions(reset, (action("MOVE_RIGHT"), hit)))
    assert render_compact_history(trace) == "\n".join([
        HEADER,
        '1 MOVE_RIGHT [1,1] -> [1,1] | combat observed, game "You hit the bat. ' + "x" * 102 + '…"',
        "",
        "",
    ])
    requests = [
        ({"type": "semantic", "action": "THROW", "item": "a", "direction": "UP_LEFT"}, hit),
        ({"type": "semantic", "action": "PUT_ON_RING", "item": "b", "hand": "LEFT"}, hit),
    ]
    _, _, trace = trace_with_progress(reset, transitions(reset, *requests))
    lines = render_compact_history(trace).splitlines()
    assert lines[1].startswith("1 THROW item=a direction=UP_LEFT [1,1] -> [1,1]")
    assert lines[2].startswith("2 PUT_ON_RING item=b hand=LEFT [1,1] -> [1,1]")


def test_recovered_boundary_qualifies_the_next_confirmed_observation():
    reset = frame(".@%.", "....")
    recovered = frame("..@.", "....", messages=("This effect was missed.",))
    confirmed = frame("..@.", "....", messages=("You wait.",))
    trace = transitions(reset, (action("MOVE_RIGHT"), recovered, "uncertain"), (action("REST"), confirmed))
    _, _, history = trace_with_progress(reset, trace)
    assert render_compact_history(history[:1]) == (
        HEADER + "\n1 MOVE_RIGHT ? -> ? | action response lost, effect unknown\n\n"
    )
    assert render_compact_history(history[1:]) == "\n".join([
        HEADER,
        "History boundary at API step 1. The prior action response was lost.",
        '2 REST ? -> [1,2] | action effect uncertain, game "You wait."',
        "",
        "",
    ])
    current = build_messages(
        history[0].output_observation, [], observation_outcome="uncertain", history_format="compact",
    )[-1].content
    assert "The action response was lost." in current
    assert HEADER not in current
