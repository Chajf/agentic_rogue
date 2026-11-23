"""Exercise the state-only comparison through the normal runner and export."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

import pytest

from game_runner.clients.rogue import Observation, RogueTransportError
from game_runner.context.transitions import DecisionTransition
from game_runner.context.working_state import replay_working_state
from game_runner.pilot import store
from game_runner.pilot.__main__ import main as pilot_main
from game_runner.pilot.manifest import Manifest
from game_runner.pilot.runner import private_rogue
from game_runner.rule_feasibility.corpus import load
from game_runner.session import run_session

from test_agent_graph import FakeClient, FakeModel, FakeRepository
from test_pilot_store import isolated_database, make_manifest  # noqa: F401
from test_prompt_profiles import REST, RUNTIME
from test_working_state import frame
from test_working_views import WORKING_LABEL

DELTA_KEY = "working_state_delta"
# The pinned build's seed 3 sequence reaches the stairs and descends.
STAIR_ACTIONS = [{"type": "key", "key": key} for key in "hkk>"]
STAIR_REPLIES = [json.dumps({"action": action, "rationale": "Follow the fixture."}) for action in STAIR_ACTIONS]


def _state_pair():
    payload = make_manifest().model_dump(mode="json")
    for trial in payload["trials"]:
        trial.update(seed=3, context_policy="history_only", prompt_profile="rules-v2")
    payload["trials"][0].pop("structured_state")
    payload["trials"][1]["structured_state"] = "map-v1"
    payload["actor"]["context_tokens"] = 16384
    payload["trial_limits"]["actions"] = 4
    payload["series_limits"]["actions"] = 8
    return payload


def test_manifest_selects_state_per_trial_and_supports_an_independent_c_run():
    payload = _state_pair()
    manifest = Manifest.model_validate(payload)
    assert [manifest.settings(RUNTIME, trial).structured_state for trial in manifest.trials] == ["off", "map-v1"]
    payload["trials"] = payload["trials"][1:]
    assert Manifest.model_validate(payload).trials[0].structured_state == "map-v1"


@pytest.mark.parametrize(("field", "value"), [
    ("prompt_profile", "legacy"),
    ("context_policy", "working_memory"),
    ("context_variant", "neighbors"),
])
def test_bad_state_combinations_fail_at_manifest_validation(field, value):
    payload = _state_pair()
    payload["trials"][1][field] = value
    with pytest.raises(ValueError, match="map-v1 requires rules-v2, history_only, and the baseline"):
        Manifest.model_validate(payload)


@pytest.mark.parametrize("history_format", ["replay", "compact"])
async def test_runtime_withholds_recovered_map_then_resumes_from_a_confirmed_frame(history_format):
    reset = frame(".@%.", "....")
    recovered = frame("..@.", "....").model_copy(update={"api_step": 1})
    confirmed = recovered.model_copy(update={"api_step": 2})
    last = recovered.model_copy(update={"api_step": 3})

    class RecoveringClient(FakeClient):
        async def action(self, episode_id, action):
            self.actions.append((episode_id, action))
            if len(self.actions) == 1:
                raise RogueTransportError("lost response")
            return confirmed if len(self.actions) == 2 else last

    client = RecoveringClient(recovered)
    client.initial = reset
    repo = FakeRepository()
    model = FakeModel("bad JSON", REST, REST, REST)
    settings = replace(
        RUNTIME, game_seed=reset.seed, prompt_profile="rules-v2", structured_state="map-v1",
        history_format=history_format,
    )

    assert await run_session(settings, repo, model=model, client=client) == "limit_reached"

    assert len(client.actions) == 3 and len(repo.events) == 4
    assert [event["outcome"] for event in repo.events] == ["succeeded", "uncertain", "succeeded", "succeeded"]
    first, retry, gap_call, resumed = [call["input_messages"][-1]["content"] for call in repo.calls]
    assert WORKING_LABEL in first and retry
    assert "recovered after an action timeout" in gap_call and WORKING_LABEL not in gap_call
    assert WORKING_LABEL in resumed
    view = json.loads(resumed.splitlines()[-1])
    assert view["landmarks"] == [] and view["recent_attempts"] == []
    assert view["through_event_id"] == 3
    assert repo.calls[2]["history_window"]["eligible_output_event_ids"] == []
    assert repo.calls[3]["history_window"]["eligible_output_event_ids"] == [3]
    assert repo.calls[3]["history_window"]["delivered_output_event_ids"] == [3]
    if history_format == "compact":
        assert "History boundary at API step 1." in resumed
        assert "action effect uncertain" in resumed
    reset_delta, gap, next_confirmed, _ = [event["progress"][DELTA_KEY] for event in repo.events]
    assert reset_delta["visit"]["count"] == next_confirmed["visit"]["count"] == 1
    assert gap["gap"] == "action_response_lost" and gap["attempt"]["to"] is None


async def test_unlinked_action_response_interrupts_after_one_dispatch():
    reset = frame(".@%.", "....")
    client = FakeClient(reset.model_copy(update={"api_step": 3}))
    client.initial = reset
    repo = FakeRepository()
    settings = replace(RUNTIME, game_seed=reset.seed, prompt_profile="rules-v2", structured_state="map-v1")

    assert await run_session(settings, repo, model=FakeModel(REST), client=client) == "interrupted"

    assert len(client.actions) == 1
    assert repo.events[1]["outcome"] == "uncertain"
    assert "observation" not in repo.events[1] and "progress" not in repo.events[1]


def _trace_from_export(events, calls_by_id):
    reset, *actions = events
    transitions = []
    by_id = {event["id"]: event for event in events}
    for output in actions:
        call = calls_by_id[output["model_call_id"]]
        source = by_id[call["observation_event_id"]]
        transitions.append(DecisionTransition(
            input_event_id=source["id"], input_observation=Observation.model_validate(source["observation"]),
            input_outcome=source["outcome"], model_call_id=call["id"],
            action=call["parsed_action"], rationale=call["decision_rationale"], output_event_id=output["id"],
            output_observation=Observation.model_validate(output["observation"]), output_outcome=output["outcome"],
        ))
    return reset["id"], Observation.model_validate(reset["observation"]), transitions


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
def test_native_bc_pair_exports_and_replays_working_state(isolated_database, monkeypatch, tmp_path):
    monkeypatch.setattr(store, "connect", isolated_database)
    monkeypatch.setenv("MODEL_BASE_URL", "http://test")
    monkeypatch.setenv("PROMPT_PROFILE", "rules-v2")
    monkeypatch.setenv("STRUCTURED_STATE", "map-v1")
    monkeypatch.setattr("game_runner.session.init_chat_model", lambda *_, **__: FakeModel(*STAIR_REPLIES))
    payload = _state_pair()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "game_runner.pilot", "run", "--manifest", str(manifest),
        "--rogue-binary", os.environ["TEST_ROGUE_BINARY"], "--runner-revision", "0" * 40,
    ])
    pilot_main()

    output = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", [
        "game_runner.pilot", "export", "--series-id", payload["series_id"], "--output", str(output),
    ])
    pilot_main()
    corpus = load(output, "development")
    assert [
        (trial["structured_state"], trial["prompt_version"], trial["outcome"])
        for trial in corpus.trials.values()
    ] == [("off", "rules-v2.2", "limited"), ("map-v1", "rules-v2.2", "limited")]

    def trial_rows(rows, trial_id):
        return [row for row in rows.values() if row["trial_id"] == trial_id]

    reference_calls = trial_rows(corpus.calls, "first")
    comparison_calls = trial_rows(corpus.calls, "second")
    reference_events = trial_rows(corpus.events, "first")
    comparison_events = trial_rows(corpus.events, "second")
    for baseline, augmented in zip(reference_events, comparison_events, strict=True):
        assert baseline["outcome"] == augmented["outcome"] == "succeeded"
        for field in ("screen", "state", "messages", "seed"):
            assert baseline["observation"][field] == augmented["observation"][field]
        assert DELTA_KEY not in baseline["progress"]
    for step, (baseline, augmented) in enumerate(zip(reference_calls, comparison_calls, strict=True)):
        left, right = baseline["input_messages"], augmented["input_messages"]
        assert left[:-1] == right[:-1]
        assert len(right) == 2 + 2 * step
        assert WORKING_LABEL not in left[-1]["content"]
        assert WORKING_LABEL in right[-1]["content"]
        assert sum(message["content"].count("Current 80x24 screen") for message in right) == 1
        view = json.loads(right[-1]["content"].splitlines()[-1])
        assert view["through_event_id"] == augmented["observation_event_id"]
        assert "opportunities" not in view
        reference, comparison = reference_events[step + 1], comparison_events[step + 1]
        assert reference["action_request"] == baseline["parsed_action"] == STAIR_ACTIONS[step]
        assert comparison["action_request"] == augmented["parsed_action"] == STAIR_ACTIONS[step]
        assert comparison["model_call_id"] == augmented["id"]

    state, deltas = replay_working_state(*_trace_from_export(comparison_events, corpus.calls))
    assert deltas == [item["progress"][DELTA_KEY] for item in comparison_events]
    assert len(comparison_events) == 5 and len(comparison_calls) == 4
    assert state.map_instance == comparison_events[-1]["id"]
    assert deltas[-1]["attempt"]["level"] == [1, 2]
    stairs, = (item for item in view["landmarks"] if item["position"] == view["player"])
    assert stairs["source"]["event_id"] < view["through_event_id"]


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
async def test_native_state_profile_fits_the_default_context_budget():
    settings = replace(RUNTIME, game_seed=3, max_game_actions=4, prompt_profile="rules-v2", structured_state="map-v1")
    repo = FakeRepository()
    model = FakeModel(*STAIR_REPLIES)
    async with private_rogue(Path(os.environ["TEST_ROGUE_BINARY"]), settings.rogue_timeout_seconds) as (client, _):
        status = await run_session(settings, repo, model=model, client=client)

    assert status == "limit_reached"
    assert len(repo.calls) == 4 and repo.events[-1]["observation"]["state"]["dungeon_level"] == 2
    for call in repo.calls:
        current = call["input_messages"][-1]["content"]
        assert current.count("Current 80x24 screen") == 1
        assert WORKING_LABEL in current
        assert len(json.loads(current.splitlines()[-1])["tiles"]) < 16
