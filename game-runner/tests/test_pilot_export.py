from hashlib import sha256
import json
import os
from uuid import uuid4

import pytest

from game_runner import migrate
from game_runner.persistence.repository import GameRepository
from game_runner.pilot.export import export_series
from game_runner.pilot.store import PilotError, claim_trial, create_series, finalize_series, finalize_trial, series_lock

from test_agent_graph import observed
from test_pilot_store import isolated_database, make_manifest  # noqa: F401


def test_export_is_reproducible_and_keeps_failed_and_unstarted_trials(isolated_database, monkeypatch, tmp_path):
    from game_runner.pilot import store

    migrate.main()
    monkeypatch.setattr(store, "connect", isolated_database)
    manifest = make_manifest()
    session, episode = uuid4(), uuid4()
    repository = GameRepository()
    with series_lock(manifest.series_id) as connection:
        create_series(connection, manifest, {"runner_revision": "abc"})
        claim_trial(connection, manifest, manifest.trials[0], session)
        repository.update_session(session, "running", episode_id=episode, seed=1234)
        reset = repository.start_game_event(session, "reset")
        state = observed(episode).model_dump(mode="json")
        repository.finish_game_event(reset, "succeeded", observation=state, api_step=0)
        action = {"type": "semantic", "action": "MOVE_UP"}
        call = repository.start_model_call(session, [], observation_event_id=reset,
                                           input_token_count=45, input_token_count_method="utf8_byte_estimate")
        repository.finish_model_call(call, "valid", raw_response={"http_headers": "fake-secret"},
                                     parsed_action=action, decision_rationale="Move.", duration_ms=15,
                                     usage_metadata={"total_tokens": 53})
        dispatch = repository.start_game_event(session, "action", model_call_id=call, action_request=action)
        repository.finish_game_event(dispatch, "succeeded", api_step=1,
                                     observation={**state, "api_step": 1}, progress={"position_change": "same"})
        repository.start_model_call(session, [], observation_event_id=dispatch)
        finalize_trial(connection, manifest.series_id, "first", "error", "provider_error")
        with pytest.raises(PilotError, match="active"):
            export_series(manifest.series_id, tmp_path / "busy")
        finalize_series(connection, manifest.series_id, "series_time_limit")

    left, right = tmp_path / "left", tmp_path / "right"
    assert export_series(manifest.series_id, left) == export_series(manifest.series_id, right)
    for file in left.iterdir():
        assert file.read_bytes() == (right / file.name).read_bytes()
        assert b"fake-secret" not in file.read_bytes()
    assert sha256((left / "SHA256SUMS").read_bytes()).hexdigest() == (left / "bundle.sha256").read_text().strip()

    first, second = [json.loads(line) for line in (left / "results.jsonl").read_text().splitlines()]
    assert (first["observed_reset_seed"], first["requested_seed"]) == (1234, 1234)
    assert [first["outcome"], second["outcome"]] == ["failed", "unstarted"]
    assert first["screening_opportunities"]["ambiguous_movement"]["records"] == [
        {"input_event_id": reset, "model_call_id": call, "output_event_id": dispatch},
    ]
    usage = first["model_usage"]["decision"]
    assert (usage["provider_total_tokens"], usage["calls_missing_provider_usage"]) == (53, 1)
    serialized = [json.loads(line) for line in (left / "trajectory.jsonl").read_text().splitlines()]
    assert [record["record_type"] for record in serialized] == ["event", "model_call", "event", "model_call"]
    assert serialized[-1]["status"] == "pending"


def test_export_closes_abandoned_series(isolated_database, monkeypatch, tmp_path):
    from game_runner.pilot import store

    migrate.main()
    monkeypatch.setattr(store, "connect", isolated_database)
    manifest = make_manifest()
    with series_lock(manifest.series_id) as connection:
        create_series(connection, manifest, {})
        claim_trial(connection, manifest, manifest.trials[0], uuid4())
    output = tmp_path / "abandoned"
    export_series(manifest.series_id, output)
    trials = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert [(trial["outcome"], trial["reason"]) for trial in trials] == [
        ("interrupted", "abandoned"), ("unstarted", "abandoned"),
    ]


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
async def test_native_series_exports_history_and_compaction(isolated_database, monkeypatch, tmp_path):
    from pathlib import Path

    from langchain_core.messages import AIMessage

    from game_runner.config import Settings
    from game_runner.pilot import store
    from game_runner.pilot.manifest import Manifest
    from game_runner.pilot.runner import run_series
    from game_runner.session import run_session

    class FixedActor:
        async def ainvoke(self, messages):
            return AIMessage(content='{"action":{"type":"semantic","action":"REST"},"rationale":"Wait."}')

    class FixedSummary:
        async def ainvoke(self, messages):
            event = json.loads(messages[-1].content)["new_trace"][-1]["output_event_id"]
            return AIMessage(content=json.dumps({
                "observations": [{"text": "The actor waited.", "source_event_ids": [event]}],
                "goals": [], "unsuccessful_attempts": [], "hypotheses": [], "uncertainties": [],
            }))

    async def execute(*args, **kwargs):
        return await run_session(*args, model=FixedActor(), summary_model=FixedSummary(), **kwargs)

    migrate.main()
    monkeypatch.setattr(store, "connect", isolated_database)
    payload = make_manifest().model_dump(mode="json")
    payload["actor"]["history_interactions"] = 1
    manifest = Manifest.model_validate(payload)
    runtime = Settings("http://test", 10, "http://test", "fake", "local", 5, 1234, 3, 2, 8192)
    await run_series(manifest, runtime, binary=Path(os.environ["TEST_ROGUE_BINARY"]).resolve(),
                     revision="0" * 40, execute_trial=execute)

    output = tmp_path / "pilot"
    export_series(manifest.series_id, output)
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert [result["observed_reset_seed"] for result in results] == [1234, 1234]
    assert [result["last_observed_api_step"] for result in results] == [3, 3]
    assert [result["model_usage"]["episode_summary"]["calls"] for result in results] == [0, 1]
    evidence = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    revision, = [row for row in evidence if row["record_type"] == "memory_revision"]
    assert revision["trial_id"] == "second"
    assert any(row.get("memory_revision_id") == revision["id"] and row.get("purpose") == "decision"
               for row in evidence)
