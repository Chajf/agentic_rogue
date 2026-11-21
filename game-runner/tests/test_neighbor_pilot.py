import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from game_runner import migrate
from game_runner.config import Settings
from game_runner.context.decision_support import neighbors_suffix
from game_runner.context.messages import build_messages
from game_runner.pilot import store
from game_runner.pilot.export import export_series
from game_runner.pilot.manifest import Manifest
from game_runner.pilot.runner import run_series
from game_runner.pilot.trajectory import analyze
from game_runner.pilot.trajectory_report import write_report
from game_runner.prompts.system import actor_prompt_version
from game_runner.rule_feasibility.artifacts import FeasibilityError
from game_runner.rule_feasibility.corpus import load
from game_runner.session import run_session

from test_agent_graph import FakeClient, FakeModel, FakeRepository, observed
from test_pilot_store import isolated_database, make_manifest  # noqa: F401
from test_rule_corpus import sample_result, write_bundle

RUNTIME = Settings("http://test", 10, "http://test", "fake", "local", 5, 1234, 3, 2, 8192)
REST = '{"action":{"type":"semantic","action":"REST"},"rationale":"Wait."}'
CUE = "Visible neighbors (null means unknown): "


def _manifest_data():
    payload = make_manifest().model_dump(mode="json")
    payload["trials"][1].update(context_policy="history_only", context_variant="neighbors")
    payload["trials"].append({**payload["trials"][0], "trial_id": "repeat"})
    payload["series_limits"] = {"actions": 9, "model_calls": 15, "seconds": 30}
    return payload


def test_neighbor_manifest_rejects_missing_baseline_and_confounded_trials():
    payload = _manifest_data()
    payload["trials"][1]["seed"] += 1
    with pytest.raises(ValueError, match="at least one baseline"):
        Manifest.model_validate(payload)
    payload = _manifest_data()
    payload["trials"][0]["context_policy"] = "working_memory"
    with pytest.raises(ValueError, match="history_only for all trials"):
        Manifest.model_validate(payload)


async def test_trial_setting_changes_only_the_confirmed_map_prompt():
    manifest = Manifest.model_validate(_manifest_data())
    episode = uuid4()
    frame = observed(episode, screen=["message", ".@|", "##.", "stats"])
    initial_messages = []
    for trial in manifest.trials[:2]:
        repository = FakeRepository()
        client = FakeClient(frame.model_copy(update={"api_step": 1}))
        client.initial = frame
        model = FakeModel(REST, REST, REST)
        await run_session(manifest.settings(RUNTIME, trial), repository, model=model, client=client)
        assert repository.prompt_version == actor_prompt_version(trial.context_variant)
        initial_messages.append(repository.calls[0]["input_messages"])
    baseline, neighbor = initial_messages
    assert baseline[:-1] == neighbor[:-1]
    assert neighbor[-1]["content"] == baseline[-1]["content"] + neighbors_suffix(frame.screen, (1, 1))
    assert CUE not in build_messages(frame, [], context_variant="neighbors", observation_outcome="uncertain")[-1].content
    inventory = frame.model_copy(update={"mode": "inventory"})
    assert CUE not in build_messages(inventory, [], context_variant="neighbors")[-1].content


def test_analysis_preserves_primary_pair_and_does_not_promote_a_repeat(tmp_path):
    payload = _manifest_data()
    payload.update(export_schema_version=1, opportunity_schema_version=1)
    results = [sample_result(payload, trial, index) for index, trial in enumerate(payload["trials"])]
    results[0].update(outcome="failed", reason="provider_error")
    results[1].update(outcome="ended", reason="dead", game_status="dead")
    results[2].update(outcome="ended", reason="won", game_status="won")
    frames = [
        {
            **{key: trial[key] for key in ("series_id", "trial_id", "session_id")},
            "record_type": "event", "id": index, "kind": "reset", "outcome": "succeeded", "api_step": 0,
            "observation": observed(trial["episode_id"]).model_copy(
                update={"seed": trial["seed"]},
            ).model_dump(mode="json"),
            "model_call_id": None, "progress": None,
        }
        for index, trial in enumerate(results, 1)
    ]
    bundle = tmp_path / "bundle"
    write_bundle(bundle, payload, results, frames)
    data, _ = analyze(load(bundle, "development"))
    summary = data["neighbors_comparison"]
    primary, = summary["primary_pairs"]
    repeat, = summary["baseline_repeats"]
    assert (primary["baseline_trial_id"], primary["comparison_trial_id"]) == ("first", "second")
    assert repeat["comparison_trial_id"] == "repeat"
    assert not primary["complete"] and primary["positions_delta"] is None
    assert summary["complete_primary_pairs"] == 0 and summary["neighbors_deaths"] == 1

    output = write_report(bundle, "development", tmp_path / "analysis")
    rendered = (output / "report.md").read_text()
    assert "0/1 complete primary pairs" in rendered
    assert "### Baseline repeats" in rendered
    assert "### Prompt exposure" in rendered

    results[1]["context_variant"] = "baseline"
    write_bundle(bundle, payload, results, frames)
    with pytest.raises(FeasibilityError, match="manifest trial"):
        load(bundle, "development")


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
async def test_native_neighbor_pilot_exports_exact_actor_prompt_and_variant(isolated_database, monkeypatch, tmp_path):
    migrate.main()
    monkeypatch.setattr(store, "connect", isolated_database)
    manifest = Manifest.model_validate(_manifest_data())

    async def execute(*args, **kwargs):
        return await run_session(*args, model=FakeModel(REST, REST, REST), **kwargs)

    await run_series(
        manifest, RUNTIME, binary=Path(os.environ["TEST_ROGUE_BINARY"]).resolve(),
        revision="0" * 40, execute_trial=execute,
    )
    output = tmp_path / "export"
    export_series(manifest.series_id, output)
    corpus = load(output, "development")
    metrics, _ = analyze(corpus)
    assert [trial["context_variant"] for trial in metrics["trials"]] == ["baseline", "neighbors", "baseline"]
    assert metrics["trials"][1]["prompt_version"] == actor_prompt_version("neighbors")
    assert [trial["neighbors_prompt_exposure"]["with_neighbors"] for trial in metrics["trials"]] == [0, 3, 0]
    for call in corpus.calls.values():
        assert call["purpose"] == "decision"
        assert (CUE in call["input_messages"][-1]["content"]) == (call["trial_id"] == "second")
    assert metrics["neighbors_comparison"]["complete_primary_pairs"] == 1
    assert metrics["neighbors_comparison"]["primary_pairs"][0]["positions_delta"] == 0
    assert metrics["neighbors_comparison"]["baseline_repeats"][0]["complete"]
