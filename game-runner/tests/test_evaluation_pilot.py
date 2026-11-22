"""Exercise Phase 5 through the native CLI, persistence, and frozen analysis."""

import json
import os
import sys
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from game_runner.context.messages import SETUP_LABEL
from game_runner.pilot import store
from game_runner.pilot.__main__ import main as pilot_main
from game_runner.pilot.manifest import Manifest
from game_runner.pilot.trajectory import analyze
from game_runner.prompts.goals import GOAL_SELECTOR_INSTRUCTION
from game_runner.rule_feasibility.artifacts import FeasibilityError
from game_runner.rule_feasibility.corpus import load

from test_agent_graph import FakeModel
from test_evaluation_manifest import evaluation_payload
from test_pilot_store import isolated_database  # noqa: F401
from test_rule_corpus import write_bundle
from test_working_state_pilot import STAIR_ACTIONS

SCENARIOS = (
    ("unknown-stairs", 4190252879, "", "h", {"kind": "new_position"}),
    ("known-doorway", 4190252879, "hhhh", "h", {"kind": "reach_target", "target": [4, 68]}),
    ("stairs-under-player", 3, "hkk", ">", {"kind": "descend_here"}),
    ("same-position-combat", 412399380, "l", "l", {"kind": "same_position_combat"}),
    ("map-transition", 3, "hkk>", ".", {"kind": "observe_current_map"}),
)


def cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["game_runner.pilot", *map(str, args)])
    pilot_main()


def use_model(monkeypatch, actions_by_trial):
    class EvaluationModel(FakeModel):
        async def ainvoke(self, messages):
            if messages[0].content == GOAL_SELECTOR_INSTRUCTION:
                evidence = json.loads(messages[-1].content.split("\nGoal evidence:\n", 1)[1])
                stair = next((row for row in evidence["opportunities"] if row["kind"] == "stairs"), None)
                return AIMessage(content=json.dumps(
                    {"choice": "select", "candidate_id": stair["candidate_id"]} if stair else {"choice": "defer"}
                ))
            return await super().ainvoke(messages)

    trials = iter(actions_by_trial)

    def init(*args, **kwargs):
        return EvaluationModel(*(json.dumps({"action": action, "rationale": "Fixture action."}) for action in next(trials)))

    monkeypatch.setattr("game_runner.session.init_chat_model", init)


def run_and_export(monkeypatch, tmp_path, name, payload):
    payload["series_id"] = str(uuid4())
    manifest_path = tmp_path / f"{name}.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    cli(monkeypatch, "run", "--manifest", manifest_path, "--rogue-binary", os.environ["TEST_ROGUE_BINARY"],
        "--runner-revision", "0" * 40)
    output = tmp_path / f"{name}-export"
    cli(monkeypatch, "export", "--series-id", payload["series_id"], "--output", output)
    return load(output, payload["split"]), output


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
def test_native_evaluation_matrix_and_held_out_seed_lock(isolated_database, monkeypatch, tmp_path):
    monkeypatch.setattr(store, "connect", isolated_database)
    monkeypatch.setenv("MODEL_BASE_URL", "http://test")

    for name, contrasts in (
        ("development", [("A", "B"), ("B", "C"), ("C", "C-compact")]),
        ("goals-replay", [("C", "D")]),
        ("goals-compact", [("C-compact", "D")]),
    ):
        payload = evaluation_payload(contrasts)
        use_model(monkeypatch, [STAIR_ACTIONS] * len(payload["trials"]))
        corpus, output = run_and_export(monkeypatch, tmp_path, name, payload)
        metadata = json.loads((output / "manifest.json").read_bytes())
        assert metadata["export_schema_version"] == 2
        assert metadata["provenance"]["rogue_timeout_seconds"] == payload["timeouts"]["rogue_seconds"]
        metrics, ledger = analyze(corpus)
        assert len(metrics["comparisons"]) == len(contrasts)
        for trial in metrics["trials"]:
            assert trial["outcome"] == "limited" and trial["max_observed_depth"] == 2
            decisions = [row for row in ledger if row["trial_id"] == trial["trial_id"] and row["record_type"] == "decision"]
            assert [row["action"] for row in decisions] == STAIR_ACTIONS
            assert all(corpus.calls[row["call_id"]]["raw_response"]["content"] for row in decisions)
            if trial["condition"] == "D":
                assert trial["goals"]["completed_goal_ids"] == [1]
                assert trial["model_usage"]["goal_selection"]["calls_missing_provider_usage"] == 1
                review, = [row for row in ledger if row["record_type"] == "goal_review"]
                assert review["action_outcome"] == "succeeded" and review["actor_call_id"]
        assert all(comparison["full_horizon_pairs"] == 1 for comparison in metrics["comparisons"])

    holdout = evaluation_payload([("C-compact", "D")], split="held_out")
    holdout["series_id"] = str(uuid4())
    from game_runner.pilot.store import PilotError, create_series

    with isolated_database() as connection:
        with pytest.raises(PilotError, match="already belongs to the development split"):
            create_series(connection, Manifest.model_validate(holdout), {})
    fresh = evaluation_payload([("A", "B")], split="held_out", seed=12345)
    use_model(monkeypatch, [[{"type": "key", "key": "."}] * 4] * 2)
    corpus, _ = run_and_export(monkeypatch, tmp_path, "fresh-holdout", fresh)
    assert corpus.manifest.split == "held_out" and len(corpus.trials) == 2


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
def test_native_diagnostics_and_reproducible_reports(isolated_database, monkeypatch, tmp_path):
    monkeypatch.setattr(store, "connect", isolated_database)
    monkeypatch.setenv("MODEL_BASE_URL", "http://test")
    actions_by_trial = []
    for scenario, seed, prefix, measured, objective in SCENARIOS:
        spec_path = tmp_path / f"{scenario}-input.json"
        spec_path.write_text(json.dumps({
            "scenario_id": scenario, "seed": seed, "objective": objective,
            "prefix": [{"type": "key", "key": key} for key in prefix],
        }), encoding="utf-8")
        frozen_path = tmp_path / f"{scenario}-frozen.json"
        cli(monkeypatch, "prepare-diagnostic", "--spec", spec_path, "--rogue-binary", os.environ["TEST_ROGUE_BINARY"],
            "--output", frozen_path)
        frozen = json.loads(frozen_path.read_bytes())
        pair = evaluation_payload([("C", "D")], seed=seed, spec=frozen)
        if not actions_by_trial:
            payload = pair
        else:
            payload["trials"].extend(pair["trials"])
        actions_by_trial.extend([[{"type": "key", "key": measured}]] * 2)
    payload["trial_limits"]["actions"] = 1

    use_model(monkeypatch, actions_by_trial)
    corpus, output = run_and_export(monkeypatch, tmp_path, "diagnostic", payload)
    metrics, ledger = analyze(corpus)
    assert metrics["comparisons"][0]["full_horizon_pairs"] == 5
    starts = {}
    for trial in metrics["trials"]:
        assert trial["action_dispatches"] == 1
        assert trial["anchor_api_step"] == trial["setup_dispatches"]
        assert trial["anchor_verification"]["verified"]
        assert trial["diagnostic_outcome"]["decision_support"] == "unknown"
        call, = [item for item in corpus.calls.values()
                 if item["trial_id"] == trial["trial_id"] and item["purpose"] == "decision"]
        final_message = call["input_messages"][-1]["content"]
        assert final_message.count("Current 80x24 screen") == 1
        setup = json.JSONDecoder().raw_decode(final_message.split(f"\n{SETUP_LABEL}\n")[1])[0]
        signature = (trial["anchor_verification"]["anchor_sha256"], setup)
        assert starts.setdefault(trial["scenario_id"], signature) == signature
        assert call["observation_event_id"] == trial["anchor_event_id"]
        assert call["history_window"]["delivered_output_event_ids"] == []
        if trial["scenario_id"] == "stairs-under-player":
            assert trial["diagnostic_outcome"]["objective_status"] == "observed"
    for name in ("report-one", "report-two"):
        cli(monkeypatch, "analyze", "--corpus", output, "--split", "development", "--output", tmp_path / name)
    for name in ("report.md", "run_metrics.json", "decision_ledger.jsonl"):
        assert (tmp_path / "report-one" / name).read_bytes() == (tmp_path / "report-two" / name).read_bytes()

    metadata = json.loads((output / "manifest.json").read_bytes())
    trajectory = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    scripted = next(row for row in trajectory if row.get("scripted_input_event_id"))
    scripted["scripted_input_event_id"] = next(row["id"] for row in trajectory if row["record_type"] == "event")
    corrupt = tmp_path / "invalid-input-link"
    write_bundle(corrupt, metadata, list(corpus.trials.values()), trajectory)
    with pytest.raises(FeasibilityError, match="outside its linked observation boundary"):
        load(corrupt, "development")

    frozen = json.loads((tmp_path / "stairs-under-player-frozen.json").read_bytes())
    frozen["anchor_sha256"] = "0" * 64
    bad = evaluation_payload([("C", "D")], seed=3, spec=frozen)
    use_model(monkeypatch, [[], []])
    failed, _ = run_and_export(monkeypatch, tmp_path, "mismatch", bad)
    assert not failed.calls
    assert all(trial["reason"] == "diagnostic_mismatch" for trial in failed.trials.values())
    assert all(trial["anchor_event_id"] is None and trial["setup_dispatches"] == 3 for trial in failed.trials.values())
    assert all(trial["anchor_verification"]["verified"] is False for trial in failed.trials.values())
