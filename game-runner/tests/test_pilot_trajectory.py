import json
import subprocess
import sys

import pytest

from game_runner.pilot.trajectory import analyze
from game_runner.rule_feasibility.artifacts import FeasibilityError
from game_runner.rule_feasibility.corpus import load
from test_agent_graph import observed
from test_rule_corpus import sample_manifest, sample_result, write_bundle


def _step(row, *, map_id=1, mode="normal", outcome="succeeded", message="", quiet=0):
    return {
        "row": row, "map_id": map_id, "mode": mode, "outcome": outcome, "message": message, "quiet": quiet,
    }


def _bundle(path, opening, steps):
    manifest = sample_manifest()
    manifest["trials"] = [
        {"trial_id": "played", "seed": 3, "context_policy": "history_only"},
        {"trial_id": "unstarted", "seed": 3, "context_policy": "working_memory"},
        {"trial_id": "no_observation", "seed": 4, "context_policy": "history_only"},
    ]
    played, skipped, failed = [
        sample_result(manifest, trial, ordinal) for ordinal, trial in enumerate(manifest["trials"])
    ]
    played.update(outcome="limited", action_dispatches=len(steps))
    skipped.update(outcome="unstarted", session_id=None, episode_id=None, observed_reset_seed=None, action_dispatches=0)
    failed.update(episode_id=None, observed_reset_seed=None, action_dispatches=0)
    owner = {key: played[key] for key in ("series_id", "trial_id", "session_id")}
    records = []
    for step, (choice, frame) in enumerate([(None, _step(opening)), *steps]):
        action = None
        call_id = None
        if choice:
            action = {"type": "key", "key": choice} if len(choice) == 1 else {"type": "semantic", "action": choice}
            call_id = 100 + step
            records.append({
                **owner, "record_type": "model_call", "id": call_id, "purpose": "decision", "status": "valid",
                "observation_event_id": step, "memory_revision_id": None, "parsed_action": action,
                "decision_rationale": "Fixture decision.",
            })
        row = frame["row"]
        observation = None
        position = None
        if row is not None:
            observation = observed(
                played["episode_id"], step, mode=frame["mode"], message=frame["message"],
                screen=["", row, ""],
            ).model_dump(mode="json")
            observation["seed"] = played["seed"]
            observation["messages"] = [frame["message"]] if frame["message"] else []
            if "@" in row:
                position = [1, row.index("@")]
        records.append({
            **owner, "record_type": "event", "id": step + 1, "kind": "action" if step else "reset",
            "api_step": step if observation else None, "outcome": frame["outcome"], "observation": observation,
            "model_call_id": call_id, "action_request": action,
            "progress": {
                "position": position, "map_event_id": frame["map_id"], "no_movement_streak": frame["quiet"],
            } if observation else None,
        })
    write_bundle(path, manifest, [played, skipped, failed], records)
    return path


def test_coverage_tracks_returns_and_breaks_runs_at_menu_gap_and_map_change(tmp_path):
    bundle = _bundle(tmp_path, ".@....", [
        ("MOVE_RIGHT", _step("..@...")),
        ("h", _step(".@....")),
        ("MOVE_RIGHT", _step("..@...")),
        ("INVENTORY", _step("a) Fake item @", mode="inventory")),
        ("l", _step("..@...")),
        ("MOVE_LEFT", _step(".@....", outcome="uncertain")),
        ("MOVE_RIGHT", _step("..@...", map_id=8)),
        ("MOVE_LEFT", _step(".@....", map_id=8)),
        ("MOVE_RIGHT", _step(None, outcome="failed", map_id=8)),
    ])
    metrics, rows = analyze(load(bundle, "development"))
    trial, skipped, failed = metrics["trials"]
    assert metrics["trial_counts"] == {"limited": 1, "unstarted": 1, "failed": 1}
    assert trial["dispatch_outcomes"] == {"succeeded": 7, "uncertain": 1, "failed": 1}
    assert trial["coverage"] == {
        "checkpoints": [
            {"api_step": limit, "observed_through_api_step": 8, "positions": 4}
            for limit in (25, 50, 100, 200)
        ],
        "positions": 4,
        "confirmed_position_outputs": 6,
        "last_new_position_api_step": 8,
        "actions_since_last_new_position": 1,
        "longest_no_new_position_move_run": {"actions": 2, "start_event_id": 3, "end_event_id": 4},
        "immediate_returns": 2,
    }
    assert [row["output_event_id"] for row in rows if row["returned_to_previous_position"]] == [3, 4]
    assert rows[4]["movement_direction"] is None
    assert rows[5]["output_location"] is None
    assert rows[-1]["output_location"] is None
    for unobserved in (skipped, failed):
        assert unobserved["coverage"]["checkpoints"][0]["positions"] is None
        assert unobserved["last_observed_api_step"] is None
    assert metrics["matched_seed_pairs"][0]["positions_delta"] is None


def test_wall_opportunities_attacks_and_stairs_remain_distinct(tmp_path):
    bundle = _bundle(tmp_path, "|@E.%..", [
        ("MOVE_LEFT", _step("|@E.%..", quiet=1)),
        ("h", _step("|@E.%..", quiet=2)),
        ("MOVE_RIGHT", _step("|@..%..", message="You have defeated the emu.")),
        ("MOVE_RIGHT", _step("|.@.%..")),
        ("MOVE_RIGHT", _step("|..@%..")),
        (">", _step("|..@%..", message="I see no way down.")),
        ("l", _step("|...@..")),
        ("DESCEND", _step("|...@..", map_id=9)),
        ("DESCEND", _step("|...@..", map_id=9)),
    ])
    metrics, rows = analyze(load(bundle, "development"))
    trial = metrics["trials"][0]
    assert trial["movement"]["repeat_wall_opportunities"] == 2
    assert trial["movement"]["repeated_wall_moves"] == 1
    repeat, attack = rows[1:3]
    assert repeat["repeated_wall_move"]
    assert attack["repeat_wall_opportunity_event_id"] == repeat["output_event_id"]
    assert attack["combat_feedback"] and attack["displaced"] is False
    assert trial["movement"]["same_position_combat"] == 1
    assert trial["stairs"] == {
        "outputs_on_known_stairs": 1,
        "on_stairs_opportunities": 1,
        "adjacent_stairs_opportunities": 2,
        "descent_attempts": 3,
        "descents_attempted_on_stairs": 1,
        "descents_attempted_adjacent": 1,
    }
    assert rows[5]["adjacent_stair_source_event_ids"] == [6]
    assert rows[7]["on_known_stairs_source_event_id"] == 7
    assert rows[8]["on_known_stairs_source_event_id"] is None


def test_stairs_do_not_survive_a_recovery_gap(tmp_path):
    bundle = _bundle(tmp_path, ".@.%..", [
        ("MOVE_RIGHT", _step("..@%..", outcome="uncertain")),
        ("MOVE_RIGHT", _step("...@..", map_id=3)),
        ("DESCEND", _step("...@..", map_id=3)),
    ])
    metrics, rows = analyze(load(bundle, "development"))
    assert metrics["trials"][0]["stairs"]["descents_attempted_on_stairs"] == 0
    assert rows[1]["adjacent_stair_source_event_ids"] == []


def test_descent_attempt_can_be_observed_without_a_position(tmp_path):
    bundle = _bundle(tmp_path, "......", [(">", _step("......"))])
    metrics, rows = analyze(load(bundle, "development"))
    assert rows[0]["descent_attempt"]
    assert metrics["trials"][0]["stairs"]["on_stairs_opportunities"] == 0


def test_refuses_disagreement_between_dispatch_and_accepted_action(tmp_path):
    bundle = _bundle(tmp_path, ".@....", [("MOVE_RIGHT", _step("..@..."))])
    corpus = load(bundle, "development")
    corpus.calls[101]["parsed_action"] = {"type": "semantic", "action": "MOVE_LEFT"}
    with pytest.raises(FeasibilityError, match="differs from its accepted decision"):
        analyze(corpus)


def test_cli_writes_repeatable_reports_and_preserves_existing_output(tmp_path, monkeypatch):
    corpus = _bundle(tmp_path / "export", ".@....", [("MOVE_RIGHT", _step("..@..."))])
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    command = [
        sys.executable, "-m", "game_runner.pilot", "analyze", "--corpus", str(corpus), "--split", "development",
    ]
    first = tmp_path / "first"
    second = tmp_path / "second"
    subprocess.run([*command, "--output", str(first)], check=True, capture_output=True)
    subprocess.run([*command, "--output", str(second)], check=True, capture_output=True)
    assert sorted(path.name for path in first.iterdir()) == ["decision_ledger.jsonl", "report.md", "run_metrics.json"]
    assert all(path.read_bytes() == (second / path.name).read_bytes() for path in first.iterdir())
    ledger = (first / "decision_ledger.jsonl").read_text()
    assert json.loads(ledger)["new_position"]
    assert "## Matched seeds" in (first / "report.md").read_text()

    duplicate = subprocess.run([*command, "--output", str(first)], capture_output=True, text=True)
    assert duplicate.returncode != 0 and "already exists" in duplicate.stderr
    assert (first / "decision_ledger.jsonl").read_text() == ledger

    (corpus / "trajectory.jsonl").write_text("{}\n")
    invalid = tmp_path / "invalid"
    damaged = subprocess.run([*command, "--output", str(invalid)], capture_output=True, text=True)
    assert damaged.returncode != 0 and "checksum mismatch" in damaged.stderr
    assert not invalid.exists()
