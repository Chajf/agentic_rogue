import json
from uuid import uuid4

import pytest

from game_runner.rule_feasibility.artifacts import FAMILIES, FeasibilityError, canonical, digest
from game_runner.rule_feasibility.corpus import FILES, load, prepare


def write_bundle(path, manifest, results, trajectory):
    path.mkdir(exist_ok=True)
    files = (canonical(manifest), b"".join(map(canonical, results)), b"".join(map(canonical, trajectory)))
    for name, content in zip(FILES, files, strict=True):
        (path / name).write_bytes(content)
    checksums = "".join(f"{digest(content)}  {name}\n" for name, content in zip(FILES, files)).encode()
    (path / "SHA256SUMS").write_bytes(checksums)
    (path / "bundle.sha256").write_text(digest(checksums) + "\n")


def sample_manifest():
    return {
        "schema_version": 1, "export_schema_version": 1, "opportunity_schema_version": 1,
        "series_id": str(uuid4()), "split": "development",
        "actor": {"model_name": "fixture-model"},
        "trial_limits": {"actions": 3, "model_calls": 5, "seconds": 10},
        "series_limits": {"actions": 6, "model_calls": 10, "seconds": 20},
        "trials": [],
    }


def sample_result(manifest, trial, ordinal=0, *, outcome="failed"):
    return {
        **trial, "series_id": manifest["series_id"], "ordinal": ordinal, "outcome": outcome,
        "session_id": str(uuid4()), "episode_id": str(uuid4()),
        "observed_reset_seed": trial["seed"], "requested_seed": trial["seed"],
        "screening_opportunities": {family: {"count": 0, "records": []} for family in FAMILIES},
    }


def empty_bundle(path):
    manifest = sample_manifest()
    trial = {"trial_id": "first", "seed": 3, "context_policy": "history_only"}
    manifest["trials"].append(trial)
    write_bundle(path, manifest, [sample_result(manifest, trial)], [])


def test_empty_screening_frame_checks_digest_and_preserves_failed_trial(tmp_path):
    empty_bundle(tmp_path)
    cases = prepare(load(tmp_path, "development"), "natural")
    assert cases["screened"] == dict.fromkeys(FAMILIES, 0)
    assert cases["cases"] == cases["exclusions"] == []
    assert cases["trial_counts"]["failed"] == 1
    with pytest.raises(FeasibilityError, match="split"):
        load(tmp_path, "pilot_evaluation")
    (tmp_path / "trajectory.jsonl").write_text("{}\n")
    with pytest.raises(FeasibilityError, match="checksum"):
        load(tmp_path, "development")


def test_bundle_rejects_cross_session_and_split_links(tmp_path):
    manifest = sample_manifest()
    manifest["trials"] = [
        {"trial_id": "first", "seed": 3, "context_policy": "history_only"},
        {"trial_id": "second", "seed": 4, "context_policy": "history_only"},
    ]
    first, second = [sample_result(manifest, trial, i) for i, trial in enumerate(manifest["trials"])]
    owner = {key: first[key] for key in ("series_id", "trial_id", "session_id")}
    event = {**owner, "record_type": "event", "id": 1, "observation": None, "model_call_id": None}
    call = {**owner, "record_type": "model_call", "id": 1, "observation_event_id": 1, "memory_revision_id": None}
    write_bundle(tmp_path, manifest, [first, second], [event, {**call, "split": "pilot_evaluation"}])
    with pytest.raises(FeasibilityError, match="split"):
        load(tmp_path, "development")
    write_bundle(tmp_path, manifest, [first, second], [
        event, {**call, "trial_id": second["trial_id"], "session_id": second["session_id"]},
    ])
    with pytest.raises(FeasibilityError, match="cross-session"):
        load(tmp_path, "development")
