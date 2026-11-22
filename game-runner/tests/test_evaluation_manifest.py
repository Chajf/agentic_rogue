from uuid import UUID

import pytest

from game_runner.pilot.manifest import Manifest

from test_pilot_store import make_manifest


def evaluation_payload(contrasts, *, split="development", seed=3, spec=None):
    profiles = {
        "A": ("legacy", "off", "replay", "off"),
        "B": ("rules-v2", "off", "replay", "off"),
        "C": ("rules-v2", "map-v1", "replay", "off"),
        "C-compact": ("rules-v2", "map-v1", "compact", "off"),
    }
    for reference, treatment in contrasts:
        if treatment == "D":
            profiles["D"] = (*profiles[reference][:3], "managed-v1")
    trials = []
    for condition in dict.fromkeys(condition for pair in contrasts for condition in pair):
        prompt, state, history, goals = profiles[condition]
        trials.append({
            "trial_id": f"{spec['scenario_id'] if spec else seed}-{condition}", "condition": condition,
            "seed": seed, "context_policy": "history_only",
            "prompt_profile": prompt, "structured_state": state,
            "history_format": history, "goal_management": goals,
            **({"diagnostic": spec} if spec else {}),
        })
    return {
        "schema_version": 2, "series_id": str(UUID(int=100)), "split": split,
        "timeouts": {"model_seconds": 10, "rogue_seconds": 10},
        "actor": {"model_name": "fixture", "context_tokens": 16384},
        "trial_limits": {"actions": 4, "model_calls": 12, "seconds": 60},
        "series_limits": {"actions": 80, "model_calls": 240, "seconds": 240},
        "contrasts": [{"reference": reference, "treatment": treatment} for reference, treatment in contrasts],
        "trials": trials,
    }


def test_evaluation_development_and_held_out_use_independent_profile_factors():
    dev = evaluation_payload([("A", "B"), ("B", "C"), ("C", "C-compact")])
    assert [trial.condition for trial in Manifest.model_validate(dev).trials] == ["A", "B", "C", "C-compact"]
    for reference in ("C", "C-compact"):
        payload = evaluation_payload([(reference, "D")], split="held_out")
        manifest = Manifest.model_validate(payload)
        assert manifest.trials[0].history_format == manifest.trials[1].history_format
        assert manifest.trials[1].goal_management == "managed-v1"
        payload["trials"][1]["history_format"] = "replay" if reference == "C-compact" else "compact"
        with pytest.raises(ValueError, match="do not match"):
            Manifest.model_validate(payload)


def test_evaluation_rejects_missing_pairs_and_diagnostic_holdouts():
    payload = evaluation_payload([("C", "D")])
    payload["trials"] = payload["trials"][:1]
    with pytest.raises(ValueError, match="must contain all"):
        Manifest.model_validate(payload)
    spec = {"scenario_id": "stairs", "seed": 3, "prefix": []}
    spec.update(binary_sha256="0"*64, anchor_sha256="0"*64, trace_sha256="0"*64)
    with pytest.raises(ValueError, match="separate development series"):
        Manifest.model_validate(evaluation_payload([("C", "D")], split="held_out", spec=spec))


def test_v1_snapshot_omits_new_evaluation_fields():
    old = make_manifest().snapshot()
    assert "contrasts" not in old and "timeouts" not in old
    assert all("condition" not in trial and "diagnostic" not in trial for trial in old["trials"])
    assert Manifest.model_validate(old).schema_version == 1
