"""Report matched outcomes without filling missing observation or usage evidence."""

from uuid import UUID

from game_runner.pilot.evaluation_evidence import trial_evidence
from game_runner.pilot.manifest import Manifest
from game_runner.pilot.trajectory import analyze
from game_runner.rule_feasibility.corpus import load

from test_evaluation_manifest import evaluation_payload
from test_rule_corpus import write_bundle
from test_working_state import frame


def test_partial_trials_use_only_the_shared_confirmed_horizon(tmp_path):
    plan = evaluation_payload([("A", "B")], seed=101)
    plan["trials"].extend(evaluation_payload([("A", "B")], seed=102)["trials"])
    manifest = Manifest.model_validate(plan)
    traces = []
    results = []
    next_id = 1
    for index, trial in enumerate(manifest.trials):
        active = index != 3
        session = str(UUID(int=200 + index)) if active else None
        episode = str(UUID(int=300 + index)) if index < 2 else None
        identity = {"series_id": plan["series_id"], "trial_id": trial.trial_id, "session_id": session}
        events = []

        def event(kind, step, outcome, observation=None, call_id=None, action=None):
            nonlocal next_id
            row = {
                **identity, "record_type": "event", "id": next_id, "kind": kind, "api_step": step,
                "outcome": outcome, "observation": observation, "model_call_id": call_id,
                "action_request": action, "progress": None,
            }
            if observation:
                row["progress"] = {
                    "map_event_id": events[0]["id"] if events else next_id,
                    "position": [1, 1] if step == 0 else [1, 2],
                }
            next_id += 1
            events.append(row)
            return row

        if index < 2:
            def observed(step, *, dead=False):
                observation = frame(".@%.", "....") if step == 0 else frame("..@.", "....")
                return observation.model_copy(update={
                    "episode_id": UUID(episode), "seed": trial.seed, "api_step": step,
                    "status": "dead" if dead else "running", "terminated": dead,
                    "mode": "game_over" if dead else "normal",
                }).model_dump(mode="json")

            previous = event("reset", 0, "succeeded", observed(0))
            for step, outcome in enumerate(("succeeded", "uncertain", "failed") if index == 0 else ("succeeded",), 1):
                request = {"type": "key", "key": "l"}
                call_id = next_id
                next_id += 1
                traces.append({
                    **identity, "record_type": "model_call", "id": call_id, "purpose": "decision",
                    "observation_event_id": previous["id"], "memory_revision_id": None, "status": "valid",
                    "parsed_action": request, "decision_rationale": "Fixture action.", "input_messages": [],
                    "usage_metadata": None, "duration_ms": None,
                    "input_token_count": None, "input_token_count_method": None,
                })
                previous = event(
                    "action", step if outcome != "failed" else None, outcome,
                    observed(step, dead=index == 1) if outcome != "failed" else None, call_id, request,
                )
        elif index == 2:
            event("reset", None, "failed")
        traces.extend(events)
        result = {
            **identity, "ordinal": index, "requested_seed": trial.seed,
            "observed_reset_seed": trial.seed if episode else None, "episode_id": episode,
            "context_policy": trial.context_policy, "prompt_profile": trial.prompt_profile,
            "prompt_version": "rogue-v1.3" if trial.condition == "A" else "rules-v2",
            "outcome": ("interrupted", "ended", "failed", "unstarted")[index],
            "game_status": ("interrupted", "dead", "error", None)[index],
            "reason": ("action_unresolved", "dead", "reset_api_error", "series_call_limit")[index],
            "action_dispatches": sum(row["kind"] == "action" for row in events),
        }
        results.append({**result, **trial_evidence(result, trial.model_dump(mode="json"), events)})

    path = tmp_path / "evidence"
    write_bundle(path, {**plan, "export_schema_version": 2, "opportunity_schema_version": 1}, results, traces)
    metrics, ledger = analyze(load(path, "development"))
    assert metrics["trial_counts"] == {"interrupted": 1, "ended": 1, "failed": 1, "unstarted": 1}
    interrupted, dead, failed, unstarted = metrics["trials"]
    assert interrupted["confirmed_prefix"] == 1
    assert interrupted["missing_observations"] == interrupted["uncertain_dispatches"] == 1
    assert interrupted["model_usage"]["decision"]["provider_total_tokens"] is None
    assert interrupted["model_usage"]["decision"]["calls_missing_provider_usage"] == 3
    assert dead["game_status"] == "dead"
    assert failed["coverage"] is None and unstarted["coverage"] is None
    first, second = metrics["comparisons"][0]["pairs"]
    assert first["shared_horizon"] == 1 and not first["full_horizon"]
    assert first["delta"]["positions"] == -1
    assert second["shared_horizon"] == 0 and second["delta"] is None
    assert len([row for row in ledger if row["record_type"] == "decision"]) == 4
