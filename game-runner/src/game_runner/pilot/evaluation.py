"""Analyze versioned decision-loop contrasts from verified, frozen evidence."""

from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
from statistics import median

from game_runner.context.messages import SETUP_LABEL, WORKING_STATE_LABEL
from game_runner.pilot.export import _usage
from game_runner.pilot.trajectory import SeenStairs, _coverage, _decision_row, _map_view
from game_runner.prompts.goals import ACTOR_GOAL_GUIDANCE


def analyze_evaluation(corpus) -> tuple[dict, list[dict]]:
    events_by_trial = defaultdict(list)
    calls_by_trial = defaultdict(list)
    for event in sorted(corpus.events.values(), key=lambda row: row["id"]):
        events_by_trial[event["trial_id"]].append(event)
    for call in sorted(corpus.calls.values(), key=lambda row: row["id"]):
        calls_by_trial[call["trial_id"]].append(call)

    trials = []
    ledger = []
    traces = {}
    horizon = corpus.manifest.trial_limits.actions
    for plan in corpus.manifest.trials:
        trial = corpus.trials[plan.trial_id]
        events, calls = events_by_trial[plan.trial_id], calls_by_trial[plan.trial_id]
        rows, anchor = _measured_trace(trial, events, corpus)
        traces[plan.trial_id] = (rows, anchor)
        coverage = _measure(rows, anchor)
        for row in rows:
            ledger.append({"record_type": "decision", **row})
        actor_calls = [call for call in calls if call["purpose"] == "decision"]
        deliveries = [_delivery(call, trial) for call in actor_calls]
        for delivery in deliveries:
            ledger.append({"record_type": "actor_context", "trial_id": plan.trial_id, **delivery})
        goal_audit = _goal_audit(events, rows)
        for item in goal_audit["reviews"]:
            ledger.append({"record_type": "goal_review", "trial_id": plan.trial_id, **item})
        usages = {}
        for purpose in ("decision", "goal_selection"):
            group = [call for call in calls if call["purpose"] == purpose]
            usages[purpose] = {
                **_usage(group), "status_counts": dict(Counter(call["status"] for call in group)),
            }
        total_time = None
        if trial.get("started_at") and trial.get("ended_at"):
            start, end = (datetime.fromisoformat(trial[key]) for key in ("started_at", "ended_at"))
            total_time = round((end - start).total_seconds() * 1000)
        depth_changes = [
            {"event_id": row["output_event_id"], "from": row["depth_before"], "to": row["depth_after"]}
            for row in rows if row["depth_before"] is not None and row["depth_after"] is not None
            and row["depth_before"] != row["depth_after"]
        ]
        trials.append({
            **{key: trial.get(key) for key in (
                "trial_id", "condition", "scenario_id", "requested_seed", "outcome", "reason", "game_status",
                "prompt_profile", "prompt_version", "structured_state", "history_format", "goal_management",
                "setup_dispatches", "anchor_event_id", "anchor_api_step", "anchor_verification", "max_observed_depth",
            )},
            "action_dispatches": len(rows), "confirmed_prefix": _confirmed_prefix(rows),
            "missing_observations": sum(not row["observation_available"] for row in rows),
            "uncertain_dispatches": sum(row["outcome"] == "uncertain" for row in rows),
            "coverage": coverage, "wall_time_ms": total_time, "level_changes": depth_changes,
            "quiet_failed_moves": sum(row["quiet_failed_move"] for row in rows),
            "actor_context": {
                "calls": len(deliveries),
                "working_state_calls": sum(item["working_state_delivered"] for item in deliveries),
                "setup_evidence_calls": sum(item["setup_evidence_delivered"] for item in deliveries),
                "goal_context_calls": sum(item["goal_context_delivered"] for item in deliveries),
                "history_formats": dict(Counter(item["history_format"] for item in deliveries if item["history_format"])),
                "eligible_interactions": sum(len(item["eligible_history"] or ()) for item in deliveries),
                "delivered_interactions": sum(len(item["delivered_history"] or ()) for item in deliveries),
                "trimmed_calls": sum(item["history_trimmed"] for item in deliveries),
            },
            "goals": {key: value for key, value in goal_audit.items() if key != "reviews"},
            "diagnostic_outcome": _diagnostic_outcome(plan.diagnostic, rows, anchor),
            "model_usage": usages,
        })
    return {
        "schema_version": 2, "series_id": corpus.series_id, "split": corpus.manifest.split,
        "bundle_sha256": corpus.sha256, "comparison_horizon": horizon, "trials": trials,
        "trial_counts": dict(Counter(trial["outcome"] for trial in trials)),
        "comparisons": _comparisons(corpus.manifest.contrasts, trials, traces, horizon),
    }, ledger


def _measured_trace(trial, events, corpus):
    stairs = SeenStairs()
    rows = []
    anchor = corpus.events.get(trial["anchor_event_id"])
    for event in events:
        view = _map_view(event)
        if event["kind"] == "reset" or event.get("scripted_input_event_id") is not None:
            stairs.observe(event, view)
            continue
        call = corpus.calls[event["model_call_id"]]
        before = corpus.events[call["observation_event_id"]]
        previous = rows[-1] if rows else None
        row = _decision_row(event, before, call, previous, _map_view(before), view, stairs)
        confirmed = event["outcome"] == before["outcome"] == "succeeded" and event["observation"] is not None
        row.update(
            measured_dispatch=len(rows) + 1,
            observation_available=event["observation"] is not None,
            depth_before=before["observation"]["state"].get("dungeon_level") if confirmed else None,
            depth_after=event["observation"]["state"].get("dungeon_level") if confirmed else None,
            quiet_failed_move=row["displaced"] is False and ((event.get("progress") or {}).get("no_movement_streak") or 0) > 0,
            active_goal_id=(call.get("decision_metadata") or {}).get("active_goal_id"),
            goal_transition=(event.get("progress") or {}).get("goal_transition"),
        )
        stairs.observe(event, view)
        row["output_on_known_stairs_source_event_id"], _ = stairs.nearby(view)
        rows.append(row)
    return rows, anchor


def _measure(rows, anchor):
    if anchor is None:
        return None
    relative = [{**row, "api_step": row["measured_dispatch"]} for row in rows]
    last_observed = max((row["api_step"] for row in relative if row["observation_available"]), default=0)
    coverage = _coverage(relative, {**anchor, "api_step": 0}, last_observed)
    for row, computed in zip(rows, relative, strict=True):
        row["new_position"] = computed["new_position"]
    return coverage


def _confirmed_prefix(rows):
    length = 0
    for row in rows:
        if row["outcome"] != "succeeded" or not row["observation_available"]:
            break
        length += 1
    return length


def _comparisons(contrasts, trials, traces, horizon):
    indexed = {(trial["requested_seed"], trial["scenario_id"], trial["condition"]): trial for trial in trials}
    blocks = list(dict.fromkeys((trial["requested_seed"], trial["scenario_id"]) for trial in trials))
    comparisons = []
    for contrast in contrasts:
        pairs = []
        for seed, scenario in blocks:
            first = indexed[seed, scenario, contrast.reference]
            second = indexed[seed, scenario, contrast.treatment]
            shared = min(first["confirmed_prefix"], second["confirmed_prefix"], horizon)
            delta = None
            start_verified = first["anchor_event_id"] is not None and second["anchor_event_id"] is not None
            if start_verified and shared:
                left, right = [_measure(*_prefix(traces[trial["trial_id"]], shared)) for trial in (first, second)]
                delta = {
                    "positions": right["positions"] - left["positions"],
                    "longest_move_streak": (
                        right["longest_no_new_position_move_run"]["actions"]
                        - left["longest_no_new_position_move_run"]["actions"]
                    ),
                }
            pairs.append({
                "seed": seed, "scenario_id": scenario,
                "reference_trial_id": first["trial_id"], "treatment_trial_id": second["trial_id"],
                "start_verified": start_verified, "shared_horizon": shared,
                "full_horizon": start_verified and shared == horizon, "delta": delta,
            })
        full = [pair["delta"] for pair in pairs if pair["full_horizon"]]
        comparisons.append({
            "reference": contrast.reference, "treatment": contrast.treatment, "pairs": pairs,
            "full_horizon_pairs": len(full),
            "median_full_horizon_delta": {key: median(delta[key] for delta in full)
                                          for key in ("positions", "longest_move_streak")} if full else None,
        })
    return comparisons


def _prefix(trace, horizon):
    rows, anchor = trace
    return [row.copy() for row in rows[:horizon]], anchor


def _delivery(call, trial):
    messages = call["input_messages"]
    latest = messages[-1]["content"] if messages else ""
    system = next((message["content"] for message in messages if message["role"] == "system"), None)
    window = call.get("history_window") or {}
    eligible, delivered = window.get("eligible_output_event_ids"), window.get("delivered_output_event_ids")
    return {
        "call_id": call["id"], "input_event_id": call["observation_event_id"], "status": call["status"],
        "prompt_version": trial["prompt_version"],
        "system_sha256": sha256(system.encode()).hexdigest() if system is not None else None,
        "working_state_delivered": "\n" + WORKING_STATE_LABEL in latest,
        "setup_evidence_delivered": "\n" + SETUP_LABEL in latest,
        "goal_context_delivered": "\n" + ACTOR_GOAL_GUIDANCE in latest,
        "history_format": window.get("format"), "eligible_history": eligible, "delivered_history": delivered,
        "history_trimmed": eligible is not None and eligible != delivered,
        "input_token_count": call.get("input_token_count"), "input_token_count_method": call.get("input_token_count_method"),
    }


def _goal_audit(events, rows):
    outcomes = {row["input_event_id"]: row for row in rows}
    reviews = []
    completed, suspended = set(), set()
    reasons = Counter()
    statuses = Counter()
    for event in events:
        progress = event.get("progress") or {}
        if review := progress.get("goal_review"):
            reasons.update(review["reasons"].keys())
            statuses[review["status"]] += 1
            goal = (review.get("selection") or {}).get("goal") or {}
            if goal.get("status") == "suspended":
                suspended.add(goal["id"])
            outcome = outcomes.get(event["id"])
            reviews.append({
                "input_event_id": event["id"], "review": review,
                "actor_call_id": outcome["call_id"] if outcome else None,
                "output_event_id": outcome["output_event_id"] if outcome else None,
                "action_outcome": outcome["outcome"] if outcome else None,
                "goal_transition": outcome["goal_transition"] if outcome else None,
            })
        transition = progress.get("goal_transition") or {}
        if transition.get("outcome") == "complete":
            completed.add(transition["goal_id"])
    return {
        "reviews": reviews, "review_reasons": dict(reasons), "review_statuses": dict(statuses),
        "completed_goal_ids": sorted(completed), "suspended_goal_ids": sorted(suspended),
    }


def _diagnostic_outcome(spec, rows, anchor):
    if spec is None:
        return None
    objective = spec.objective
    result = {"objective": objective.kind if objective else None, "objective_status": "unknown",
              "evidence_event_id": None, "decision_support": "unknown"}
    if objective is None or anchor is None:
        return result
    anchor_map = (anchor["progress"] or {}).get("map_event_id")
    for row in rows:
        if row["outcome"] != "succeeded" or not row["observation_available"]:
            continue
        location = row["output_location"]
        if objective.kind == "new_position":
            observed = row["new_position"]
        elif objective.kind == "reach_target":
            observed = location is not None and location == (anchor_map, *objective.target)
        elif objective.kind == "descend_here":
            observed = (
                row["on_known_stairs_source_event_id"] is not None and row["descent_attempt"]
                and row["depth_before"] is not None and row["depth_after"] == row["depth_before"] + 1
            )
        elif objective.kind == "same_position_combat":
            observed = row["displaced"] is False and row["combat_feedback"]
        else:
            observed = location is not None and location[0] == anchor_map
        if observed:
            return {**result, "objective_status": "observed", "evidence_event_id": row["output_event_id"]}
    if rows and _confirmed_prefix(rows) == len(rows):
        result["objective_status"] = "not_observed"
    return result
