"""Export one finalized series as canonical, linked evidence records."""

from hashlib import file_digest, sha256
import json
from pathlib import Path
import tempfile
from uuid import UUID

from psycopg.rows import dict_row

from game_runner.context.progress import movement_direction
from game_runner.pilot.store import PilotError, finalize_series, series_lock

EXPORT_VERSION = 1
OPPORTUNITY_VERSION = 1


def _canonical(record) -> bytes:
    return (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":"), default=str) + "\n").encode("utf-8")


def _rows(connection, query: str, params: tuple):
    with connection.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(query, params).fetchall()


def _opportunities(events: list[dict], calls: list[dict]) -> dict[str, list[dict]]:
    result = {key: [] for key in ("ambiguous_movement", "conditional_rest", "item_identity")}
    by_event = {event["id"]: event for event in events}
    by_call = {call["id"]: call for call in calls}
    decisions = {}
    for call in calls:
        if call["purpose"] == "decision":
            decisions.setdefault(call["observation_event_id"], call["id"])

    for event_id, call_id in decisions.items():
        event = by_event.get(event_id)
        if not event or not event["observation"]:
            continue
        observation = event["observation"]
        state = observation.get("state") or {}
        hp, maximum = state.get("hp"), state.get("max_hp")
        # Exclude the message and status rows. Uppercase dungeon glyphs are
        # threat cues only on an observation with a recognizable map position.
        threat = bool((event["progress"] or {}).get("position")) and any(
            any("A" <= glyph <= "Z" for glyph in row) for row in observation.get("screen", [])[1:-1]
        )
        if hp is not None and maximum is not None and hp < maximum and (state.get("hunger") or threat):
            result["conditional_rest"].append({"input_event_id": event_id, "model_call_id": call_id})

    for event in events:
        if event["outcome"] != "succeeded" or not event["observation"] or event["kind"] != "action":
            continue
        call = by_call.get(event["model_call_id"])
        action = event["action_request"]
        if not call or not action or call["observation_event_id"] not in by_event:
            continue
        progress = event["progress"] or {}
        refs = {"input_event_id": call["observation_event_id"], "output_event_id": event["id"],
                "model_call_id": call["id"]}
        if movement_direction(action) and progress.get("position_change") == "same":
            result["ambiguous_movement"].append(refs)
        before = by_event[call["observation_event_id"]]["observation"] or {}
        named_item = action.get("item") and action.get("action") in {"EAT", "QUAFF", "READ", "ZAP", "THROW"}
        key = action.get("key")
        selected_item = before.get("mode") == "select_item" and key and len(key) == 1 and "a" <= key <= "z"
        if named_item or selected_item:
            result["item_identity"].append(refs)
    return result


def _usage(group: list[dict]) -> dict:
    totals = [usage["total_tokens"] for call in group if isinstance(usage := call["usage_metadata"], dict)
              and type(usage.get("total_tokens")) is int and usage["total_tokens"] >= 0]
    latencies = [call["duration_ms"] for call in group if call["duration_ms"] is not None]
    counts: dict[str, dict[str, int]] = {}
    for call in group:
        method = call["input_token_count_method"]
        count = call["input_token_count"]
        if method is not None and count is not None:
            bucket = counts.setdefault(method, {"calls": 0, "input_tokens": 0})
            bucket["calls"] += 1
            bucket["input_tokens"] += count
    return {
        "calls": len(group), "latency_ms": sum(latencies) if latencies else None,
        "calls_with_latency": len(latencies),
        "provider_total_tokens": sum(totals) if totals else None,
        "calls_with_provider_usage": len(totals), "calls_missing_provider_usage": len(group) - len(totals),
        "local_input_counts": counts,
    }


def _summarize(trial: dict, events: list[dict], calls: list[dict]) -> dict:
    depths = [
        (event["id"], (event["observation"].get("state") or {}).get("dungeon_level"))
        for event in events if event["observation"]
    ]
    known_depths = [(event_id, depth) for event_id, depth in depths if depth is not None]
    level_changes = [
        {"from_event_id": before_id, "to_event_id": after_id, "from": before, "to": after}
        for (before_id, before), (after_id, after) in zip(depths, depths[1:])
        if before is not None and after is not None and before != after
    ]
    stalls = [{
        "event_id": event["id"], "api_step": event["api_step"], "streak": event["progress"]["no_movement_streak"],
    } for event in events if (event["progress"] or {}).get("no_movement_streak", 0) >= 2]
    opportunities = _opportunities(events, calls)
    return {
        **trial,
        "action_dispatches": sum(event["kind"] == "action" for event in events),
        "last_observed_api_step": max((event["api_step"] for event in events
                                       if event["observation"] and event["api_step"] is not None), default=None),
        "max_observed_depth": max((depth for _, depth in known_depths), default=None),
        "level_changes": level_changes, "stalls": stalls,
        "screening_opportunities": {family: {"count": len(records), "records": records}
                                    for family, records in opportunities.items()},
        "model_usage": {purpose: _usage([call for call in calls if call["purpose"] == purpose])
                        for purpose in ("decision", "episode_summary")},
    }


def export_series(series_id: UUID, output: Path) -> str:
    destination = output.resolve()
    if destination.exists():
        raise PilotError("output directory already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with series_lock(series_id) as connection:
        row = connection.execute("SELECT ended_at FROM pilot_series WHERE id = %s", (series_id,)).fetchone()
        if row is None:
            raise PilotError("series does not exist")
        if row[0] is None:
            finalize_series(connection, series_id, "abandoned")
        with tempfile.TemporaryDirectory(prefix=".pilot-export-", dir=destination.parent) as tmp:
            scratch = Path(tmp) / "bundle"
            scratch.mkdir()
            with connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                _write_snapshot(connection, series_id, scratch)
            hashes = []
            for name in ("manifest.json", "results.jsonl", "trajectory.jsonl"):
                with (scratch / name).open("rb") as stream:
                    hashes.append(f"{file_digest(stream, 'sha256').hexdigest()}  {name}\n")
            checksums = "".join(hashes).encode("ascii")
            bundle_hash = sha256(checksums).hexdigest()
            (scratch / "SHA256SUMS").write_bytes(checksums)
            (scratch / "bundle.sha256").write_text(bundle_hash + "\n", encoding="ascii")
            scratch.rename(destination)
    return bundle_hash


def _write_snapshot(connection, series_id: UUID, directory: Path) -> None:
    series, = _rows(connection,
                    "SELECT manifest, started_at, ended_at, stop_reason FROM pilot_series WHERE id = %s", (series_id,))
    trials = _rows(connection, """
        SELECT t.trial_id, t.ordinal, t.requested_seed, t.context_policy, t.session_id,
               t.outcome, t.reason, t.started_at, t.ended_at, s.episode_id,
               s.seed AS observed_reset_seed, s.status AS game_status
        FROM pilot_trials t LEFT JOIN game_sessions s ON t.session_id = s.id
        WHERE t.series_id = %s ORDER BY t.ordinal
        """, (series_id,))
    counts = {outcome: sum(trial["outcome"] == outcome for trial in trials)
              for outcome in ("ended", "limited", "failed", "interrupted", "unstarted")}
    (directory / "manifest.json").write_bytes(_canonical({
        **series["manifest"], "export_schema_version": EXPORT_VERSION,
        "opportunity_schema_version": OPPORTUNITY_VERSION,
        "execution": {"started_at": series["started_at"], "ended_at": series["ended_at"],
                      "stop_reason": series["stop_reason"], "counts": {"planned": len(trials), **counts}},
    }))
    queries = {
        "event": """
            SELECT id, model_call_id, kind, action_request, api_step, http_status, observation,
                   outcome, progress, requested_at, completed_at
            FROM game_events WHERE session_id = %s ORDER BY id
        """,
        "model_call": """
            SELECT id, observation_event_id, memory_revision_id, purpose, status, input_messages,
                   parsed_action, decision_rationale, usage_metadata, duration_ms, created_at,
                   input_token_count, input_token_count_method
            FROM model_calls WHERE session_id = %s ORDER BY id
        """,
        "memory_revision": """
            SELECT id, previous_revision_id, through_event_id, schema_version, policy_version,
                   content, model_call_id, gap_event_ids, created_at
            FROM episode_memory_revisions WHERE session_id = %s ORDER BY id
        """,
    }
    with (directory / "results.jsonl").open("wb") as results, (directory / "trajectory.jsonl").open("wb") as trajectory:
        for trial in trials:
            identity = {"series_id": series_id, "trial_id": trial["trial_id"], "session_id": trial["session_id"]}
            source = {kind: _rows(connection, query, (trial["session_id"],)) if trial["session_id"] else []
                      for kind, query in queries.items()}
            results.write(_canonical({"series_id": series_id, **_summarize(trial, source["event"], source["model_call"])}))
            order = {"model_call": 0, "event": 1, "memory_revision": 2}
            records = [(kind, record) for kind, rows in source.items() for record in rows]
            records.sort(key=lambda entry: (
                entry[1].get("created_at", entry[1].get("requested_at")), order[entry[0]], entry[1]["id"],
            ))
            for kind, record in records:
                trajectory.write(_canonical({**identity, "record_type": kind, **record}))
