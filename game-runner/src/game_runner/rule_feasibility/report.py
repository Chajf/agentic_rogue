"""Deterministic accounting and feasibility conclusions from offline artifacts."""

from collections import Counter
from pathlib import Path

from game_runner.decision import parse_decision

from .artifacts import FAMILIES, FeasibilityError
from .probes import CONDITIONS


def checked_results(frozen: dict, frozen_digest: str, artifact: dict) -> dict[str, dict]:
    if (artifact["frozen_sha256"] != frozen_digest or artifact["corpus_sha256"] != frozen["corpus_sha256"]
            or artifact["split"] != frozen["split"] or artifact["origin"] != frozen["origin"]):
        raise FeasibilityError("probe results do not belong to this frozen specification")
    cases = {case["probe_id"]: case for case in frozen["probes"]}
    results = {probe_id: {} for probe_id in cases}
    calls = {call["id"]: call for call in artifact["analysis"]["calls"]}
    for row in artifact["results"]:
        probe_id, condition = row["probe_id"], row["condition"]
        if probe_id not in cases or condition not in CONDITIONS or condition in results[probe_id]:
            raise FeasibilityError("missing or duplicate paired case reference")
        if row["status"] not in {"valid", "invalid", "error", "unstarted"}:
            raise FeasibilityError("unknown probe outcome")
        if row["status"] != "unstarted":
            call = calls.get(row.get("call_id"))
            if call is None or call["label"] != f"{probe_id}:{condition}":
                raise FeasibilityError("probe result has no matching analysis call")
        if row["status"] == "valid":
            decision = parse_decision(call["raw_response"], cases[probe_id]["mode"])
            if (row["action"] != decision.action.model_dump(mode="json", exclude_none=True)
                    or row["rationale"] != decision.rationale):
                raise FeasibilityError("probe decision disagrees with the recorded reply")
            rubric = cases[probe_id]["rubric"]
            score = next((label for label, actions in rubric.items() if row["action"] in actions), "indeterminate")
            if row["score"] != score:
                raise FeasibilityError("probe score disagrees with the frozen human rubric")
        results[probe_id][condition] = row
    if any(set(group) != set(CONDITIONS) for group in results.values()):
        raise FeasibilityError("incomplete paired result artifact")
    return results


def _usage(calls: list[dict]) -> dict:
    totals = [call["usage"]["total_tokens"] for call in calls
              if call["usage"] is not None and "total_tokens" in call["usage"]]
    cost = [call["estimated_cost_usd"] for call in calls if call["estimated_cost_usd"] is not None]
    return {
        "calls": len(calls), "calls_with_usage": len(totals), "calls_missing_usage": len(calls) - len(totals),
        "total_tokens": sum(totals) if totals else None,
        "estimated_cost_usd": round(sum(cost), 9) if cost else None,
        "calls_with_estimated_cost": len(cost),
    }


def summarize(cases: dict, candidates: dict | None = None, frozen: dict | None = None,
              results: dict[str, dict] | None = None, probe_calls: list[dict] | None = None) -> dict:
    families = {}
    candidate_rows = {row["case_id"]: row for row in candidates["candidates"]} if candidates else {}
    audit_rows = {row["candidate_id"]: row for row in frozen["audits"]} if frozen else {}
    for family in FAMILIES:
        selected = [case for case in cases["cases"] if case["family"] == family]
        ids = {case["case_id"] for case in selected}
        audits = [audit_rows[key] for key in ids if key in audit_rows]
        counterevents = {event for audit in audits for event in audit["counterevidence_event_ids"]}
        probes = [probe for probe in frozen["probes"] if probe["family"] == family] if frozen else []
        counts = Counter()
        details = []
        transfer_by_candidate: dict[str, set[str]] = {}
        changed_transfer = set()
        family_call_ids = set()
        for case in probes:
            group = (results or {}).get(case["probe_id"], {})
            outcome = {key: row["score"] if row["status"] == "valid" else row["status"]
                       for key, row in group.items()}
            for status in outcome.values():
                if status in {"invalid", "unstarted", "error"}:
                    counts[f"{status}_replies"] += 1
            family_call_ids.update(row["call_id"] for row in group.values() if "call_id" in row)
            if group and all(row["status"] == "valid" for row in group.values()):
                counts["complete_triplets"] += 1
                if case["relation"] == "transfer":
                    transfer_by_candidate.setdefault(case["candidate_id"], set()).add(case["applicability"])
            changed = None
            avoided = lost = False
            if "no_rule" in group and group["no_rule"]["status"] == "valid":
                counts[f"no_rule_{group['no_rule']['score']}"] += 1
            if group and all(group[key]["status"] == "valid" for key in ("original", "corrected")):
                original, corrected = group["original"], group["corrected"]
                changed = original["action"] != corrected["action"]
                if changed and case["relation"] == "transfer" and group["no_rule"]["status"] == "valid":
                    changed_transfer.add(case["candidate_id"])
                counts["changed" if changed else "unchanged"] += 1
                avoided = original["score"] == "harmful" and corrected["score"] == "acceptable"
                lost = original["score"] == "acceptable" and corrected["score"] == "harmful"
                counts["harmful_choices_avoided"] += int(avoided)
                counts["appropriate_recommendations_lost"] += int(lost)
            details.append({
                "probe_id": case["probe_id"], "relation": case["relation"],
                "applicability": case["applicability"], "outcomes": outcome,
                "changed": changed, "harmful_choice_avoided": avoided, "appropriate_recommendation_lost": lost,
            })
        episodes = {case["episode_id"] for case in selected}
        independent_seeds = {case["seed"] for case in selected if case["seed"] is not None}
        paired_candidates = {
            key for key, contexts in transfer_by_candidate.items() if contexts == {"matching", "out_of_scope"}
            and audit_rows[key]["counterevidence_event_ids"]
        }
        family_usage = _usage([call for call in probe_calls or [] if call["id"] in family_call_ids])
        if cases["origin"] != "natural":
            decision, reason = "insufficient evidence", "Fixture results verify the workflow only."
        elif len(independent_seeds) < 2 or len(episodes) < 2:
            decision, reason = "insufficient evidence", "Too few independent source episodes."
        elif not paired_candidates:
            decision, reason = "insufficient evidence", "Independent audits, counterevidence, or transfer pairs are missing."
        elif not paired_candidates & changed_transfer:
            decision, reason = "revise pilot", "The completed probes showed no action change after correction."
        elif family not in frozen["cost_viable_families"] or not family_usage["calls_with_usage"]:
            # A reproducible report cannot certify that cost is viable without
            # an explicit budget judgment for a follow-on study.
            decision, reason = "revise pilot", "Review paired decisions and set a follow-on cost gate before proceeding."
        else:
            decision, reason = "proceed", "Independent evidence and paired decision changes support a follow-on study."
        families[family] = {
            "screened": cases["screened"][family], "eligible": len(selected),
            "excluded": sum(case["family"] == family for case in cases["exclusions"]),
            "source_episodes": len(episodes), "source_seeds": len(independent_seeds),
            "probe_episodes": len({case["probe_episode_id"] for case in probes}),
            "probe_usage": family_usage,
            "candidate_outcomes": dict(Counter(candidate_rows[key]["status"] for key in ids if key in candidate_rows)),
            "audited_candidates": len(audits), "audited_counterevidence_events": len(counterevents),
            "scope_errors": sum(audit["scope"] in {"too_broad", "too_narrow"} for audit in audits),
            "probes": len(probes), "paired_counts": {key: counts[key] for key in (
                "complete_triplets", "changed", "unchanged", "harmful_choices_avoided",
                "appropriate_recommendations_lost", "invalid_replies", "error_replies", "unstarted_replies",
                "no_rule_acceptable", "no_rule_harmful", "no_rule_indeterminate",
            )},
            "paired_cases": details, "decision": decision, "reason": reason,
        }
    extraction_calls = candidates["analysis"]["calls"] if candidates else []
    return {
        "kind": "feasibility_report", "corpus_sha256": cases["corpus_sha256"],
        "series_id": cases["series_id"], "split": cases["split"], "origin": cases["origin"],
        "planned_trials": sum(cases["trial_counts"].values()), "trial_counts": cases["trial_counts"],
        "families": families, "usage": {"extraction": _usage(extraction_calls), "probes": _usage(probe_calls or [])},
    }


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Rogue rule feasibility", "",
        f"Evidence origin **{summary['origin']}**, split **{summary['split']}**.", "",
        f"Series `{summary['series_id']}` · corpus SHA-256 `{summary['corpus_sha256']}`.", "",
        f"Planned trials **{summary['planned_trials']}**. "
        + ", ".join(f"{count} {outcome}" for outcome, count in summary["trial_counts"].items()) + ".", "",
        "| Rule family | Screened | Eligible | Source episodes | Audited | Counterevidence events | Complete probe triplets | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, family in summary["families"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {family['screened']} | {family['eligible']} | {family['source_episodes']} | "
            f"{family['audited_candidates']} | {family['audited_counterevidence_events']} | "
            f"{family['paired_counts']['complete_triplets']} | {family['decision']} |"
        )
    lines.append("")
    for name, family in summary["families"].items():
        counts = family["paired_counts"]
        lines.append(
            f"- **{name.replace('_', ' ').capitalize()}** · {family['reason']} "
            f"Correction changed {counts['changed']} decisions and left {counts['unchanged']} unchanged. "
            f"Harmful choices avoided {counts['harmful_choices_avoided']}. "
            f"Appropriate recommendations lost {counts['appropriate_recommendations_lost']}."
        )
    lines.extend(["", "Probe comparisons measure decision sensitivity within a frozen state. Gameplay outcomes require a separate evaluation.", ""])
    with path.open("x", encoding="utf-8", newline="\n") as output:
        output.write("\n".join(lines))
