"""Execute and verify paired context probes without a game client."""

from collections import Counter, defaultdict
from pathlib import Path

from game_runner.decision import parse_decision

from .analysis import AnalysisRun, ModelSettings
from .artifacts import FeasibilityError
from .context_probes import CONTEXT_VERSION, messages_for, prompt_rows, score_action
from .report import _usage


def _label(entry: dict) -> str:
    return f"{entry['case_id']}:{entry['repetition']}:{entry['condition']}"


def _grade(call: dict, case: dict) -> dict:
    if call["status"] == "error":
        return {"status": "error"}
    if call["status"] != "returned":
        raise FeasibilityError("invalid context probe call status")
    try:
        decision = parse_decision(call["raw_response"], case["mode"])
    except ValueError:
        return {"status": "invalid"}
    action = decision.action.model_dump(mode="json", exclude_none=True)
    return {
        "status": "valid", "action": action, "rationale": decision.rationale,
        "score": score_action(action, case["mode"], case["rubric"]),
    }


def _identity(spec: dict, frozen_digest: str) -> dict:
    if spec["prompt_version"] != CONTEXT_VERSION:
        raise FeasibilityError("unsupported context probe version")
    return {
        "frozen_sha256": frozen_digest, "corpus_sha256": spec["corpus_sha256"],
        "series_id": spec["series_id"], "split": spec["split"], "origin": spec["origin"],
    }


async def run_context(spec: dict, frozen_digest: str, run: AnalysisRun | None) -> dict:
    identity = _identity(spec, frozen_digest)
    if run and ModelSettings.model_validate(spec["actor"]) != run.settings:
        raise FeasibilityError("context probe actor does not match frozen settings")
    cases = {case["case_id"]: case for case in spec["cases"]}
    results = []
    for block in spec["blocks"]:
        case = cases[block["case_id"]]
        can_start = run is not None and run.available(len(block["conditions"]))
        for condition in block["conditions"]:
            entry = {"case_id": block["case_id"], "repetition": block["repetition"], "condition": condition}
            if not can_start or not run.available():
                results.append({**entry, "status": "unstarted"})
                continue
            call, _ = await run.request(
                _label(entry), CONTEXT_VERSION, messages_for(prompt_rows(case, condition)),
            )
            results.append({**entry, "call_id": call["id"], **_grade(call, case)})
    return {
        "kind": "context_results", **identity, "results": results,
        "analysis": run.provenance() if run else {"calls": []},
    }


def _checked(spec: dict, frozen_digest: str, artifact: dict):
    identity = _identity(spec, frozen_digest)
    if any(artifact.get(key) != value for key, value in identity.items()):
        raise FeasibilityError("context results do not belong to this frozen specification")
    cases = {case["case_id"]: case for case in spec["cases"]}
    expected = [
        (block["case_id"], block["repetition"], condition)
        for block in spec["blocks"] for condition in block["conditions"]
    ]
    rows = artifact["results"]
    if expected != [(row["case_id"], row["repetition"], row["condition"]) for row in rows]:
        raise FeasibilityError("context results do not match the frozen schedule")
    calls = artifact["analysis"]["calls"]
    if [row.get("call_id") for row in rows if row["status"] != "unstarted"] != [call["id"] for call in calls]:
        raise FeasibilityError("context results do not match the recorded calls")
    if calls and ModelSettings.model_validate(artifact["analysis"]["model"]) != ModelSettings.model_validate(spec["actor"]):
        raise FeasibilityError("recorded calls do not match the frozen actor settings")
    called = {call["id"]: call for call in calls}
    for row in rows:
        case = cases[row["case_id"]]
        expected_row = {key: row[key] for key in ("case_id", "repetition", "condition")}
        if row["status"] == "unstarted":
            expected_row["status"] = "unstarted"
        else:
            call = called[row["call_id"]]
            if (
                call["label"] != _label(row) or call["prompt_version"] != CONTEXT_VERSION
                or call["input_messages"] != prompt_rows(case, row["condition"])
            ):
                raise FeasibilityError("recorded request differs from its frozen treatment")
            expected_row.update(call_id=call["id"], **_grade(call, case))
        if row != expected_row:
            raise FeasibilityError("recorded result differs from the frozen rubric or response")
    return rows, called


def _outcome(row: dict) -> str:
    return row["score"] if row["status"] == "valid" else row["status"]


def _contrast(cases: list[dict], trials: dict, baseline: str, treatment: str) -> dict:
    gained = lost = control_delta = 0
    by_seed = Counter()
    transitions = Counter()
    planned = 0
    complete = True
    for case in cases:
        for outcomes in trials[case["case_id"]].values():
            before, after = outcomes[baseline], outcomes[treatment]
            transitions[(before, after)] += 1
            planned += 1
            complete &= "unstarted" not in (before, after)
            change = int(after == "acceptable") - int(before == "acceptable")
            by_seed[case["seed"]] += change
            gained += change > 0
            lost += change < 0
            if case["control"]:
                control_delta += int(after == "harmful") - int(before == "harmful")
    return {
        "baseline": baseline, "treatment": treatment, "planned_pairs": planned, "complete": complete,
        "acceptable_delta_pp": round(100 * (gained - lost) / planned, 2),
        "acceptable_delta_count": gained - lost, "control_harmful_delta": control_delta,
        "controls": sum(case["control"] for case in cases),
        "improved_seeds": sorted(seed for seed, delta in by_seed.items() if delta > 0),
        "harmful_to_acceptable": transitions[("harmful", "acceptable")],
        "acceptable_to_harmful": transitions[("acceptable", "harmful")],
        "transitions": [
            {"before": before, "after": after, "count": count}
            for (before, after), count in sorted(transitions.items())
        ],
    }


def context_report(spec: dict, frozen_digest: str, artifact: dict) -> dict:
    results, calls = _checked(spec, frozen_digest, artifact)
    outcomes = defaultdict(lambda: defaultdict(dict))
    calls_by_case = defaultdict(list)
    for row in results:
        outcomes[row["case_id"]][row["repetition"]][row["condition"]] = _outcome(row)
        if "call_id" in row:
            calls_by_case[row["case_id"]].append(calls[row["call_id"]])

    groups = {}
    eligible = []
    priority = ("instruction", "neighbors", "source_fact")
    for kind in ("grounding", "history", "correction"):
        subset = [case for case in spec["cases"] if case["kind"] == kind]
        if not subset:
            continue
        baseline, *treatments = subset[0]["conditions"]
        comparisons = [(baseline, treatment, subset) for treatment in treatments]
        if kind == "correction":
            comparisons.extend(
                ("original", "corrected", scoped)
                for scope in ("matching", "out_of_scope")
                if (scoped := [case for case in subset if case["applicability"] == scope])
            )
        contrasts = []
        for before, after, scoped in comparisons:
            contrast = _contrast(scoped, outcomes, before, after)
            contrast["scope"] = scoped[0]["applicability"] if scoped is not subset else "all"
            contrasts.append(contrast)
            qualifies = (
                spec["origin"] == "natural" and after in priority and contrast["complete"]
                and contrast["acceptable_delta_pp"] >= 10 and len(contrast["improved_seeds"]) >= 2
                and contrast["controls"] > 0 and contrast["control_harmful_delta"] <= 0
            )
            contrast["qualifies_for_online_pilot"] = qualifies
            if qualifies:
                eligible.append((contrast["acceptable_delta_pp"], -priority.index(after), after))
        groups[kind] = {
            "cases": len(subset), "seeds": sorted({case["seed"] for case in subset}),
            "counts_by_condition": {
                condition: dict(Counter(
                    repetition[condition] for case in subset for repetition in outcomes[case["case_id"]].values()
                ))
                for condition in subset[0]["conditions"]
            },
            "contrasts": contrasts,
            "usage": _usage([call for case in subset for call in calls_by_case[case["case_id"]]]),
        }
    return {
        "kind": "context_report", **_identity(spec, frozen_digest),
        "repetitions": spec["repetitions"], "planned_calls": len(results),
        "unstarted_calls": sum(row["status"] == "unstarted" for row in results),
        "groups": groups, "usage": _usage(list(calls.values())),
        "recommended_intervention": max(eligible)[2] if eligible else None,
        "cases": [
            {key: case[key] for key in ("case_id", "kind", "seed", "decision_call_id", "pre_event_id", "control")}
            | {key: case[key] for key in ("applicability", "relation") if key in case}
            | {"outcomes": {str(key): value for key, value in outcomes[case["case_id"]].items()}}
            for case in spec["cases"]
        ],
    }


def write_context_markdown(path: Path, summary: dict) -> None:
    selected = summary["recommended_intervention"] or "None qualified"
    lines = [
        "# Rogue offline context probes", "",
        f"Series `{summary['series_id']}` · split `{summary['split']}` · origin `{summary['origin']}`", "",
        f"**{summary['planned_calls']} planned calls**, {summary['unstarted_calls']} unstarted.",
        f"**Online pilot candidate** · {selected}.", "",
        "| Probe group | Scope | Contrast | Planned pairs | Δ acceptable (pp) | Harmful → acceptable | Acceptable → harmful |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for kind, group in summary["groups"].items():
        for row in group["contrasts"]:
            lines.append(
                f"| {kind} | {row['scope']} | {row['baseline']} → {row['treatment']} | {row['planned_pairs']} | "
                f"{row['acceptable_delta_pp']:+.2f} | {row['harmful_to_acceptable']} | {row['acceptable_to_harmful']} |"
            )
    lines.extend(["", "The denominator includes invalid, uncertain, error and unstarted responses. "
                  "Repeated responses share a case and seed. See the JSON report for outcomes and provenance.", ""])
    with path.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines))
