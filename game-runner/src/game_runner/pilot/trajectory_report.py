"""Write local, reproducible artifacts from a frozen pilot export."""

import json
from pathlib import Path
import tempfile

from game_runner.rule_feasibility.artifacts import FeasibilityError, canonical
from game_runner.rule_feasibility.corpus import load

from .trajectory import analyze


def _table(headers, rows) -> str:
    def line(values):
        fields = [
            "—" if value is None else str(value).replace("|", "&#124;").replace("\n", " ")
            for value in values
        ]
        return "| " + " | ".join(fields) + " |"

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def _progress_row(trial: dict) -> tuple:
    coverage = trial["coverage"]
    positions = [point["positions"] for point in coverage["checkpoints"]]
    return (
        trial["trial_id"], trial["requested_seed"], trial["context_policy"], trial["outcome"], trial["reason"],
        trial["action_dispatches"], trial["last_observed_api_step"], trial["max_observed_depth"], *positions,
        coverage["actions_since_last_new_position"], coverage["longest_no_new_position_move_run"]["actions"],
        coverage["immediate_returns"],
    )


def _opportunity_row(trial: dict) -> tuple:
    movement, stairs = trial["movement"], trial["stairs"]
    return (
        trial["trial_id"], movement["comparable_moves"], movement["moves_without_displacement"],
        movement["same_position_combat"], movement["repeated_wall_moves"], movement["repeat_wall_opportunities"],
        stairs["descent_attempts"], stairs["descents_attempted_on_stairs"], stairs["on_stairs_opportunities"],
        stairs["descents_attempted_adjacent"], stairs["adjacent_stairs_opportunities"],
    )


def _usage_rows(trials: list[dict]):
    for trial in trials:
        for purpose, usage in trial["model_usage"].items():
            counts = usage["status_counts"]
            yield (
                trial["trial_id"], purpose, sum(counts.values()), counts.get("invalid", 0), counts.get("error", 0),
                usage.get("provider_total_tokens"), usage.get("calls_missing_provider_usage"), usage.get("latency_ms"),
            )


def _variant_table(pairs: list[dict]) -> str:
    return _table(
        ("Seed", "Baseline trial", "Comparison trial", "Complete", "Δ positions", "Δ longest move streak"),
        (
            (pair["seed"], pair["baseline_trial_id"], pair["comparison_trial_id"],
             "yes" if pair["complete"] else "no", pair["positions_delta"],
             pair["longest_no_new_position_move_run_delta"])
            for pair in pairs
        ),
    )


def _neighbors_sections(comparison: dict, trials: list[dict]) -> list[str]:
    complete = comparison["complete_primary_pairs"]
    planned = len(comparison["primary_pairs"])
    sections = [
        "## Visible-neighbor pilot",
        f"Action horizon {comparison['action_horizon']} · {complete}/{planned} complete primary pairs.",
        f"Complete pairs with improved coverage · {comparison['pairs_with_improved_coverage']}. "
        f"Median longest-streak difference {comparison['median_primary_streak_delta'] if complete else 'unavailable'}. "
        f"Deaths in primary trials · baseline {comparison['baseline_deaths']}, "
        f"neighbors {comparison['neighbors_deaths']}.",
        "The first baseline in manifest order is the reference. Differences are comparison minus baseline. "
        "Pairs with missing observations, uncertain dispatches, or technical stops remain in the table.",
        _variant_table(comparison["primary_pairs"]),
    ]
    if comparison["baseline_repeats"]:
        sections.extend(["### Baseline repeats", _variant_table(comparison["baseline_repeats"])])
    sections.extend([
        "### Prompt exposure",
        _table(
            ("Trial", "Variant", "Prompts with the neighbor block", "Recorded decision prompts"),
            (
                (trial["trial_id"], trial.get("context_variant", "baseline"),
                 exposure["with_neighbors"], exposure["recorded_decision_prompts"])
                for trial in trials if (exposure := trial.get("neighbors_prompt_exposure")) is not None
            ),
        ),
    ])
    return sections


def _markdown(metrics: dict) -> str:
    if metrics["schema_version"] == 2:
        return _evaluation_markdown(metrics)
    trials = metrics["trials"]
    sections = [
        "# Rogue pilot trajectory analysis",
        f"Series `{metrics['series_id']}` · split `{metrics['split']}` · export SHA-256 `{metrics['bundle_sha256']}`",
        f"**{len(trials)} planned trials** and **{metrics['action_dispatches']} recorded action dispatches**.",
        "Trial outcomes · " + ", ".join(f"{outcome} {count}" for outcome, count in sorted(metrics["trial_counts"].items())),
        "## Observed progress",
        _table(
            (
                "Trial", "Seed", "Policy", "Outcome", "Reason", "Dispatches", "Last API step", "Max depth",
                "Positions @25", "@50", "@100", "@200", "Actions since last new position",
                "Longest move streak without a new position", "A→B→A",
            ),
            map(_progress_row, trials),
        ),
        "Position counts include the reset when visible. Each checkpoint uses observations through the earlier "
        "of that checkpoint and the last recorded API step. Position counts use confirmed map observations only. "
        "The longest streak counts comparable movement actions and breaks at uncertain outcomes, menus, and map changes. "
        "A→B→A requires two consecutive displacements on the same map.",
        "## Decision opportunities",
        _table(
            (
                "Trial", "Comparable moves", "Same position", "Same-position combat", "Repeated wall moves",
                "Wall-repeat opportunities", "Descent attempts", "On-stair attempts", "On-stair opportunities",
                "Adjacent attempts", "Adjacent opportunities",
            ),
            map(_opportunity_row, trials),
        ),
        "Opportunities are counted before evaluating the chosen action. Stair sightings are scoped to the observed "
        "map instance. A wall-repeat opportunity requires a preceding quiet wall-facing move and a still-visible wall. "
        "The evidence and the raw action and rationale are in `decision_ledger.jsonl`, keyed by input event, model call, "
        "and output event.",
        "## Model usage",
        _table(
            ("Trial", "Purpose", "Calls", "Invalid", "Errors", "Provider tokens", "Calls missing usage", "Latency (ms)"),
            _usage_rows(trials),
        ),
    ]
    if pairs := metrics["matched_seed_pairs"]:
        sections.extend([
            "## Matched seeds",
            "Differences are working memory minus history only, each measured over its observed trajectory.",
            _table(
                ("Seed", "History-only trial", "Working-memory trial", "Δ positions", "Δ longest move streak"),
                (
                    (
                        pair["seed"], pair["history_only_trial_id"], pair["working_memory_trial_id"],
                        pair["positions_delta"], pair["longest_no_new_position_move_run_delta"],
                    )
                    for pair in pairs
                ),
            ),
        ])
    if comparison := metrics.get("neighbors_comparison"):
        sections.extend(_neighbors_sections(comparison, trials))
    return "\n\n".join(sections) + "\n"


def _evaluation_markdown(metrics: dict) -> str:
    trials = metrics["trials"]
    sections = [
        "# Rogue decision-loop evaluation",
        f"Series `{metrics['series_id']}` · split `{metrics['split']}` · export SHA-256 `{metrics['bundle_sha256']}`",
        "## Trial outcomes",
        _table(
            ("Trial", "Condition", "Seed", "Scenario", "Outcome", "Reason", "Game", "Setup", "Measured",
             "Confirmed prefix", "Missing frames", "Uncertain", "Max depth", "Wall time (ms)"),
            ((t["trial_id"], t["condition"], t["requested_seed"], t["scenario_id"], t["outcome"], t["reason"],
              t["game_status"], t["setup_dispatches"], t["action_dispatches"], t["confirmed_prefix"],
              t["missing_observations"], t["uncertain_dispatches"], t["max_observed_depth"], t["wall_time_ms"])
             for t in trials),
        ),
        "## Planned settings and delivered actor context",
        _table(
            ("Trial", "Guide", "Prompt version", "State", "History", "Goals", "Actor calls", "State blocks",
             "Setup blocks", "Goal blocks", "Delivered history format", "Eligible history", "Delivered history", "Trimmed calls"),
            ((t["trial_id"], t["prompt_profile"], t["prompt_version"], t["structured_state"], t["history_format"],
              t["goal_management"], c["calls"], c["working_state_calls"], c["setup_evidence_calls"], c["goal_context_calls"],
              ", ".join(sorted(c["history_formats"])) or None, c["eligible_interactions"], c["delivered_interactions"],
              c["trimmed_calls"]) for t in trials for c in (t["actor_context"],)),
        ),
        "## Movement and goal outcomes",
        _table(
            ("Trial", "Visits", "Longest move streak", "A→B→A", "Quiet failures", "Level changes", "Reviews",
             "Completed goals", "Suspended goals", "Review reasons"),
            ((t["trial_id"], coverage.get("positions"), coverage.get("longest_no_new_position_move_run", {}).get("actions"),
              coverage.get("immediate_returns"), t["quiet_failed_moves"], len(t["level_changes"]),
              sum(t["goals"]["review_statuses"].values()), len(t["goals"]["completed_goal_ids"]),
              len(t["goals"]["suspended_goal_ids"]),
              ", ".join(f"{reason} ({count})" for reason, count in sorted(t["goals"]["review_reasons"].items())) or None)
             for t in trials for coverage in (t["coverage"] or {},)),
        ),
        "## Model usage",
        _table(
            ("Trial", "Purpose", "Calls", "Invalid", "Errors", "Provider tokens", "Calls missing usage", "Latency (ms)"),
            _usage_rows(trials),
        ),
    ]
    diagnostics = [(trial, trial["diagnostic_outcome"]) for trial in trials if trial["diagnostic_outcome"] is not None]
    if diagnostics:
        sections.extend([
            "## Diagnostic evidence",
            _table(
                ("Trial", "Verified anchor", "Objective", "Observed outcome", "Evidence event", "Decision support"),
                ((trial["trial_id"], trial["anchor_event_id"], outcome["objective"], outcome["objective_status"],
                  outcome["evidence_event_id"], outcome["decision_support"]) for trial, outcome in diagnostics),
            ),
            "Objective results come from recorded effects. Decision support requires review against the frozen rubric.",
        ])
    sections.extend([
        "## Matched comparisons",
        f"The planned horizon is {metrics['comparison_horizon']} measured actions. Each difference uses the common "
        "confirmed prefix. Shorter prefixes are reported separately from full-horizon medians.",
    ])
    for contrast in metrics["comparisons"]:
        median_delta = contrast["median_full_horizon_delta"]
        sections.extend([
            f"### {contrast['reference']} versus {contrast['treatment']}",
            f"Full-horizon pairs {contrast['full_horizon_pairs']} of {len(contrast['pairs'])}."
            + (f" Median Δ visits {median_delta['positions']:g} and Δ longest move streak "
               f"{median_delta['longest_move_streak']:g}." if median_delta is not None else ""),
            _table(
                ("Seed", "Scenario", "Reference", "Treatment", "Shared horizon", "Full horizon", "Δ positions", "Δ longest streak"),
                ((pair["seed"], pair["scenario_id"], pair["reference_trial_id"], pair["treatment_trial_id"],
                  pair["shared_horizon"], "yes" if pair["full_horizon"] else "no",
                  (pair["delta"] or {}).get("positions"), (pair["delta"] or {}).get("longest_move_streak"))
                 for pair in contrast["pairs"]),
            ),
        ])
    sections.append("The decision ledger links measured actions, delivered actor context, and goal reviews to their source events.")
    return "\n\n".join(sections) + "\n"


def write_report(source: Path, split: str, output: Path) -> Path:
    """Verify the source bundle and publish a complete report to a fresh directory."""
    source = source.resolve()
    output = output.resolve()
    if output.is_relative_to(source):
        raise FeasibilityError("analysis output must be outside the source bundle")
    if output.exists():
        raise FeasibilityError("analysis output directory already exists")

    metrics, ledger = analyze(load(source, split))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".trajectory-analysis-", dir=output.parent) as scratch:
        directory = Path(scratch) / "report"
        directory.mkdir()
        with (directory / "decision_ledger.jsonl").open("wb") as stream:
            for row in ledger:
                stream.write(canonical(row))
        (directory / "run_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (directory / "report.md").write_text(_markdown(metrics), encoding="utf-8")
        directory.rename(output)
    return output
