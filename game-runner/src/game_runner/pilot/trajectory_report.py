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


def _markdown(metrics: dict) -> str:
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
