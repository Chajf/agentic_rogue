"""Generate a pilot schedule, run it, and export the persisted evidence."""

import argparse
import os
from pathlib import Path
from random import SystemRandom
import subprocess
import sys
from uuid import uuid4

from game_runner.pilot.manifest import Actor, Limits, Manifest, Trial

POLICIES = ("history_only", "working_memory")
SEED_COUNT = 2**32


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_manifest(args: argparse.Namespace) -> Manifest:
    policies = POLICIES if args.policy == "both" else (args.policy,)
    if args.seeds is not None:
        seeds = args.seeds
        if any(not 0 <= seed < SEED_COUNT for seed in seeds):
            raise ValueError("seeds must be between 0 and 4294967295")
        if len(seeds) != len(set(seeds)):
            raise ValueError("seeds must be unique")
    else:
        runs = args.runs if args.runs is not None else len(policies)
        seed_count, remainder = divmod(runs, len(policies))
        if remainder:
            raise ValueError("--runs must be even with --policy both")
        if seed_count > SEED_COUNT:
            raise ValueError("too many runs to assign unique seeds")
        seeds = SystemRandom().sample(range(SEED_COUNT), seed_count)
    trials = tuple(
        Trial(trial_id=f"seed-{seed}-{policy}", seed=seed, context_policy=policy)
        for seed in seeds for policy in policies
    )
    trial_limits = Limits(
        actions=args.max_turns,
        model_calls=args.max_model_calls or 4 * args.max_turns,
        seconds=args.trial_seconds,
    )
    series_limits = Limits(
        actions=trial_limits.actions * len(trials),
        model_calls=trial_limits.model_calls * len(trials),
        seconds=args.series_seconds or trial_limits.seconds * len(trials),
    )
    return Manifest(
        schema_version=1, series_id=uuid4(), split=args.split,
        actor=Actor(
            model_name=args.model, temperature=args.temperature,
            reasoning_enabled={"default": None, "on": True, "off": False}[args.reasoning],
            output_tokens=args.output_tokens, summary_output_tokens=args.summary_output_tokens,
            context_tokens=args.context_tokens,
            summary_context_tokens=args.summary_context_tokens or args.context_tokens,
            history_interactions=args.history_interactions,
        ),
        trial_limits=trial_limits, series_limits=series_limits, trials=trials,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--runs", type=positive_int, help="total games; with both policies, use an even number")
    schedule.add_argument("--seeds", type=int, nargs="+", help="explicit seeds; each is run under every selected policy")
    parser.add_argument("--policy", choices=("both", *POLICIES), default="both", help="context policies (default: both)")
    parser.add_argument("--split", choices=("development", "pilot_evaluation"), default="development")
    parser.add_argument("--max-turns", type=positive_int, default=20, help="maximum action dispatches per game (default: 20)")
    parser.add_argument("--max-model-calls", type=positive_int, help="per-game cap including retries and summaries (default: 4 × max turns)")
    parser.add_argument("--trial-seconds", type=positive_int, default=180, help="per-game time limit (default: 180)")
    parser.add_argument("--series-seconds", type=positive_int, help="whole-series time limit (default: sum of per-game limits)")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"), help="model ID (default: MODEL_NAME)")
    parser.add_argument("--reasoning", choices=("default", "on", "off"), default="default")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--output-tokens", type=positive_int, default=512)
    parser.add_argument("--summary-output-tokens", type=positive_int, default=1024)
    parser.add_argument("--context-tokens", type=positive_int, default=16384)
    parser.add_argument("--summary-context-tokens", type=positive_int, help="defaults to --context-tokens")
    parser.add_argument("--history-interactions", type=int, default=4)
    parser.add_argument("--rogue-binary", type=Path, default=os.getenv("ROGUE_ENV_ROGUE_BINARY"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments"), help="parent of the series directories")
    parser.add_argument("--plan-only", action="store_true", help="write the manifest without running games")
    args = parser.parse_args()
    if not args.model:
        parser.error("set MODEL_NAME or supply --model")
    if not args.plan_only and (
        args.rogue_binary is None or not args.rogue_binary.is_file() or not os.access(args.rogue_binary, os.X_OK)
    ):
        parser.error("supply an executable pinned Rogue build with --rogue-binary")
    if args.split == "pilot_evaluation" and args.policy != "both":
        parser.error("pilot_evaluation requires --policy both")
    try:
        manifest = build_manifest(args)
    except ValueError as exc:
        parser.error(str(exc))

    directory = (args.output_dir / str(manifest.series_id)).resolve()
    plan = directory / "plan.json"
    bundle = directory / "export"
    try:
        directory.mkdir(parents=True)
        plan.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"Could not write the experiment plan. {exc}\n")
    print(f"Planned {len(manifest.trials)} games with up to {manifest.trial_limits.actions} actions each.", flush=True)
    print(f"Manifest written to {plan}", flush=True)
    if args.plan_only:
        return 0

    pilot = [sys.executable, "-m", "game_runner.pilot"]
    try:
        run = subprocess.run([
            *pilot, "run", "--manifest", str(plan), "--rogue-binary", str(args.rogue_binary), "--progress",
        ])
        if run.returncode:
            print(f"Pilot run failed. Use series ID {manifest.series_id} to inspect or export any saved data.", file=sys.stderr)
            return run.returncode if run.returncode > 0 else 128 - run.returncode
        export = subprocess.run([
            *pilot, "export", "--series-id", str(manifest.series_id), "--output", str(bundle),
        ])
    except KeyboardInterrupt:
        print(f"Interrupted series {manifest.series_id}. Saved data can be exported with the P3 CLI.", file=sys.stderr)
        return 130
    if export.returncode:
        print(f"Export failed for series {manifest.series_id}. Retry the export with the P3 CLI.", file=sys.stderr)
        return export.returncode if export.returncode > 0 else 128 - export.returncode
    print(f"Export written to {bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
