"""Command line for running and exporting a pilot series."""

import argparse
import asyncio
import os
from pathlib import Path
import re
import subprocess
from uuid import UUID

from pydantic import ValidationError

from game_runner.config import Settings
from game_runner.migrate import main as migrate
from game_runner.pilot.manifest import Manifest
from game_runner.pilot.runner import run_series
from game_runner.pilot.store import PilotError
from game_runner.rule_feasibility.artifacts import FeasibilityError


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m game_runner.pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--rogue-binary", type=Path, default=os.getenv("ROGUE_ENV_ROGUE_BINARY"))
    run.add_argument("--runner-revision", default=os.getenv("PILOT_RUNNER_REVISION"))
    export = commands.add_parser("export")
    export.add_argument("--series-id", required=True, type=UUID)
    export.add_argument("--output", required=True, type=Path)
    analyze = commands.add_parser("analyze", help="analyze a frozen pilot export without running Rogue")
    analyze.add_argument("--corpus", required=True, type=Path, help="verified pilot export directory")
    analyze.add_argument("--split", required=True, choices=("development", "pilot_evaluation"))
    analyze.add_argument("--output", required=True, type=Path, help="fresh local directory for the report")
    args = parser.parse_args()
    try:
        if args.command == "export":
            from game_runner.pilot.export import export_series

            print(export_series(args.series_id, args.output))
            return
        if args.command == "analyze":
            from game_runner.pilot.trajectory_report import write_report

            print(write_report(args.corpus, args.split, args.output))
            return
        manifest = Manifest.read(args.manifest)
        binary = args.rogue_binary
        if binary is None or not binary.is_file() or not os.access(binary, os.X_OK):
            raise PilotError("provide an executable pinned build using --rogue-binary")
        revision = args.runner_revision or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise PilotError("runner revision must be a full Git SHA")
        # The manifest supplies inference settings, including the model name and prompt profile.
        os.environ["MODEL_NAME"] = manifest.actor.model_name
        runtime = Settings.from_env(prompt_profile="legacy", structured_state="off", history_format="replay")
        migrate()
        reason = asyncio.run(run_series(manifest, runtime, binary=binary.resolve(), revision=revision))
        print(f"series {manifest.series_id} ended with {reason}")
    except ValidationError as exc:
        errors = ", ".join(".".join(map(str, item["loc"])) or "manifest" for item in exc.errors())
        parser.exit(2, f"Invalid manifest fields: {errors}\n")
    except (PilotError, FeasibilityError) as exc:
        parser.exit(1, f"{exc}\n")
    except Exception as exc:
        parser.exit(1, f"Pilot command failed ({type(exc).__name__}).\n")


if __name__ == "__main__":
    main()
