"""Run one stage of the offline rule-feasibility workflow."""

import argparse
import asyncio
from pathlib import Path

from .analysis import AnalysisRun, ModelSettings, provider
from .artifacts import FeasibilityError, read_artifact, read_json, write_artifact
from .candidates import extract
from .corpus import checked_cases, load, prepare
from .context_probes import freeze_context, freeze_correction
from .context_results import context_report, run_context, write_context_markdown
from .probes import freeze, probe, verify_candidates
from .report import checked_results, summarize, write_markdown


def _arguments():
    cli = argparse.ArgumentParser(prog="python -m game_runner.rule_feasibility")
    stages = cli.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "extract", "freeze", "probe", "report", "context-freeze", "context-probe", "context-report"):
        parser = stages.add_parser(name)
        parser.add_argument("--output", required=True, type=Path)
        if name in {"prepare", "extract", "freeze", "report", "context-freeze"}:
            parser.add_argument("--corpus", required=True, type=Path)
            parser.add_argument("--split", choices=("development", "pilot_evaluation"), required=True)
        if name in {"extract", "freeze", "report"}:
            parser.add_argument("--cases", required=True, type=Path)
        if name in {"freeze", "report"}:
            parser.add_argument("--candidates", type=Path, required=name == "freeze")
        if name == "freeze":
            parser.add_argument("--review", type=Path, required=True)
        if name in {"extract", "probe", "context-probe"}:
            parser.add_argument("--max-calls", type=int)
            parser.add_argument("--seconds", type=int)
        if name == "extract":
            parser.add_argument("--selection", type=Path, help="JSON list of case IDs for extraction")
        if name in {"extract", "freeze"}:
            parser.add_argument("--model-config", type=Path, required=name == "freeze")
        if name in {"probe", "report", "context-probe", "context-report"}:
            parser.add_argument("--frozen", type=Path, required=name != "report")
        if name in {"report", "context-report"}:
            parser.add_argument("--results", type=Path, required=name == "context-report")
        if name == "context-freeze":
            source = parser.add_mutually_exclusive_group(required=True)
            source.add_argument("--review", type=Path, help="reviewed context-case JSON")
            source.add_argument("--rule-frozen", type=Path, help="human-reviewed P4 specification")
            parser.add_argument("--repetitions", type=int, default=2, help="repetitions per condition (default 2)")
        if name in {"report", "context-report"}:
            parser.add_argument("--markdown", type=Path)
        if name == "prepare":
            parser.add_argument("--origin", choices=("natural", "scripted", "injected"), required=True)
    return cli, cli.parse_args()


def _run(args, settings):
    if args.max_calls is None or args.seconds is None or args.max_calls <= 0 or args.seconds <= 0:
        raise FeasibilityError("provider use requires positive --max-calls and --seconds")
    return AnalysisRun(provider(settings), settings, args.max_calls, args.seconds)


def _frozen(path: Path, kind: str = "frozen"):
    frozen, sha = read_artifact(path, kind)
    if path.with_name(path.name + ".sha256").read_text(encoding="ascii").strip() != sha:
        raise FeasibilityError("frozen probe specification checksum mismatch")
    return frozen, sha


def _write_frozen(path: Path, specification: dict) -> str:
    sha = write_artifact(path, specification)
    with path.with_name(path.name + ".sha256").open("x", encoding="ascii") as checksum:
        checksum.write(sha + "\n")
    return sha


def main():
    cli, args = _arguments()
    try:
        print(_execute(args))
    except FeasibilityError as error:
        cli.exit(2, f"{error}\n")
    except (ValueError, OSError, KeyError, TypeError) as error:
        cli.exit(2, f"Offline analysis could not read or validate its inputs ({type(error).__name__}).\n")


def _execute(args):
    destinations = [args.output]
    if args.stage in {"freeze", "context-freeze"}:
        destinations.append(args.output.with_name(args.output.name + ".sha256"))
    if args.stage in {"report", "context-report"} and args.markdown:
        destinations.append(args.markdown)
    if len({path.resolve() for path in destinations}) != len(destinations) or any(
        path.exists() for path in destinations
    ):
        raise FeasibilityError("output paths must be distinct and unused")
    if args.stage == "probe":
        frozen, sha = _frozen(args.frozen)
        runner = _run(args, ModelSettings.model_validate(frozen["actor"])) if frozen["probes"] else None
        return write_artifact(args.output, asyncio.run(probe(frozen, sha, runner)))

    if args.stage in {"context-probe", "context-report"}:
        frozen, sha = _frozen(args.frozen, "context_frozen")
        if args.stage == "context-probe":
            runner = _run(args, ModelSettings.model_validate(frozen["actor"])) if frozen["cases"] else None
            return write_artifact(args.output, asyncio.run(run_context(frozen, sha, runner)))
        artifact, _ = read_artifact(args.results, "context_results")
        summary = context_report(frozen, sha, artifact)
        sha = write_artifact(args.output, summary)
        if args.markdown:
            write_context_markdown(args.markdown, summary)
        return sha

    corpus = load(args.corpus, args.split)
    if args.stage == "context-freeze":
        if args.rule_frozen:
            frozen, sha = _frozen(args.rule_frozen)
            specification = freeze_correction(corpus, frozen, sha, args.repetitions)
        else:
            specification = freeze_context(corpus, read_json(args.review), args.repetitions)
        return _write_frozen(args.output, specification)
    if args.stage == "prepare":
        return write_artifact(args.output, prepare(corpus, args.origin))

    cases, cases_sha = read_artifact(args.cases, "cases")
    checked_cases(corpus, cases)
    if args.stage == "extract":
        runner = None
        if args.model_config and cases["cases"]:
            runner = _run(args, ModelSettings.model_validate(read_json(args.model_config)))
        selection = read_json(args.selection) if args.selection else None
        if args.selection and (not isinstance(selection, list) or any(not isinstance(item, str) for item in selection)):
            raise FeasibilityError("selection must be a JSON list of case IDs")
        return write_artifact(args.output, asyncio.run(extract(corpus, cases, cases_sha, runner, selection=selection)))

    candidates, candidates_sha = read_artifact(args.candidates, "candidates") if args.candidates else (None, None)
    if candidates is not None:
        verify_candidates(corpus, cases, cases_sha, candidates)
    if args.stage == "freeze":
        specification = freeze(
            corpus, cases, cases_sha, candidates, candidates_sha, read_json(args.review),
            ModelSettings.model_validate(read_json(args.model_config)),
        )
        return _write_frozen(args.output, specification)

    frozen, frozen_sha = _frozen(args.frozen) if args.frozen else (None, None)
    if frozen and (
        frozen["corpus_sha256"] != corpus.sha256 or frozen["cases_sha256"] != cases_sha
        or frozen["candidates_sha256"] != candidates_sha or frozen["split"] != args.split
        or frozen["origin"] != cases["origin"]
    ):
        raise FeasibilityError("frozen probes require their exact source corpus and artifacts")
    if args.results and not frozen:
        raise FeasibilityError("--results requires --frozen")
    artifact = read_artifact(args.results, "probe_results")[0] if args.results else None
    results = checked_results(frozen, frozen_sha, artifact) if artifact else None
    summary = summarize(cases, candidates, frozen, results, artifact["analysis"]["calls"] if artifact else None)
    sha = write_artifact(args.output, summary)
    if args.markdown:
        write_markdown(args.markdown, summary)
    return sha


if __name__ == "__main__":
    main()
