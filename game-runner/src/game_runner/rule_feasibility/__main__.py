"""Run one stage of the offline rule-feasibility workflow."""

import argparse
import asyncio
from pathlib import Path

from .analysis import AnalysisRun, ModelSettings, provider
from .artifacts import FeasibilityError, read_artifact, read_json, write_artifact
from .candidates import extract
from .corpus import checked_cases, load, prepare
from .probes import freeze, probe, verify_candidates
from .report import checked_results, summarize, write_markdown


def _arguments():
    cli = argparse.ArgumentParser(prog="python -m game_runner.rule_feasibility")
    stages = cli.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "extract", "freeze", "probe", "report"):
        parser = stages.add_parser(name)
        parser.add_argument("--output", required=True, type=Path)
        if name in {"prepare", "extract", "freeze", "report"}:
            parser.add_argument("--corpus", required=True, type=Path)
            parser.add_argument("--split", choices=("development", "pilot_evaluation"), required=True)
        if name in {"extract", "freeze", "report"}:
            parser.add_argument("--cases", required=True, type=Path)
        if name in {"freeze", "report"}:
            parser.add_argument("--candidates", type=Path, required=name == "freeze")
        if name == "freeze":
            parser.add_argument("--review", type=Path, required=True)
        if name in {"extract", "probe"}:
            parser.add_argument("--max-calls", type=int)
            parser.add_argument("--seconds", type=int)
        if name in {"extract", "freeze"}:
            parser.add_argument("--model-config", type=Path, required=name == "freeze")
        if name in {"probe", "report"}:
            parser.add_argument("--frozen", type=Path, required=name == "probe")
        if name == "report":
            parser.add_argument("--results", type=Path)
            parser.add_argument("--markdown", type=Path)
        if name == "prepare":
            parser.add_argument("--origin", choices=("natural", "scripted", "injected"), required=True)
    return cli, cli.parse_args()


def _run(args, settings):
    if args.max_calls is None or args.seconds is None or args.max_calls <= 0 or args.seconds <= 0:
        raise FeasibilityError("provider use requires positive --max-calls and --seconds")
    return AnalysisRun(provider(settings), settings, args.max_calls, args.seconds)


def _frozen(path: Path):
    frozen, sha = read_artifact(path, "frozen")
    if path.with_name(path.name + ".sha256").read_text(encoding="ascii").strip() != sha:
        raise FeasibilityError("frozen probe specification checksum mismatch")
    return frozen, sha


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
    if args.stage == "freeze":
        destinations.append(args.output.with_name(args.output.name + ".sha256"))
    if args.stage == "report" and args.markdown:
        destinations.append(args.markdown)
    if len({path.resolve() for path in destinations}) != len(destinations) or any(
        path.exists() for path in destinations
    ):
        raise FeasibilityError("output paths must be distinct and unused")
    if args.stage == "probe":
        frozen, sha = _frozen(args.frozen)
        runner = _run(args, ModelSettings.model_validate(frozen["actor"])) if frozen["probes"] else None
        return write_artifact(args.output, asyncio.run(probe(frozen, sha, runner)))

    corpus = load(args.corpus, args.split)
    if args.stage == "prepare":
        return write_artifact(args.output, prepare(corpus, args.origin))

    cases, cases_sha = read_artifact(args.cases, "cases")
    checked_cases(corpus, cases)
    if args.stage == "extract":
        runner = None
        if args.model_config and cases["cases"]:
            runner = _run(args, ModelSettings.model_validate(read_json(args.model_config)))
        return write_artifact(args.output, asyncio.run(extract(corpus, cases, cases_sha, runner)))

    candidates, candidates_sha = read_artifact(args.candidates, "candidates") if args.candidates else (None, None)
    if candidates is not None:
        verify_candidates(corpus, cases, cases_sha, candidates)
    if args.stage == "freeze":
        specification = freeze(
            corpus, cases, cases_sha, candidates, candidates_sha, read_json(args.review),
            ModelSettings.model_validate(read_json(args.model_config)),
        )
        sha = write_artifact(args.output, specification)
        with args.output.with_name(args.output.name + ".sha256").open("x", encoding="ascii") as checksum:
            checksum.write(sha + "\n")
        return sha

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
