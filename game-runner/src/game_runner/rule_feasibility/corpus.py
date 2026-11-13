"""Verify a frozen P3 export, then enumerate its screened source evidence."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal
from uuid import UUID

from game_runner.clients.rogue import Observation
from game_runner.pilot.manifest import Manifest

from .artifacts import FAMILIES, FeasibilityError, digest, read_rows

FILES = ("manifest.json", "results.jsonl", "trajectory.jsonl")
FINAL_OUTCOMES = {"ended", "limited", "failed", "interrupted", "unstarted"}
Origin = Literal["natural", "scripted", "injected"]


@dataclass
class Corpus:
    manifest: Manifest
    sha256: str
    trials: dict[str, dict]
    events: dict[int, dict]
    calls: dict[int, dict]
    revisions: dict[int, dict]

    @property
    def series_id(self) -> str:
        return str(self.manifest.series_id)


def load(directory: Path, split: str) -> Corpus:
    contents = {name: (directory / name).read_bytes() for name in FILES}
    checksums = "".join(f"{digest(data)}  {name}\n" for name, data in contents.items()).encode("ascii")
    bundle = digest(checksums)
    if (directory / "SHA256SUMS").read_bytes() != checksums or (
        directory / "bundle.sha256"
    ).read_text(encoding="ascii").strip() != bundle:
        raise FeasibilityError("P3 bundle checksum mismatch")

    metadata = json.loads(contents["manifest.json"])
    if metadata.get("export_schema_version") != 1 or metadata.get("opportunity_schema_version") != 1:
        raise FeasibilityError("unsupported P3 export or screening schema")
    manifest = Manifest.model_validate({
        key: metadata[key] for key in Manifest.model_fields
    })
    if manifest.split != split:
        raise FeasibilityError("P3 bundle split does not match the requested split")
    corpus = Corpus(manifest, bundle, {}, {}, {}, {})
    sessions: set[UUID] = set()
    episodes: set[UUID] = set()
    for index, (planned, result) in enumerate(zip(manifest.trials, read_rows(contents["results.jsonl"]), strict=True)):
        rule = (
            result["series_id"] == corpus.series_id and result["ordinal"] == index
            and result["trial_id"] == planned.trial_id and result["requested_seed"] == planned.seed
            and result["context_policy"] == planned.context_policy and result["outcome"] in FINAL_OUTCOMES
        )
        if not rule or result.get("split", split) != split:
            raise FeasibilityError("result does not match its manifest trial or split")
        for key, seen in (("session_id", sessions), ("episode_id", episodes)):
            if result[key] is not None:
                identifier = UUID(result[key])
                if identifier in seen:
                    raise FeasibilityError(f"duplicated trial {key}")
                seen.add(identifier)
        if (result["session_id"] is None) != (result["outcome"] == "unstarted"):
            raise FeasibilityError("result session does not match trial outcome")
        if result["episode_id"] is not None and result["session_id"] is None:
            raise FeasibilityError("episode is missing its session")
        corpus.trials[planned.trial_id] = result

    stores = {"event": corpus.events, "model_call": corpus.calls, "memory_revision": corpus.revisions}
    for row in read_rows(contents["trajectory.jsonl"]):
        store = stores[row["record_type"]]
        result = corpus.trials.get(row["trial_id"])
        if (row["series_id"] != corpus.series_id or row.get("split", split) != split
                or result is None or result["session_id"] is None or row["session_id"] != result["session_id"]):
            raise FeasibilityError("trajectory row belongs to another trial, session, series, or split")
        if type(row["id"]) is not int or row["id"] <= 0 or row["id"] in store:
            raise FeasibilityError("invalid or duplicate trajectory ID")
        store[row["id"]] = row
    _validate_links(corpus)
    return corpus


def same_trial(store: dict[int, dict], ref: int, owner: dict) -> dict:
    value = store.get(ref)
    if value is None or value["trial_id"] != owner["trial_id"] or value["session_id"] != owner["session_id"]:
        raise FeasibilityError(f"missing or cross-session evidence reference {ref}")
    return value


def _validate_links(corpus: Corpus) -> None:
    events, calls, revisions = corpus.events, corpus.calls, corpus.revisions
    for event in events.values():
        trial = corpus.trials[event["trial_id"]]
        if event["observation"] is not None:
            observation = Observation.model_validate(event["observation"])
            if (str(observation.episode_id) != trial["episode_id"] or observation.api_step != event["api_step"]
                    or observation.seed != trial["observed_reset_seed"]):
                raise FeasibilityError("observation does not match its recorded episode, step, or seed")
        if event["model_call_id"] is not None:
            call = same_trial(calls, event["model_call_id"], event)
            if call["status"] != "valid" or call["purpose"] != "decision":
                raise FeasibilityError("action does not reference a valid decision call")
            if call["observation_event_id"] is None or call["observation_event_id"] >= event["id"]:
                raise FeasibilityError("action input must precede the action event")
    for call in calls.values():
        if call["observation_event_id"] is not None:
            same_trial(events, call["observation_event_id"], call)
        if call["memory_revision_id"] is not None:
            same_trial(revisions, call["memory_revision_id"], call)
    for revision in revisions.values():
        boundary = same_trial(events, revision["through_event_id"], revision)
        call = same_trial(calls, revision["model_call_id"], revision)
        if call["purpose"] != "episode_summary" or call["status"] != "valid" or call["observation_event_id"] != boundary["id"]:
            raise FeasibilityError("revision does not match its generating summary")
        if revision["previous_revision_id"] is not None:
            previous = same_trial(revisions, revision["previous_revision_id"], revision)
            if previous["through_event_id"] >= boundary["id"]:
                raise FeasibilityError("revision coverage does not advance")
        for gap in revision["gap_event_ids"]:
            if same_trial(events, gap, revision)["outcome"] != "uncertain" or gap > boundary["id"]:
                raise FeasibilityError("invalid revision gap reference")


def prepare(corpus: Corpus, origin: Origin) -> dict:
    cases, exclusions = [], []
    screened = dict.fromkeys(FAMILIES, 0)
    for trial_id, trial in corpus.trials.items():
        buckets = trial["screening_opportunities"]
        if set(buckets) != set(FAMILIES):
            raise FeasibilityError("unexpected P3 screening families")
        for family in FAMILIES:
            records = buckets[family]["records"]
            if buckets[family]["count"] != len(records):
                raise FeasibilityError("screening count does not match recorded references")
            seen = set()
            for record in sorted(records, key=lambda row: (row["input_event_id"], row.get("output_event_id", 0), row["model_call_id"])):
                input_event = same_trial(corpus.events, record["input_event_id"], trial)
                call = same_trial(corpus.calls, record["model_call_id"], trial)
                if call["observation_event_id"] != input_event["id"] or call["purpose"] != "decision":
                    raise FeasibilityError("screening call does not reference the input observation")
                output_event = None
                if "output_event_id" in record:
                    output_event = same_trial(corpus.events, record["output_event_id"], trial)
                    if output_event["model_call_id"] != call["id"]:
                        raise FeasibilityError("screening output does not belong to its decision")
                anchor = output_event or input_event
                if anchor["id"] in seen:
                    raise FeasibilityError("duplicate screening opportunity")
                seen.add(anchor["id"])
                screened[family] += 1
                identity = {
                    "case_id": f"{family}:{trial_id}:{anchor['id']}", "family": family,
                    "trial_id": trial_id, "session_id": trial["session_id"], "episode_id": trial["episode_id"],
                    "seed": trial["observed_reset_seed"], **record,
                }
                if input_event["observation"] is None or (output_event and (
                    output_event["outcome"] != "succeeded" or output_event["observation"] is None
                )):
                    exclusions.append({**identity, "reason": "missing_or_uncertain_observation"})
                    continue
                cases.append({
                    **identity, "evidence_event_ids": [input_event["id"]] + ([anchor["id"]] if output_event else []),
                })
    return {
        "kind": "cases", "corpus_sha256": corpus.sha256, "series_id": corpus.series_id,
        "split": corpus.manifest.split, "origin": origin,
        "trial_counts": {outcome: sum(trial["outcome"] == outcome for trial in corpus.trials.values())
                         for outcome in sorted(FINAL_OUTCOMES)},
        "screened": screened, "cases": cases, "exclusions": exclusions,
    }


def checked_cases(corpus: Corpus, artifact: dict) -> dict:
    origin = artifact.get("origin")
    if origin not in {"natural", "scripted", "injected"}:
        raise FeasibilityError("invalid evidence origin")
    expected = prepare(corpus, origin)
    if expected != {key: value for key, value in artifact.items() if key != "schema_version"}:
        raise FeasibilityError("case artifact does not match the verified corpus")
    return expected
