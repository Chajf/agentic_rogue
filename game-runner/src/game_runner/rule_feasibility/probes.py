"""Freeze human reviews and perform paired, action-free decisions."""

from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import Field

from game_runner.clients.rogue import ActionRequest, KeyActionRequest, Observation
from game_runner.context.messages import build_messages
from game_runner.context.transitions import DecisionTransition
from game_runner.decision import Decision, parse_decision, validate_mode
from game_runner.pilot.manifest import ManifestModel
from game_runner.prompts.system import PROMPT_VERSION

from .analysis import AnalysisRun, ModelSettings
from .artifacts import FeasibilityError, canonical, digest
from .candidates import EventRef, Text, validate_candidate
from .corpus import Corpus

PROBE_VERSION = f"rule-probe-v1/{PROMPT_VERSION}"
CONDITIONS = ("no_rule", "original", "corrected")
Action = Annotated[ActionRequest | KeyActionRequest, Field(discriminator="type")]


class Audit(ManifestModel):
    candidate_id: str
    factual_correctness: Literal["correct", "incorrect", "unknown"]
    evidential_support: Literal["supported", "refuted", "insufficient", "conflicting"]
    scope: Literal["appropriate", "too_broad", "too_narrow", "unknown"]
    applicability: Literal["applicable", "conditional", "inapplicable", "unknown"]
    utility: Literal["useful", "harmful", "mixed", "unknown"]
    rationale: Text
    original: Text
    corrected: Text
    corroborating_event_ids: list[EventRef] = Field(default_factory=list)
    counterevidence_event_ids: list[EventRef] = Field(default_factory=list)


class ProbeReview(ManifestModel):
    candidate_id: str
    pre_event_id: EventRef
    applicability: Literal["matching", "out_of_scope"]
    acceptable: list[Action]
    harmful: list[Action]
    indeterminate: list[Action] = Field(default_factory=list)
    rationale: Text


class Review(ManifestModel):
    schema_version: Literal[1]
    corpus_sha256: str
    candidates_sha256: str
    reviewer: Text
    approved: Literal[True]
    audits: list[Audit]
    probes: list[ProbeReview]


def _action_key(action: dict) -> bytes:
    return canonical(action)


def _before(corpus: Corpus, event_id: int):
    event = corpus.events.get(event_id)
    if event is None or event["observation"] is None or event["outcome"] not in {"succeeded", "uncertain"}:
        raise FeasibilityError("probe requires a recorded observation")
    obs = Observation.model_validate(event["observation"])
    if obs.terminated or obs.mode in {"game_over", "more"}:
        raise FeasibilityError("probe state cannot accept a decision")
    trial = corpus.trials[event["trial_id"]]
    return event, obs, trial


def _replay(corpus: Corpus, event: dict) -> list[DecisionTransition]:
    history = []
    for _ in range(corpus.manifest.actor.history_interactions):
        if event["kind"] != "action" or event["outcome"] != "succeeded":
            break
        call = corpus.calls.get(event["model_call_id"])
        if call is None or call["parsed_action"] is None or call["decision_rationale"] is None:
            break
        before = corpus.events[call["observation_event_id"]]
        if before["observation"] is None or before["outcome"] not in {"succeeded", "uncertain"}:
            break
        history.append(DecisionTransition(
            before["id"], Observation.model_validate(before["observation"]), before["outcome"],
            call["id"], call["parsed_action"], call["decision_rationale"], event["id"],
            Observation.model_validate(event["observation"]), event["outcome"],
        ))
        event = before
    history.reverse()
    return history


def verify_candidates(corpus: Corpus, cases: dict, cases_digest: str, artifact: dict) -> dict[str, dict]:
    if (artifact["corpus_sha256"] != corpus.sha256 or artifact["cases_sha256"] != cases_digest
            or artifact["split"] != cases["split"] or artifact["origin"] != cases["origin"]):
        raise FeasibilityError("candidates do not belong to this case artifact and corpus")
    records = {}
    for case, record in zip(cases["cases"], artifact["candidates"], strict=True):
        if record["candidate_id"] != case["case_id"] or record["case_id"] != case["case_id"]:
            raise FeasibilityError("candidate does not belong to the prepared case")
        candidate = record.get("candidate")
        if candidate is not None:
            validated = validate_candidate(candidate, case)
            if record["status"] != validated.assessment:
                raise FeasibilityError("candidate status does not match assessment")
        elif record["status"] not in {"invalid", "error", "unstarted"}:
            raise FeasibilityError("candidate is missing its schema-validated draft")
        records[record["candidate_id"]] = record
    return records


def freeze(
    corpus: Corpus, cases: dict, cases_digest: str, candidates: dict, candidates_digest: str,
    review_data: dict, actor: ModelSettings,
) -> dict:
    records = verify_candidates(corpus, cases, cases_digest, candidates)
    review = Review.model_validate(review_data)
    if review.corpus_sha256 != corpus.sha256 or review.candidates_sha256 != candidates_digest:
        raise FeasibilityError("human review must approve these exact corpus and candidate artifacts")
    by_case = {case["case_id"]: case for case in cases["cases"]}
    audits = {}
    for audit in review.audits:
        if audit.candidate_id in audits or not records.get(audit.candidate_id, {}).get("candidate"):
            raise FeasibilityError("audit references a missing or duplicate candidate")
        draft = records[audit.candidate_id]["candidate"]
        if audit.original != draft["claim"]:
            raise FeasibilityError("review must preserve the original candidate claim")
        for event_id in audit.corroborating_event_ids + audit.counterevidence_event_ids:
            event = corpus.events.get(event_id)
            if event is None or event["observation"] is None or event["outcome"] != "succeeded":
                raise FeasibilityError("audit cites an unavailable or uncertain observation")
        audits[audit.candidate_id] = audit

    frozen = {}
    for probe in review.probes:
        audit = audits.get(probe.candidate_id)
        if audit is None:
            raise FeasibilityError("probe requires a human-reviewed candidate")
        if audit.evidential_support not in {"supported", "refuted"} or audit.factual_correctness == "unknown":
            raise FeasibilityError("probe requires a resolved evidence audit")
        if audit.original == audit.corrected:
            raise FeasibilityError("the corrected rule must differ from the original")
        source = by_case[probe.candidate_id]
        event, obs, trial = _before(corpus, probe.pre_event_id)
        call_ids = [row["id"] for row in corpus.calls.values()
                    if row["trial_id"] == event["trial_id"] and row["purpose"] == "decision"
                    and row["observation_event_id"] == event["id"]]
        if not call_ids:
            raise FeasibilityError("probe must be anchored to a recorded pre-decision observation")
        if source["episode_id"] == trial["episode_id"]:
            relation = "same_episode_diagnostic"
            if max(source["evidence_event_ids"]) > event["id"]:
                raise FeasibilityError("probe would receive evidence from its own future")
        elif source["seed"] is not None and trial["observed_reset_seed"] is not None and source["seed"] != trial["observed_reset_seed"]:
            relation = "transfer"
        else:
            raise FeasibilityError("transfer probe requires distinct episodes and observed seeds")
        reviewed_evidence = audit.corroborating_event_ids + audit.counterevidence_event_ids
        if any(
            corpus.events[ref]["session_id"] == trial["session_id"] and ref > event["id"]
            for ref in reviewed_evidence
        ):
            raise FeasibilityError("reviewed rule would use a probe observation from the future")
        probe_id = f"{probe.candidate_id}@{event['id']}"
        if probe_id in frozen:
            raise FeasibilityError("duplicate probe")
        rubric = {}
        actions_seen = set()
        for label in ("acceptable", "harmful", "indeterminate"):
            group = getattr(probe, label)
            rubric[label] = []
            for action in group:
                validate_mode(Decision(action=action, rationale="Human rubric."), obs.mode)
                dumped = action.model_dump(mode="json", exclude_none=True)
                key = _action_key(dumped)
                if key in actions_seen:
                    raise FeasibilityError("rubric assigns an action more than once")
                actions_seen.add(key)
                rubric[label].append(dumped)
        if not rubric["acceptable"]:
            raise FeasibilityError("rubric requires an acceptable action")
        replay = _replay(corpus, event)
        messages = build_messages(obs, replay, observation_outcome=event["outcome"])
        frozen[probe_id] = {
            "probe_id": probe_id, "candidate_id": probe.candidate_id, "family": source["family"],
            "source_evidence_event_ids": source["evidence_event_ids"],
            "source_session_id": source["session_id"], "source_episode_id": source["episode_id"],
            "probe_session_id": trial["session_id"], "probe_episode_id": trial["episode_id"],
            "pre_event_id": event["id"], "decision_call_id": min(call_ids),
            "history_event_ids": [history.output_event_id for history in replay],
            "relation": relation, "applicability": probe.applicability, "mode": obs.mode,
            "rubric": rubric, "rubric_rationale": probe.rationale,
            "rules": {
                "no_rule": None,
                "original": {
                    key: records[audit.candidate_id]["candidate"][key]
                    for key in ("claim", "scope", "conditions", "exceptions")
                },
                "corrected": {"claim": audit.corrected},
            },
            "messages": [{"role": message.type, "content": message.content} for message in messages],
        }

    return {
        "kind": "frozen", "corpus_sha256": corpus.sha256, "cases_sha256": cases_digest,
        "candidates_sha256": candidates_digest, "split": cases["split"], "origin": cases["origin"],
        "review_sha256": digest(canonical(review.model_dump(mode="json"))), "reviewer": review.reviewer,
        "prompt_version": PROBE_VERSION, "actor": actor.model_dump(mode="json"),
        "audits": [audit.model_dump(mode="json") for audit in audits.values()], "probes": list(frozen.values()),
    }


def probe_messages(case: dict, condition: str):
    classes = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}
    messages = [classes[message["role"]](content=message["content"]) for message in case["messages"]]
    # JSON encoding keeps each treatment inside a single prompt field.
    messages[-1] = HumanMessage(content=messages[-1].content + "\nExperiential rule: "
                                + canonical(case["rules"][condition]).decode().rstrip("\n"))
    return messages


async def probe(frozen: dict, frozen_digest: str, run: AnalysisRun | None) -> dict:
    if frozen["prompt_version"] != PROBE_VERSION:
        raise FeasibilityError("unsupported frozen probe prompt version")
    if run and frozen["actor"] != run.settings.model_dump(mode="json"):
        raise FeasibilityError("probe actor configuration must match the frozen specification")
    results = []
    for case in frozen["probes"]:
        can_start = run is not None and run.available(len(CONDITIONS))
        for condition in CONDITIONS:
            entry = {"probe_id": case["probe_id"], "condition": condition}
            if not can_start or not run.available():
                results.append({**entry, "status": "unstarted"})
                continue
            call, content = await run.request(f"{case['probe_id']}:{condition}", PROBE_VERSION,
                                              probe_messages(case, condition))
            entry["call_id"] = call["id"]
            if call["status"] == "error":
                results.append({**entry, "status": "error"})
                continue
            try:
                decision = parse_decision(content, case["mode"])
            except ValueError:
                results.append({**entry, "status": "invalid"})
                continue
            action = decision.action.model_dump(mode="json", exclude_none=True)
            score = next((label for label, actions in case["rubric"].items() if action in actions), "indeterminate")
            results.append({**entry, "status": "valid", "action": action, "rationale": decision.rationale, "score": score})
    return {
        "kind": "probe_results", "corpus_sha256": frozen["corpus_sha256"], "frozen_sha256": frozen_digest,
        "split": frozen["split"], "origin": frozen["origin"],
        "results": results, "analysis": run.provenance() if run else {"calls": []},
    }
