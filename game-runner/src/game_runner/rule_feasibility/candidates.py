"""Draft structured rules using a case's bounded evidence."""

from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from game_runner.pilot.manifest import ManifestModel
from game_runner.responses import decode_json

from .analysis import AnalysisRun
from .artifacts import FeasibilityError, canonical
from .corpus import Corpus

EXTRACTION_PROMPT = "rule-extraction-v1"
EventRef = Annotated[int, Field(strict=True, gt=0)]
Text = Annotated[str, Field(min_length=1, max_length=800)]


class Candidate(ManifestModel):
    claim: Text
    kind: Literal["fact", "strategy", "hypothesis"]
    scope: Literal["map_instance", "episode", "conditional", "general"]
    conditions: list[Text] = Field(max_length=8)
    exceptions: list[Text] = Field(max_length=8)
    support_event_ids: list[EventRef] = Field(max_length=8)
    counterevidence_event_ids: list[EventRef] = Field(max_length=8)
    uncertainty: Text
    assessment: Literal["draft", "insufficient", "conflicting"]


def validate_candidate(value, case: dict) -> Candidate:
    candidate = Candidate.model_validate(value)
    referenced = set(candidate.support_event_ids) | set(candidate.counterevidence_event_ids)
    if referenced - set(case["evidence_event_ids"]):
        raise FeasibilityError("candidate cites evidence outside its supplied case")
    if candidate.assessment == "draft" and not candidate.support_event_ids:
        raise FeasibilityError("a draft candidate needs a support reference")
    return candidate


async def extract(corpus: Corpus, cases: dict, cases_digest: str, run: AnalysisRun | None) -> dict:
    results = []
    schema = canonical(Candidate.model_json_schema()).decode()
    for case in cases["cases"]:
        identity = {"candidate_id": case["case_id"], "case_id": case["case_id"]}
        if run is None or not run.available():
            results.append({**identity, "status": "unstarted", "reason": "extraction_disabled_or_budget"})
            continue
        evidence = [{key: event[key] for key in ("id", "observation", "progress", "outcome")}
                    for event_id in case["evidence_event_ids"] if (event := corpus.events.get(event_id))]
        action = corpus.calls[case["model_call_id"]].get("parsed_action") if "output_event_id" in case else None
        messages = [
            SystemMessage(content="Draft one experiential rule from the supplied observations. "
                          "Distinguish observed effects from interpretations. Keep randomized item identities local. "
                          "Use only supplied event IDs. Return JSON matching this schema.\n" + schema),
            HumanMessage(content=canonical({"family": case["family"], "evidence": evidence, "action": action}).decode()),
        ]
        call, content = await run.request(case["case_id"], EXTRACTION_PROMPT, messages)
        record = {**identity, "call_id": call["id"]}
        if call["status"] == "error":
            results.append({**record, "status": "error"})
            continue
        try:
            candidate = validate_candidate(decode_json(content), case)
        except ValueError:
            results.append({**record, "status": "invalid", "reason": "schema_or_evidence_reference"})
            continue
        results.append({**record, "status": candidate.assessment, "candidate": candidate.model_dump(mode="json")})
    return {
        "kind": "candidates", "cases_sha256": cases_digest, "corpus_sha256": corpus.sha256,
        "split": cases["split"], "origin": cases["origin"], "candidates": results,
        "analysis": run.provenance() if run else {"calls": []},
    }
