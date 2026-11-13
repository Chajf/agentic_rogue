import json
from pathlib import Path

import pytest

from game_runner.rule_feasibility.analysis import AnalysisRun, ModelSettings
from game_runner.rule_feasibility.artifacts import FAMILIES, FeasibilityError
from game_runner.rule_feasibility.candidates import extract, validate_candidate
from game_runner.rule_feasibility.corpus import load, prepare
from game_runner.rule_feasibility.probes import CONDITIONS, freeze, probe, probe_messages

from test_agent_graph import FakeModel, observed
from test_rule_corpus import sample_manifest, sample_result, write_bundle

CANDIDATE = {
    "claim": "Wait here on every visit.",
    "kind": "strategy", "scope": "general", "conditions": [], "exceptions": [],
    "support_event_ids": [1], "counterevidence_event_ids": [],
    "uncertainty": "Only the conditions in the source observation are known.", "assessment": "draft",
}
SETTINGS = ModelSettings(model_name="fixed-test-model")
REST = {"type": "semantic", "action": "REST"}
UP = {"type": "semantic", "action": "MOVE_UP"}


@pytest.fixture
def scripted_bundle(tmp_path):
    manifest = sample_manifest()
    manifest["trials"] = [
        {"trial_id": "source", "seed": 3, "context_policy": "history_only"},
        {"trial_id": "target", "seed": 4, "context_policy": "working_memory"},
    ]
    source, target = [sample_result(manifest, trial, i) for i, trial in enumerate(manifest["trials"])]
    for family in FAMILIES:
        ref = {"input_event_id": 1, "model_call_id": 1}
        if family != "conditional_rest":
            ref["output_event_id"] = 2
        source["screening_opportunities"][family] = {"count": 1, "records": [ref]}
    records = []
    for result, first in [(source, 1), (target, 10)]:
        owner = {key: result[key] for key in ("series_id", "trial_id", "session_id")}
        for offset in range(3):
            observation = observed(result["episode_id"], offset, message=f"observation {first + offset}")
            records.append({
                **owner, "record_type": "event", "id": first + offset,
                "kind": "reset" if offset == 0 else "action", "api_step": offset,
                "outcome": "uncertain" if first == 10 and offset == 1 else "succeeded",
                "observation": observation.model_copy(update={"seed": result["seed"]}).model_dump(mode="json"),
                "model_call_id": first + offset - 1 if offset else None,
                "progress": None,
            })
            records.append({
                **owner, "record_type": "model_call", "id": first + offset, "purpose": "decision",
                "observation_event_id": first + offset, "status": "valid", "memory_revision_id": None,
                "parsed_action": REST, "decision_rationale": "Fixture decision.",
            })
    path = tmp_path / "scripted"
    write_bundle(path, manifest, [source, target], records)
    return path


async def prepare_review(path: Path):
    corpus = load(path, "development")
    cases = prepare(corpus, "scripted")
    run = AnalysisRun(FakeModel(*(json.dumps(CANDIDATE) for _ in FAMILIES)), SETTINGS, 3, 10)
    candidates = await extract(corpus, cases, "cases-sha", run)
    assert len(run.calls) == 3
    audits = [{
        "candidate_id": candidate["candidate_id"], "factual_correctness": "incorrect",
        "evidential_support": "refuted", "scope": "too_broad", "applicability": "conditional",
        "utility": "mixed", "rationale": "The fixture supports a narrower rule.",
        "original": CANDIDATE["claim"], "corrected": "Wait only under the original conditions.",
        "counterevidence_event_ids": [12],
    } for candidate in candidates["candidates"]]
    review = {
        "schema_version": 1, "corpus_sha256": corpus.sha256, "candidates_sha256": "candidates-sha",
        "reviewer": "Fixture reviewer", "approved": True, "audits": audits,
        "probes": [{
            "candidate_id": audit["candidate_id"], "pre_event_id": event,
            "applicability": scope, "acceptable": [UP], "harmful": [REST],
            "rationale": "Approved fixture rubric.",
        } for audit in audits for event, scope in [(3, "matching"), (12, "out_of_scope")]],
    }
    return corpus, cases, candidates, review


def action_reply(action):
    return json.dumps({"action": action, "rationale": "Fixture model."})


async def test_frozen_probes_reuse_one_prompt_and_pre_decision_history(scripted_bundle):
    corpus, cases, candidates, review = await prepare_review(scripted_bundle)
    frozen = freeze(corpus, cases, "cases-sha", candidates, "candidates-sha", review, SETTINGS)
    local, transfer = frozen["probes"][:2]
    assert (local["relation"], transfer["relation"]) == ("same_episode_diagnostic", "transfer")
    assert local["history_event_ids"] == [2, 3]
    assert transfer["history_event_ids"] == [12]
    assert "observation 10" not in str(transfer["messages"])
    rendered = [probe_messages(transfer, condition) for condition in CONDITIONS]
    assert rendered[0][:-1] == rendered[1][:-1] == rendered[2][:-1]
    assert len({variant[-1].content.rsplit("\nExperiential rule: ", 1)[0] for variant in rendered}) == 1

    model = FakeModel(*([action_reply(UP), "bad JSON", action_reply(UP)] * 6))
    runner = AnalysisRun(model, SETTINGS, 18, 10)
    result = await probe(frozen, "frozen-sha", runner)
    assert len(result["results"]) == 18
    assert [row["condition"] for row in result["results"][:3]] == list(CONDITIONS)
    assert [row["status"] for row in result["results"][:3]] == ["valid", "invalid", "valid"]
    assert len(runner.calls) == 18
    assert result["analysis"]["calls"][1]["raw_response"] == "bad JSON"


async def test_rejects_unapproved_reviews_future_evidence_and_partial_call_budget(scripted_bundle):
    corpus, cases, candidates, review = await prepare_review(scripted_bundle)
    with pytest.raises(ValueError):
        freeze(corpus, cases, "cases-sha", candidates, "candidates-sha", {**review, "approved": False}, SETTINGS)
    review["probes"][0]["pre_event_id"] = 1
    with pytest.raises(FeasibilityError, match="future"):
        freeze(corpus, cases, "cases-sha", candidates, "candidates-sha", review, SETTINGS)
    review["probes"][0]["pre_event_id"] = 3
    frozen = freeze(corpus, cases, "cases-sha", candidates, "candidates-sha", review, SETTINGS)
    run = AnalysisRun(FakeModel(), SETTINGS, 2, 10)
    result = await probe(frozen, "frozen-sha", run)
    assert {row["status"] for row in result["results"]} == {"unstarted"}
    assert run.calls == []


def test_candidate_cannot_cite_an_event_outside_the_supplied_case():
    with pytest.raises(FeasibilityError, match="outside"):
        validate_candidate({**CANDIDATE, "counterevidence_event_ids": [99]}, {"evidence_event_ids": [1]})
