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
        "corrected_scope": "conditional",
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


async def test_reports_paired_changes_losses_and_unchanged_without_game_access(scripted_bundle, monkeypatch, tmp_path):
    from game_runner.clients.rogue import RogueClient
    from game_runner.persistence import database
    from game_runner.rule_feasibility.artifacts import write_artifact
    from game_runner.rule_feasibility.report import checked_results, summarize, write_markdown

    def unexpected(*args, **kwargs):
        pytest.fail("offline analysis accessed gameplay")

    monkeypatch.setattr(RogueClient, "__init__", unexpected)
    monkeypatch.setattr(database, "connect", unexpected)
    corpus, cases, candidates, review = await prepare_review(scripted_bundle)
    frozen = freeze(corpus, cases, "cases-sha", candidates, "candidates-sha", review, SETTINGS)
    responses = [action_reply(action) for sequence in (
        [UP, REST, UP], [UP, UP, REST], [UP, UP, UP],
        [UP, UP, UP], [UP, UP, UP], [UP, UP, UP],
    ) for action in sequence]
    run = AnalysisRun(FakeModel(*responses), SETTINGS, 18, 10)
    result = await probe(frozen, "frozen-sha", run)
    groups = checked_results(frozen, "frozen-sha", result)
    summary = summarize(cases, candidates, frozen, groups, run.calls)
    movement = summary["families"]["ambiguous_movement"]["paired_counts"]
    assert (movement["changed"], movement["harmful_choices_avoided"], movement["appropriate_recommendations_lost"]) == (2, 1, 1)
    assert summary["families"]["conditional_rest"]["paired_counts"]["unchanged"] == 2
    assert {family["decision"] for family in summary["families"].values()} == {"insufficient evidence"}
    for name in ("first", "second"):
        write_artifact(tmp_path / f"{name}.json", summary)
        write_markdown(tmp_path / f"{name}.md", summary)
    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()
    assert (tmp_path / "first.md").read_bytes() == (tmp_path / "second.md").read_bytes()


async def test_cli_freeze_probe_and_report_share_the_exact_frozen_artifacts(scripted_bundle, monkeypatch, tmp_path):
    import sys

    from game_runner.rule_feasibility import __main__ as cli
    from game_runner.rule_feasibility.artifacts import canonical, digest, write_artifact

    corpus, cases, candidates, review = await prepare_review(scripted_bundle)
    cases_path, drafts_path, review_path, actor_path, frozen_path, results_path, report_path = [
        tmp_path / name for name in ("cases.json", "drafts.json", "review.json", "actor.json",
                                     "frozen.json", "probes.json", "report.json")
    ]
    cases_sha = write_artifact(cases_path, cases)
    candidate_sha = write_artifact(drafts_path, {**candidates, "cases_sha256": cases_sha})
    review_path.write_bytes(canonical({**review, "candidates_sha256": candidate_sha}))
    actor_path.write_bytes(canonical(SETTINGS.model_dump(mode="json")))

    def command(*args):
        monkeypatch.setattr(sys, "argv", ["rule_feasibility", *map(str, args)])
        _, namespace = cli._arguments()
        return cli._execute(namespace)

    common = ("--corpus", scripted_bundle, "--split", "development", "--cases", cases_path,
              "--candidates", drafts_path)
    frozen_sha = command("freeze", *common, "--review", review_path, "--model-config", actor_path,
                         "--output", frozen_path)
    assert digest(frozen_path.read_bytes()) == frozen_sha
    assert frozen_path.with_suffix(".json.sha256").read_text().strip() == frozen_sha

    # Invoke the async CLI from outside this test's event loop.
    import asyncio
    monkeypatch.setattr(cli, "provider", lambda _: FakeModel(*([action_reply(UP)] * 18)))
    args = ("probe", "--frozen", frozen_path, "--max-calls", "18", "--seconds", "10", "--output", results_path)
    await asyncio.to_thread(command, *args)
    command("report", *common, "--frozen", frozen_path, "--results", results_path, "--output", report_path)
    result = json.loads(report_path.read_bytes())
    assert result["usage"]["probes"]["calls"] == 18
    assert result["corpus_sha256"] == corpus.sha256
    frozen_path.write_bytes(frozen_path.read_bytes() + b" ")
    with pytest.raises(FeasibilityError, match="checksum"):
        command("report", *common, "--frozen", frozen_path, "--results", results_path,
                "--output", tmp_path / "second-report.json")
