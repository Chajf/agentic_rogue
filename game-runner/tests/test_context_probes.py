from copy import deepcopy
import json

import pytest

from game_runner.rule_feasibility.analysis import AnalysisRun, ModelSettings
from game_runner.rule_feasibility.artifacts import FeasibilityError, canonical
from game_runner.rule_feasibility.candidates import extract
from game_runner.rule_feasibility.context_probes import (
    PROGRESS, UNCOVERED, freeze_context, freeze_correction, prompt_rows, score_action,
)
from game_runner.rule_feasibility.context_results import context_report, run_context
from game_runner.rule_feasibility.corpus import load, prepare

from test_agent_graph import FakeModel
from test_pilot_trajectory import _bundle, _step
from test_rule_probes import scripted_bundle as scripted_case_bundle  # noqa: F401


def _action(name):
    return {"type": "semantic", "action": name}


@pytest.fixture
def replay_case(tmp_path):
    corpus = load(_bundle(tmp_path, ".@%...", [
        ("MOVE_RIGHT", _step("..@...")), ("REST", _step("..@...")),
    ]), "development")
    call = corpus.calls[102]
    event = corpus.events[2]
    content = {"hypotheses": [{"text": "The tile might be stairs.", "source_event_ids": [1]}]}
    corpus.revisions[10] = {
        "id": 10, "trial_id": call["trial_id"], "session_id": call["session_id"], "through_event_id": 1,
        "gap_event_ids": [], "content": content,
    }
    call["memory_revision_id"] = 10
    call["input_messages"] = [
        {"role": "system", "content": "Original instructions."},
        {"role": "human", "content": "Original observation."},
        {"role": "ai", "content": "Original decision."},
        {"role": "human", "content": (
            "Current screen\n..@..."
            + PROGRESS + json.dumps(event["progress"])
            + "\nEpisode memory through event 1 (historical evidence, 0 recovery gaps): " + json.dumps(content)
            + UNCOVERED + "[]" + "\nReturn one action."
        )},
    ]
    review = {
        "schema_version": 1, "corpus_sha256": corpus.sha256, "origin": "natural",
        "reviewer": "Fixture", "approved": True,
        "cases": [{
            "case_id": "hidden-stairs", "decision_call_id": 102, "kind": "history",
            "rubric": {"acceptable": [_action("DESCEND")], "harmful": [_action("MOVE_UP")],
                       "rationale": "The player occupies the last-seen tile."},
            "summary_entry": {"category": "hypotheses", "index": 0}, "stair_event_id": 1,
        }],
    }
    return corpus, call, review


def test_freezes_only_the_planned_prompt_edits(replay_case):
    corpus, original, review = replay_case
    history = freeze_context(corpus, review)["cases"][0]
    base = prompt_rows(history, "baseline")
    assert base[:-1] == original["input_messages"][:-1]
    assert base[-1]["content"].endswith("\nReturn one action.")
    assert PROGRESS in base[-1]["content"]
    assert "Episode memory" not in base[-1]["content"] and UNCOVERED not in base[-1]["content"]
    expected = {
        "recorded_summary": "The tile might be stairs.",
        "source_fact": "Stairs (%) last seen at row 1, column 2 on this map (observation 1).",
    }
    for name, claim in expected.items():
        treated = prompt_rows(history, name)
        assert treated[:-1] == base[:-1]
        assert treated[-1]["content"].removeprefix(base[-1]["content"]) == (
            '\nHistorical map evidence: ' + canonical({"claim": claim}).decode().rstrip("\n")
        )

    review["cases"][0].update(kind="grounding", stair_event_id=None, summary_entry=None)
    ground = freeze_context(corpus, review)["cases"][0]
    assert prompt_rows(ground, "baseline") == original["input_messages"]
    assert prompt_rows(ground, "instruction")[1:] == original["input_messages"][1:]
    assert prompt_rows(ground, "neighbors")[:-1] == original["input_messages"][:-1]
    assert '"UP":null' in prompt_rows(ground, "neighbors")[-1]["content"]


def test_rejects_future_evidence_and_equivalent_rubric_actions(replay_case):
    corpus, _, review = replay_case
    future = deepcopy(review)
    future["cases"][0]["stair_event_id"] = 3
    with pytest.raises(FeasibilityError, match="earlier confirmed"):
        freeze_context(corpus, future)
    review["cases"][0]["rubric"]["harmful"].append({"type": "key", "key": ">"})
    with pytest.raises(FeasibilityError, match="equivalent"):
        freeze_context(corpus, review)


def test_scores_aliases_only_in_normal_mode():
    key = {"type": "key", "key": "l"}
    rubric = {"acceptable": [_action("MOVE_RIGHT")], "harmful": [], "indeterminate": []}
    assert score_action(key, "normal", rubric) == "acceptable"
    assert score_action(key, "select_item", rubric) == "indeterminate"


def _answer(action):
    return json.dumps({"action": action, "rationale": "Fixture answer."})


async def test_probe_and_report_account_for_partial_execution_and_tampering(replay_case):
    corpus, _, review = replay_case
    spec = freeze_context(corpus, review)
    assert spec["blocks"][0]["conditions"] == list(reversed(spec["blocks"][1]["conditions"]))
    model = FakeModel(_answer(_action("MOVE_UP")), "{broken}", _answer({"type": "key", "key": ">"}))
    settings = ModelSettings.model_validate(spec["actor"])
    run = AnalysisRun(model, settings, max_calls=4, seconds=10)
    result = await run_context(spec, "spec-sha", run)
    report = context_report(spec, "spec-sha", result)
    assert report["planned_calls"] == 6 and report["unstarted_calls"] == 3
    assert report["groups"]["history"]["counts_by_condition"] == {
        "baseline": {"harmful": 1, "unstarted": 1},
        "recorded_summary": {"invalid": 1, "unstarted": 1},
        "source_fact": {"acceptable": 1, "unstarted": 1},
    }
    assert report["groups"]["history"]["contrasts"][1]["acceptable_delta_pp"] == 50
    assert report["recommended_intervention"] is None
    damaged = deepcopy(result)
    damaged["analysis"]["calls"][0]["input_messages"][-1]["content"] += "tampered"
    with pytest.raises(FeasibilityError, match="differs from its frozen treatment"):
        context_report(spec, "spec-sha", damaged)
    result["results"][0]["score"] = "acceptable"
    with pytest.raises(FeasibilityError, match="differs from the frozen rubric"):
        context_report(spec, "spec-sha", result)


async def test_correction_reuses_the_observed_prompt_and_scores_both_directions(replay_case):
    corpus, call, review = replay_case
    rules = {
        "corpus_sha256": corpus.sha256, "split": "development", "origin": "natural",
        "approval": "human_review", "reviewer": "Fixture",
        "probes": [{
            "probe_id": "p@2", "pre_event_id": 2, "probe_session_id": call["session_id"],
            "rules": {"no_rule": None, "original": "Walk away.", "corrected": "Descend on the stair."},
            "rubric": {k: v for k, v in review["cases"][0]["rubric"].items() if k != "rationale"},
            "rubric_rationale": "Choose descent here.", "applicability": "matching",
            "relation": "same_episode_diagnostic", "source_evidence_event_ids": [1],
        }],
    }
    spec = freeze_correction(corpus, rules, "rules-sha", repetitions=2)
    assert spec["cases"][0]["messages"][:-1] == call["input_messages"][:-1]
    assert PROGRESS in spec["cases"][0]["messages"][-1]["content"]

    harmonic = _answer(_action("MOVE_UP"))
    acceptable = _answer(_action("DESCEND"))
    model = FakeModel(harmonic, harmonic, acceptable, harmonic, acceptable, harmonic)
    run = AnalysisRun(model, ModelSettings.model_validate(spec["actor"]), 6, 10)
    result = await run_context(spec, "spec-sha", run)
    report = context_report(spec, "spec-sha", result)
    scoped = report["groups"]["correction"]["contrasts"][-1]
    assert scoped["scope"] == "matching"
    assert scoped["harmful_to_acceptable"] == 1 and scoped["acceptable_to_harmful"] == 1


async def test_rule_extraction_selection_leaves_unselected_cases_unstarted(scripted_case_bundle):
    corpus = load(scripted_case_bundle, "development")
    cases = prepare(corpus, "scripted")
    case_id = cases["cases"][-1]["case_id"]
    model = FakeModel("{}")
    runner = AnalysisRun(model, ModelSettings(model_name="fixture"), 1, 10)
    result = await extract(corpus, cases, "cases", runner, selection=[case_id])
    assert result["selected_case_ids"] == [case_id]
    assert [row["reason"] for row in result["candidates"][:-1]] == ["not_selected"] * (len(cases["cases"]) - 1)
    assert runner.calls[0]["label"] == case_id
    with pytest.raises(FeasibilityError, match="duplicate or unknown"):
        await extract(corpus, cases, "cases", runner, selection=["missing"])



def test_cli_freezes_runs_and_reports_with_a_fake_provider(tmp_path, replay_case, monkeypatch):
    import sys

    from game_runner.rule_feasibility import __main__ as cli

    corpus, _, review = replay_case
    monkeypatch.setattr(cli, "load", lambda *_: corpus)
    model = FakeModel(*[_answer(_action("DESCEND"))] * 6)
    monkeypatch.setattr(cli, "provider", lambda _: model)
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    frozen = tmp_path / "frozen.json"
    results = tmp_path / "results.json"
    report = tmp_path / "report.json"
    markdown = tmp_path / "report.md"
    for args in (
        ["context-freeze", "--corpus", str(tmp_path), "--split", "development", "--review", str(review_path),
         "--output", str(frozen)],
        ["context-probe", "--frozen", str(frozen), "--max-calls", "6", "--seconds", "10", "--output", str(results)],
        ["context-report", "--frozen", str(frozen), "--results", str(results), "--output", str(report),
         "--markdown", str(markdown)],
    ):
        monkeypatch.setattr(sys, "argv", ["rule_feasibility", *args])
        cli.main()
    assert len(model.inputs) == 6
    assert json.loads(report.read_text())["groups"]["history"]["cases"] == 1
    assert "6 planned calls" in markdown.read_text()

    frozen.write_text(frozen.read_text() + " ")
    monkeypatch.setattr(sys, "argv", [
        "rule_feasibility", "context-probe", "--frozen", str(frozen), "--max-calls", "6", "--seconds", "10",
        "--output", str(tmp_path / "tampered.json"),
    ])
    with pytest.raises(SystemExit, match="2"):
        cli.main()
    assert len(model.inputs) == 6


def test_provider_preserves_explicit_reasoning_setting(monkeypatch):
    from langchain import chat_models

    from game_runner.rule_feasibility.analysis import provider

    observed_settings = []
    monkeypatch.setenv("MODEL_BASE_URL", "http://unused.local")
    monkeypatch.setattr(chat_models, "init_chat_model", lambda name, **kwargs: observed_settings.append(kwargs))
    provider(ModelSettings.model_validate({"model_name": "fixture"}))
    provider(ModelSettings.model_validate({"model_name": "fixture", "reasoning_enabled": False}))
    assert "extra_body" not in observed_settings[0]
    assert observed_settings[1]["extra_body"] == {"reasoning": {"enabled": False}}
    assert observed_settings[1]["max_retries"] == 0
