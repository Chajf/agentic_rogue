"""Verify the goal selector's audit, budgets, and action ownership."""

from dataclasses import replace
import json
import os
import sys

import pytest
from langchain_core.messages import AIMessage

from game_runner.budget import RequestBudget, TrialBudget
from game_runner.pilot import store
from game_runner.pilot.__main__ import main as pilot_main
from game_runner.pilot.manifest import Manifest
from game_runner.prompts.goals import ACTOR_GOAL_GUIDANCE, GOAL_SELECTOR_INSTRUCTION
from game_runner.rule_feasibility.artifacts import FeasibilityError
from game_runner.rule_feasibility.corpus import load
from game_runner.session import run_session

from test_agent_graph import FakeClient, FakeModel, FakeRepository
from test_pilot_store import isolated_database  # noqa: F401
from test_prompt_profiles import REST, RUNTIME
from test_rule_corpus import write_bundle
from test_working_state import action, frame
from test_working_state_pilot import STAIR_ACTIONS, STAIR_REPLIES, _state_pair


class AuditRepository(FakeRepository):
    def record_goal_audit(self, session_id, event_id, field, metadata):
        event = self.events[event_id - 1]
        assert event["session_id"] == session_id and event["outcome"] in {"succeeded", "uncertain"}
        assert event["observation"]
        event["progress"][field] = metadata


class ScriptedClient(FakeClient):
    def __init__(self, reset, *results):
        super().__init__(reset)
        self.initial = reset
        self.results = iter(results)

    async def action(self, episode_id, requested):
        self.actions.append((episode_id, requested))
        return next(self.results).model_copy(update={"api_step": len(self.actions)})


def managed_settings(reset, *, actions):
    return replace(
        RUNTIME, game_seed=reset.seed, context_token_budget=16384, max_game_actions=actions,
        prompt_profile="rules-v2", structured_state="map-v1", goal_management="managed-v1",
    )


def decision(request, *, review=False):
    return json.dumps({"action": request, "rationale": "Choose a safe immediate action.", "request_goal_review": review})


RETREAT = '{"choice":"propose","kind":"retreat","target":[1,0],"source_event_id":1}'


async def test_selector_preserves_model_action_and_invalid_review_retains_the_goal():
    reset = frame(".@%.", "....")
    on_stairs = frame("..@.", "....")
    client = ScriptedClient(reset, on_stairs, on_stairs)
    repo = AuditRepository()
    model = FakeModel(
        RETREAT, decision(action("MOVE_RIGHT"), review=True),
        '{"choice":"select","candidate_id":99}', REST,
    )
    status = await run_session(managed_settings(reset, actions=2), repo, model=model, client=client)

    assert status == "limit_reached"
    assert [request.model_dump(mode="json", exclude_none=True) for _, request in client.actions] == [
        action("MOVE_RIGHT"), action("REST"),
    ]
    first, actor, invalid, actor_retry_boundary = repo.calls
    assert [call.get("purpose", "decision") for call in repo.calls] == ["goal_selection", "decision"] * 2
    assert invalid["status"] == "invalid" and invalid["decision_metadata"]["status"] == "invalid"
    assert first["decision_metadata"]["selection"]["goal"]["kind"] == "retreat"
    assert actor["decision_metadata"]["active_goal_id"] == actor_retry_boundary["decision_metadata"]["active_goal_id"]
    assert [event.get("model_call_id") for event in repo.events] == [None, 2, 4]


async def test_review_waits_while_actor_finishes_an_item_selection():
    reset = frame(".@%.", "....")
    menu = frame("", "a) Food", mode="select_item")
    moved = frame("..@.", "....")
    client = ScriptedClient(reset, menu, moved, moved)
    repo = AuditRepository()
    model = FakeModel(
        RETREAT, decision({"type": "key", "key": "e"}, review=True), decision({"type": "key", "key": "a"}),
        '{"choice":"keep"}', REST,
    )

    assert await run_session(managed_settings(reset, actions=3), repo, model=model, client=client) == "limit_reached"

    selector_calls = [call for call in repo.calls if call.get("purpose") == "goal_selection"]
    assert [call["observation_event_id"] for call in selector_calls] == [1, 3]
    assert [request.key for _, request in client.actions[:2]] == ["e", "a"]
    assert selector_calls[-1]["decision_metadata"]["reasons"] == {"actor_requested": 2}


@pytest.mark.parametrize(("trial_calls", "series_calls"), [(1, 5), (5, 1)])
async def test_review_reserves_the_remaining_actor_call(trial_calls, series_calls):
    reset = frame(".@%.", "....")
    repo = AuditRepository()
    client = ScriptedClient(reset, reset)
    budget = TrialBudget(
        RequestBudget(5, trial_calls, 60, "trial"), RequestBudget(5, series_calls, 60, "series"),
    )

    assert await run_session(
        managed_settings(reset, actions=1), repo, model=FakeModel(REST), client=client, budget=budget,
    ) == "limit_reached"

    review = repo.events[0]["progress"]["goal_review"]
    assert (review["status"], review["detail"]) == ("skipped", "reserved_actor_call")
    assert review["reasons"] == {"actionable_map": 1}
    assert len(repo.calls) == len(client.actions) == 1
    assert repo.calls[0]["parsed_action"] == action("REST")
    assert budget.trial.calls == budget.series.calls == 1


async def test_deadline_between_selection_and_actor_keeps_the_review_audit():
    reset = frame(".@%.", "....")
    repo = AuditRepository()
    client = ScriptedClient(reset)
    tick = [0.0]
    budget = TrialBudget(
        RequestBudget(5, 5, 10, "trial", clock=lambda: tick[0]),
        RequestBudget(5, 5, 60, "series", clock=lambda: tick[0]),
    )

    class Selector(FakeModel):
        async def ainvoke(self, messages):
            response = await super().ainvoke(messages)
            tick[0] = 10
            return response

    assert await run_session(
        managed_settings(reset, actions=2), repo, model=Selector(RETREAT), client=client, budget=budget,
    ) == "limit_reached"

    review = repo.events[0]["progress"]["goal_review"]
    assert review["status"] == "resolved" and review["selection"]["goal"]["kind"] == "retreat"
    assert len(repo.calls) == len(repo.events) == 1
    assert repo.calls[0]["purpose"] == "goal_selection" and not client.actions
    assert budget.trial.calls == 1 and budget.trial.actions == 0


def _goal_pair():
    payload = _state_pair()
    payload["trials"][0].update(structured_state="map-v1")
    payload["trials"][1]["goal_management"] = "managed-v1"
    payload["series_limits"]["model_calls"] = 14
    payload["trial_limits"]["model_calls"] = 7
    return payload


def test_manifest_defaults_goal_management_and_serializes_thresholds():
    payload = _goal_pair()
    manifest = Manifest.model_validate(payload)
    assert [trial.goal_management for trial in manifest.trials] == ["off", "managed-v1"]
    assert manifest.model_dump(mode="json")["trials"][1]["goal_review"] == {
        "same_edge_returns": 2, "movement_stall_actions": 6,
    }

    payload["trials"][1]["goal_review"]["movement_stall_actions"] = 0
    with pytest.raises(ValueError, match="greater than 0"):
        Manifest.model_validate(payload)
    payload["trials"][1]["goal_review"]["movement_stall_actions"] = 6

    payload["trials"][1]["structured_state"] = "off"
    with pytest.raises(ValueError, match="managed-v1 requires map-v1"):
        Manifest.model_validate(payload)


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
@pytest.mark.parametrize("history_format", ["replay", "compact"])
def test_native_goal_pair_preserves_actor_actions_and_exports_goal_trace(
    isolated_database, monkeypatch, tmp_path, history_format,
):
    monkeypatch.setattr(store, "connect", isolated_database)
    monkeypatch.setenv("MODEL_BASE_URL", "http://test")
    monkeypatch.setenv("GOAL_MANAGEMENT", "managed-v1")

    class NativeModel(FakeModel):
        async def ainvoke(self, messages):
            if messages[0].content == GOAL_SELECTOR_INSTRUCTION:
                evidence = json.loads(messages[-1].content.split("\nGoal evidence:\n", 1)[1])
                stair = next(row for row in evidence["opportunities"] if row["kind"] == "stairs")
                return AIMessage(content=json.dumps({"choice": "select", "candidate_id": stair["candidate_id"]}))
            return await super().ainvoke(messages)

    monkeypatch.setattr("game_runner.session.init_chat_model", lambda *_, **__: NativeModel(*STAIR_REPLIES))
    payload = _goal_pair()
    for trial in payload["trials"]:
        trial["history_format"] = history_format
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "game_runner.pilot", "run", "--manifest", str(manifest),
        "--rogue-binary", os.environ["TEST_ROGUE_BINARY"], "--runner-revision", "0" * 40,
    ])
    pilot_main()
    output = tmp_path / "export"
    monkeypatch.setattr(sys, "argv", [
        "game_runner.pilot", "export", "--series-id", payload["series_id"], "--output", str(output),
    ])
    pilot_main()

    corpus = load(output, "development")
    assert [(r["goal_management"], r["outcome"]) for r in corpus.trials.values()] == [
        ("off", "limited"), ("managed-v1", "limited"),
    ]
    usage = corpus.trials["second"]["model_usage"]["goal_selection"]
    assert usage["calls"] == 1 and usage["calls_missing_provider_usage"] == 1
    for trial in corpus.trials:
        events = [event for event in corpus.events.values() if event["trial_id"] == trial]
        reset, *actions = events
        assert [event["action_request"] for event in actions] == STAIR_ACTIONS
        assert all(event["outcome"] == "succeeded" for event in actions)
        assert actions[-1]["observation"]["state"]["dungeon_level"] == 2
        for event in actions:
            call = corpus.calls[event["model_call_id"]]
            assert call["purpose"] == "decision" and call["parsed_action"] == event["action_request"]
            assert (ACTOR_GOAL_GUIDANCE in call["input_messages"][-1]["content"]) == (trial == "second")
            assert sum(message["content"].count("Current 80x24 screen") for message in call["input_messages"]) == 1
        if trial == "second":
            selector, = [row for row in corpus.calls.values() if row["purpose"] == "goal_selection"]
            assert selector["parsed_action"] is None and selector["raw_response"]["content"]
            assert selector["observation_event_id"] == reset["id"]
            assert reset["progress"]["goal_review"]["selector_call_id"] == selector["id"]
            goal_id = selector["decision_metadata"]["selection"]["goal"]["id"]
            assert all(event["progress"]["goal_transition"]["goal_id"] == goal_id for event in actions)
            assert actions[-2]["progress"]["goal_transition"]["stage"] == "at_stairs"
            assert actions[-1]["progress"]["goal_transition"]["outcome"] == "complete"
        else:
            assert all("goal_review" not in (event["progress"] or {}) for event in events)

    metadata = json.loads((output / "manifest.json").read_bytes())
    rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    audited_call = next(row for row in rows if row.get("purpose") == "goal_selection")
    later_event = max(row["id"] for row in rows if row["record_type"] == "event")
    selected = audited_call["decision_metadata"]["selection"]["goal"]
    source_event = selected["source_event_id"]
    selected["source_event_id"] = later_event
    tampered = tmp_path / "future-source"
    write_bundle(tampered, metadata, list(corpus.trials.values()), rows)
    with pytest.raises(FeasibilityError, match="goal evidence is outside"):
        load(tampered, "development")
    selected["source_event_id"] = source_event
    actor_call = next(row for row in rows if (row.get("decision_metadata") or {}).get("active_goal_id"))
    actor_call["decision_metadata"]["active_goal_id"] = 99
    tampered_actor = tmp_path / "unknown-goal"
    write_bundle(tampered_actor, metadata, list(corpus.trials.values()), rows)
    with pytest.raises(FeasibilityError, match="actor goal has no preceding selection"):
        load(tampered_actor, "development")


async def test_selector_error_is_audited_and_actor_can_still_defend():
    reset = frame(".@B.", "....")
    repo = AuditRepository()
    client = ScriptedClient(reset, reset)

    class FlakySelector(FakeModel):
        async def ainvoke(self, messages):
            if messages[0].content == GOAL_SELECTOR_INSTRUCTION:
                raise RuntimeError("provider unavailable")
            return await super().ainvoke(messages)

    assert await run_session(
        managed_settings(reset, actions=1), repo, model=FlakySelector(decision(action("MOVE_RIGHT"))), client=client,
    ) == "limit_reached"

    selector, actor = repo.calls
    assert selector["purpose"] == "goal_selection" and selector["status"] == "error"
    assert selector["decision_metadata"]["status"] == "error"
    assert actor["decision_metadata"]["active_goal_id"] is None
    assert repo.events[1]["model_call_id"] == 2 and actor["parsed_action"] == action("MOVE_RIGHT")
