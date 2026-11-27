import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from game_runner.budget import BudgetStop, RequestBudget, TrialBudget
from game_runner.clients.decisions import DecisionsClient
from game_runner.config import Settings
from game_runner.context.budget import ContextLimitError
from game_runner.context.progress import InventorySnapshot, Progress
from game_runner.graph.builder import build_graph
from game_runner.graph.nodes.actor import DecisionValidationError
from game_runner.graph.nodes.system_one import choice_menu
from game_runner.pilot.export import _usage
from game_runner.pilot.manifest import Manifest
from game_runner.pilot.runner import private_rogue
from game_runner.session import run_session
from test_agent_graph import FakeClient, FakeRepository, input_state, observed


def decisions_client(handler):
    return DecisionsClient(
        "typesafe/jev-1.14", "https://openrouter.ai/api/alpha", "mock-api-key", 2,
        transport=httpx.MockTransport(handler),
    )


def answer(choice):
    return {
        "id": "gen-dec-test", "model": "typesafe/jev-1.14-pinned", "provider": "TypeSafe",
        "answers": {"next_action": {"type": "choice", "choice": choice, "probabilities": {choice: 1}}},
        "usage": {"input_tokens": 100, "output_tokens": 10, "cost": 0.01},
    }


def test_normal_menu_uses_typed_movements_and_stages_argument_commands():
    observation = observed(uuid4(), screen=["---@|"])
    menu = choice_menu(observation, Progress.initial(1, observation))
    assert {name: action.action for name, (_, action) in menu.items() if name.startswith("move_")} == {
        "move_up": "MOVE_UP", "move_down": "MOVE_DOWN", "move_left": "MOVE_LEFT", "move_right": "MOVE_RIGHT",
        "move_up_left": "MOVE_UP_LEFT", "move_up_right": "MOVE_UP_RIGHT",
        "move_down_left": "MOVE_DOWN_LEFT", "move_down_right": "MOVE_DOWN_RIGHT",
    }
    assert menu["throw"][1].key == "t"
    assert menu["put_on_ring"][1].key == "P"


def test_prompt_menus_follow_the_observation_and_inventory_freshness():
    episode = uuid4()
    item = observed(episode, mode="select_item", screen=["a) Some food", "d) A scroll"])
    progress = Progress(event_id=1, map_event_id=None, map_level=1, position=None,
                        inventory=InventorySnapshot(1, (("b", "a potion"),), stale=True))
    assert set(choice_menu(item, progress)) == {"item_a", "item_d", "list_items", "cancel"}
    item = item.model_copy(update={"screen": []})
    assert "item_z" in choice_menu(item, progress)
    direction = choice_menu(observed(episode, mode="select_direction"), progress)
    assert direction["direction_up_right"][1].key == "u"
    hand = choice_menu(observed(episode, mode="select_hand"), progress)
    assert {hand[name][1].key for name in ("hand_left", "hand_right")} == {"l", "r"}
    confirm = choice_menu(observed(episode, mode="confirm"), progress)
    assert {option[1].action for option in confirm.values()} == {"CONFIRM", "CANCEL"}


async def test_retries_choice_on_same_observation_then_dispatches_once():
    episode = uuid4()
    repo = FakeRepository()
    rogue = FakeClient(observed(episode, 1))
    requests = []

    def handler(request):
        assert request.url == "https://openrouter.ai/api/alpha/decisions"
        assert request.headers["Authorization"] == "Bearer mock-api-key"
        payload = json.loads(request.content)
        assert "@  ." in payload["state"]["messages"][-1]["content"]
        assert "Return only JSON" not in payload["state"]["messages"][0]["content"]
        requests.append(payload)
        return httpx.Response(200, json=answer("bogus" if len(requests) == 1 else "move_left"))

    async with decisions_client(handler) as model:
        graph = build_graph(model, rogue, repo, context_token_budget=8192, max_model_calls=2)
        result = await graph.ainvoke(input_state(episode))

    assert result["next_observation"].api_step == 1
    assert [action.action for _, action in rogue.actions] == ["MOVE_LEFT"]
    assert [call["status"] for call in repo.calls] == ["invalid", "valid"]
    assert requests[0]["state"] == requests[1]["state"]
    assert "previous answer was invalid" in requests[1]["questions"]["next_action"]["instructions"]
    call = repo.calls[1]
    assert call["usage_metadata"] == {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "cost": 0.01}
    assert call["response_metadata"]["provider"] == "TypeSafe"
    assert call["decision_metadata"]["rationale_source"] == "actor_adapter"
    assert call["decision_metadata"]["question"] == requests[1]["questions"]["next_action"]
    assert call["decision_metadata"]["option_order"] == list(requests[1]["questions"]["next_action"]["criteria"])
    assert call["decision_metadata"]["model"] == "typesafe/jev-1.14-pinned"
    assert call["decision_metadata"]["answer"]["probabilities"] == {"move_left": 1}
    assert call["raw_response"]["answers"]["next_action"]["probabilities"] == {"move_left": 1}
    assert repo.events[0]["model_call_id"] == 2


@pytest.mark.parametrize("response", [
    httpx.Response(200, json={"answers": {"next_action": {"type": "refusal"}}}),
    httpx.Response(200, text="malformed"),
    httpx.Response(200, json={"answers": {"next_action": {"type": "choice", "choice": []}}}),
])
async def test_invalid_answer_never_dispatches(response):
    episode = uuid4()
    rogue = FakeClient(observed(episode, 1))
    repo = FakeRepository()
    async with decisions_client(lambda _: response) as model:
        graph = build_graph(model, rogue, repo, context_token_budget=8192, max_model_calls=1)
        with pytest.raises(DecisionValidationError):
            await graph.ainvoke(input_state(episode))
    assert [call["status"] for call in repo.calls] == ["invalid"]
    assert not rogue.actions


async def test_forced_enter_records_action_without_a_provider_call():
    episode = uuid4()
    repo = FakeRepository()
    rogue = FakeClient(observed(episode, 1))
    budget = TrialBudget(RequestBudget(2, 1, 30, "trial"), RequestBudget(2, 1, 30, "series"))
    state = {**input_state(episode), "observation": observed(episode, screen=["[Press return to continue]"])}
    async with decisions_client(lambda _: pytest.fail("forced input called provider")) as model:
        graph = build_graph(model, rogue, repo, context_token_budget=8192, max_model_calls=2, budget=budget)
        await graph.ainvoke(state)
    assert [action.key for _, action in rogue.actions] == ["ENTER"]
    assert budget.trial.calls == 0 and budget.trial.actions == 1
    assert repo.calls[0]["decision_metadata"]["decision_source"] == "forced"
    assert repo.calls[0]["usage_metadata"] is None


def test_usage_excludes_forced_inputs_and_reports_provider_cost_coverage():
    call = {
        "usage_metadata": {"total_tokens": 200, "cost": 0.01}, "duration_ms": 100,
        "input_token_count_method": "utf8_byte_estimate", "input_token_count": 400,
    }
    forced = {**call, "usage_metadata": None, "decision_metadata": {"decision_source": "forced"}}
    usage = _usage([call, forced, {**call, "usage_metadata": None}])
    assert usage["forced_inputs"] == 1
    assert usage["calls"] == 2
    assert usage["latency_ms"] == 200
    assert usage["provider_total_tokens"] == 200
    assert usage["calls_missing_provider_usage"] == 1
    assert usage["provider_cost_usd"] == 0.01
    assert usage["calls_with_provider_cost"] == 1


async def test_provider_timeout_is_audited_and_does_not_dispatch():
    async def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    episode = uuid4()
    repo = FakeRepository()
    rogue = FakeClient(observed(episode, 1))
    async with decisions_client(handler) as model:
        graph = build_graph(model, rogue, repo, context_token_budget=8192, max_model_calls=2)
        with pytest.raises(httpx.ReadTimeout):
            await graph.ainvoke(input_state(episode))
    assert [call["validation_error"] for call in repo.calls] == ["ReadTimeout"]
    assert not rogue.actions


async def test_call_budget_and_context_size_stop_before_provider_or_dispatch():
    episode = uuid4()
    repo = FakeRepository()
    rogue = FakeClient(observed(episode, 1))
    budget = TrialBudget(RequestBudget(1, 0, 30, "trial"), RequestBudget(1, 1, 30, "series"))
    async with decisions_client(lambda _: pytest.fail("exhausted budget called provider")) as model:
        graph = build_graph(model, rogue, repo, context_token_budget=8192, max_model_calls=1, budget=budget)
        with pytest.raises(BudgetStop, match="trial_model_call_limit"):
            await graph.ainvoke(input_state(episode))
        graph = build_graph(model, rogue, repo, context_token_budget=1024, max_model_calls=1)
        with pytest.raises(ContextLimitError):
            await graph.ainvoke(input_state(episode))
    assert not repo.calls and not rogue.actions


def test_manifest_selects_actor_and_preserves_legacy_snapshot(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "https://openrouter.ai/api/alpha")
    monkeypatch.setenv("MODEL_NAME", "typesafe/jev-1.14")
    monkeypatch.setenv("ACTOR_TYPE", "system_one")
    monkeypatch.setenv("PROMPT_PROFILE", "rules-v2")
    runtime = Settings.from_env()
    assert runtime.actor_type == "system_one"
    assert runtime.prompt_version == "system-one-v1"
    source = {
        "schema_version": 1, "series_id": str(uuid4()), "split": "development",
        "actor": {"model_name": runtime.model_name},
        "trial_limits": {"actions": 4, "model_calls": 6, "seconds": 30},
        "series_limits": {"actions": 4, "model_calls": 6, "seconds": 30},
        "trials": [{
            "trial_id": "seed-12345", "seed": 12345, "context_policy": "history_only",
            "prompt_profile": "rules-v2", "structured_state": "map-v1",
        }],
    }
    chat = Manifest.model_validate(source)
    assert chat.settings(runtime, chat.trials[0]).actor_type == "chat"
    assert "type" not in chat.snapshot()["actor"]
    source["actor"]["type"] = "system_one"
    manifest = Manifest.model_validate(source)
    settings = manifest.settings(runtime, manifest.trials[0])
    assert settings.actor_type == manifest.snapshot()["actor"]["type"] == "system_one"
    assert settings.prompt_version == manifest.prompt_version("baseline", "rules-v2")
    source["trials"][0]["goal_management"] = "managed-v1"
    with pytest.raises(ValueError, match="System One requires"):
        Manifest.model_validate(source)


@pytest.mark.skipif(not os.environ.get("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
async def test_system_one_session_plays_native_rogue_with_replay(monkeypatch):
    monkeypatch.setenv("ACTOR_TYPE", "system_one")
    monkeypatch.setenv("PROMPT_PROFILE", "rules-v2")
    monkeypatch.setenv("STRUCTURED_STATE", "map-v1")
    monkeypatch.setenv("CONTEXT_TOKEN_BUDGET", "16384")
    monkeypatch.setenv("MODEL_NAME", "typesafe/jev-1.14")
    monkeypatch.setenv("MODEL_BASE_URL", "https://openrouter.ai/api/alpha")
    monkeypatch.setenv("MODEL_API_KEY", "mock-api-key")
    monkeypatch.setenv("GAME_SEED", "12345")
    monkeypatch.setenv("MAX_GAME_ACTIONS", "7")
    settings = Settings.from_env()
    repo = FakeRepository()
    choices = iter(["inventory", "eat", "item_a", "throw", "direction_right", "item_e", "move_left"])
    delivered = []

    def handler(request):
        payload = json.loads(request.content)
        messages = payload["state"]["messages"]
        options = payload["questions"]["next_action"]["criteria"]
        choice = next(choices)
        assert choice in options
        delivered.append(messages)
        return httpx.Response(200, json=answer(choice))

    class MockedDecisionsClient(DecisionsClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("game_runner.session.DecisionsClient", MockedDecisionsClient)
    async with private_rogue(Path(os.environ["TEST_ROGUE_BINARY"]), 2) as (rogue, _):
        status = await run_session(settings, repo, client=rogue)

    assert status == "limit_reached"
    assert repo.prompt_version == "system-one-v1"
    assert [event["api_step"] for event in repo.events] == list(range(8))
    assert repo.events[2]["observation"]["mode"] == "select_item"
    assert any("tasted good" in message.lower() for message in repo.events[3]["observation"]["messages"])
    assert repo.events[4]["observation"]["mode"] == "select_direction"
    assert repo.events[5]["observation"]["mode"] == "select_item"
    for index, messages in enumerate(delivered):
        assert f"Current API step: {index}" in messages[-1]["content"]
        assert repo.calls[index]["history_window"]["delivered_output_event_ids"] == list(range(2, index + 2))
    assert any('"key":"a"' in message["content"] for message in delivered[-1] if message["role"] == "ai")
