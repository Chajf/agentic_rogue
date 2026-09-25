import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from game_runner.clients.rogue import Observation, RogueAPIError, RogueTransportError
from game_runner.context.messages import build_messages
from game_runner.graph.builder import build_graph
from game_runner.graph.nodes.step import Decision, DecisionValidationError, SessionInterrupted, validate_mode
from game_runner.config import Settings
from game_runner.session import run_session


def observed(episode_id, step=0, *, status="running", mode="normal") -> Observation:
    return Observation.model_validate({
        "episode_id": str(episode_id), "seed": 123, "api_step": step,
        "status": status, "mode": mode, "messages": ["A message"],
        "screen": ["@  .", "   %"], "cursor": {"row": 0, "column": 0},
        "state": {"hp": 12, "dungeon_level": 1}, "raw_output": "",
        "terminated": status != "running", "error": None,
    })


class FakeModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.inputs = []

    async def ainvoke(self, messages):
        self.inputs.append(messages)
        return AIMessage(content=next(self.responses), usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        })


class FakeRepository:
    def __init__(self):
        self.calls = []
        self.events = []

    def recent_actions(self, _):
        return []

    def interrupt_open_sessions(self):
        self.interrupted_old_sessions = True

    def create_session(self, session_id, model_name, prompt_version):
        self.session_id = session_id
        self.model_name = model_name
        self.prompt_version = prompt_version

    def update_session(self, session_id, status, **kwargs):
        self.session_status = status
        self.session_details = kwargs

    def start_model_call(self, session_id, input_messages):
        self.calls.append({"session_id": session_id, "input_messages": input_messages})
        return len(self.calls)

    def finish_model_call(self, call_id, status, **kwargs):
        self.calls[call_id - 1].update(status=status, **kwargs)

    def start_game_event(self, session_id, kind, **kwargs):
        self.events.append({"session_id": session_id, "kind": kind, **kwargs})
        return len(self.events)

    def finish_game_event(self, event_id, outcome, **kwargs):
        self.events[event_id - 1].update(outcome=outcome, **kwargs)


class FakeClient:
    def __init__(self, response, *, timeout=False, api_error=None):
        self.response = response
        self.timeout = timeout
        self.api_error = api_error
        self.actions = []

    async def action(self, episode_id, action):
        self.actions.append((episode_id, action))
        if self.timeout:
            raise RogueTransportError("timeout")
        if self.api_error is not None:
            raise self.api_error
        return self.response

    async def observe(self):
        return self.response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def reset(self, seed=None):
        return self.initial


@pytest.mark.asyncio
async def test_one_graph_invocation_retries_model_but_executes_one_action() -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 1))
    model = FakeModel(
        "not JSON",
        json.dumps({"action": {"type": "semantic", "action": "MOVE_UP"}, "rationale": "Explore north."}),
    )
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=2)

    result = await graph.ainvoke({
        "session_id": uuid4(), "observation": observed(episode_id), "attempts": 0,
    })

    assert result["next_observation"].api_step == 1
    assert [call["status"] for call in repository.calls] == ["invalid", "valid"]
    assert len(client.actions) == len(repository.events) == 1
    assert repository.events[0]["outcome"] == "succeeded"
    assert repository.calls[1]["usage_metadata"]["input_tokens"] == 10


@pytest.mark.asyncio
async def test_invalid_decision_never_reaches_game() -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 1))
    model = FakeModel(json.dumps({
        "action": {"type": "semantic", "action": "EAT"}, "rationale": "Eat.",
    }))
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=1)

    with pytest.raises(DecisionValidationError):
        await graph.ainvoke({
            "session_id": uuid4(), "observation": observed(episode_id), "attempts": 0,
        })
    assert not client.actions
    assert repository.calls[0]["status"] == "invalid"


@pytest.mark.asyncio
async def test_timeout_with_unchanged_step_interrupts_without_retry() -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 0), timeout=True)
    model = FakeModel(json.dumps({
        "action": {"type": "semantic", "action": "REST"}, "rationale": "Wait.",
    }))
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=1)

    with pytest.raises(SessionInterrupted):
        await graph.ainvoke({
            "session_id": uuid4(), "observation": observed(episode_id), "attempts": 0,
        })
    assert len(client.actions) == 1
    assert repository.events[0]["outcome"] == "uncertain"


@pytest.mark.asyncio
async def test_timeout_with_advanced_step_recovers_without_retry() -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 1), timeout=True)
    model = FakeModel(json.dumps({
        "action": {"type": "semantic", "action": "REST"}, "rationale": "Wait.",
    }))
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=1)

    result = await graph.ainvoke({
        "session_id": uuid4(), "observation": observed(episode_id), "attempts": 0,
    })
    assert result["next_observation"].api_step == 1
    assert len(client.actions) == 1
    assert repository.events[0]["outcome"] == "uncertain"
    assert repository.events[0]["observation"]["raw_output"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [409, 422])
async def test_rejected_game_action_is_logged_and_not_retried(status_code) -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 0), api_error=RogueAPIError(status_code, "rejected"))
    model = FakeModel(json.dumps({
        "action": {"type": "semantic", "action": "REST"}, "rationale": "Wait.",
    }))
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=1)

    with pytest.raises(RogueAPIError):
        await graph.ainvoke({
            "session_id": uuid4(), "observation": observed(episode_id), "attempts": 0,
        })
    assert len(client.actions) == 1
    assert repository.events[0]["outcome"] == "failed"
    assert repository.events[0]["http_status"] == status_code


def test_history_omits_previous_screen_and_preserves_current_screen() -> None:
    episode_id = uuid4()
    history = [{
        "action_request": {"type": "semantic", "action": "REST"},
        "observation": {
            "screen": ["OLD SECRET SCREEN"], "messages": ["You wait."],
            "state": {"hp": 12}, "mode": "normal", "status": "running",
        },
    }]
    messages = build_messages(observed(episode_id), history, 8192)
    assert "OLD SECRET SCREEN" not in messages[1].content
    assert "@  ." in messages[1].content
    assert "You wait." in messages[1].content


def test_selection_mode_accepts_only_matching_key() -> None:
    valid = Decision.model_validate({
        "action": {"type": "key", "key": "k"}, "rationale": "Choose north.",
    })
    validate_mode(valid, "select_direction")
    with pytest.raises(ValueError, match="invalid key"):
        validate_mode(valid, "select_hand")


@pytest.mark.asyncio
async def test_runner_stops_after_terminal_observation(monkeypatch) -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 1, status="dead", mode="game_over"))
    client.initial = observed(episode_id)
    monkeypatch.setattr("game_runner.session.RogueClient", lambda *_: client)
    model = FakeModel(json.dumps({
        "action": {"type": "semantic", "action": "REST"}, "rationale": "Wait.",
    }))
    settings = Settings(
        rogue_api_url="http://rogue-api:8000", rogue_timeout_seconds=10,
        model_base_url="http://localhost:8080/v1", model_name="test-model", model_api_key="local",
        model_timeout_seconds=120, game_seed=123, max_game_actions=10,
        max_model_calls_per_action=1, context_token_budget=8192,
    )

    status = await run_session(settings, repository, model)

    assert status == repository.session_status == "dead"
    assert len(client.actions) == 1
    assert [event["kind"] for event in repository.events] == ["reset", "action"]
    assert repository.interrupted_old_sessions
