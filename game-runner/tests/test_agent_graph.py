import json
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from game_runner.clients.rogue import Observation, RogueAPIError, RogueTransportError
from game_runner.context.messages import build_messages
from game_runner.context.budget import count_messages
from game_runner.context.planning import _fit_recent
from game_runner.graph.builder import build_graph
from game_runner.decision import Decision, validate_mode
from game_runner.graph.nodes.step import DecisionValidationError, SessionInterrupted
from game_runner.config import Settings
from game_runner.persistence.repository import DecisionTransition
from game_runner.session import run_session


def observed(
    episode_id, step=0, *, status="running", mode="normal", message="A message",
    screen=None,
) -> Observation:
    return Observation.model_validate({
        "episode_id": str(episode_id), "seed": 123, "api_step": step,
        "status": status, "mode": mode, "messages": [message],
        "screen": screen if screen is not None else ["@  .", "   %"],
        "cursor": {"row": 0, "column": 0},
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
        self.history_queries = []

    def recent_transitions(self, session_id, observation_event_id, limit):
        self.history_queries.append((session_id, observation_event_id, limit))
        history = []
        for _ in range(limit):
            if observation_event_id > len(self.events):
                break
            output = self.events[observation_event_id - 1]
            if output["kind"] != "action" or output["outcome"] != "succeeded":
                break
            call_id = output["model_call_id"]
            call = self.calls[call_id - 1]
            input_id = call["observation_event_id"]
            before = self.events[input_id - 1]
            history.append(DecisionTransition(
                input_event_id=input_id,
                input_observation=Observation.model_validate(before["observation"]),
                input_outcome=before["outcome"], model_call_id=call_id,
                action=call["parsed_action"], rationale=call["decision_rationale"],
                output_event_id=observation_event_id,
                output_observation=Observation.model_validate(output["observation"]),
                output_outcome=output["outcome"],
            ))
            observation_event_id = input_id
        return list(reversed(history))

    def create_session(self, session_id, model_name, prompt_version, *, context_policy):
        self.session_id = session_id
        self.model_name = model_name
        self.prompt_version = prompt_version
        self.context_policy = context_policy

    def update_session(self, session_id, status, **kwargs):
        self.session_status = status
        self.session_details = kwargs

    def start_model_call(self, session_id, input_messages, *, observation_event_id, **kwargs):
        self.calls.append({
            "session_id": session_id, "input_messages": input_messages,
            "observation_event_id": observation_event_id,
            **kwargs,
        })
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


def input_state(episode_id):
    return {
        "session_id": uuid4(), "observation": observed(episode_id), "attempts": 0,
        "observation_event_id": 100, "observation_outcome": "succeeded",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", [
    "{}", "```json\n{}\n```", "```\n{}\n```", "````json\n{}\n````",
    " \n```JSON\r\n{}\r\n```\n ",
])
async def test_one_graph_invocation_retries_model_but_executes_one_action(wrapper) -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 1))
    model = FakeModel(
        "not JSON",
        wrapper.format(json.dumps({"action": {"type": "semantic", "action": "MOVE_UP"}, "rationale": "Explore north."})),
    )
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=2)

    result = await graph.ainvoke(input_state(episode_id))

    assert result["next_observation"].api_step == 1
    assert result["next_observation_event_id"] == 1
    assert result["next_observation_outcome"] == "succeeded"
    assert [call["status"] for call in repository.calls] == ["invalid", "valid"]
    assert [call["observation_event_id"] for call in repository.calls] == [100, 100]
    assert len(client.actions) == len(repository.events) == 1
    assert repository.events[0]["outcome"] == "succeeded"
    assert repository.events[0]["model_call_id"] == 2
    assert repository.calls[1]["usage_metadata"]["input_tokens"] == 10
    assert repository.calls[1]["raw_response"]["content"] == wrapper.format(json.dumps({
        "action": {"type": "semantic", "action": "MOVE_UP"}, "rationale": "Explore north.",
    }))


@pytest.mark.asyncio
async def test_provider_error_preserves_observation_link_and_exact_messages() -> None:
    class FailingModel:
        async def ainvoke(self, messages):
            self.messages = messages
            raise RuntimeError("provider unavailable")

    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 1))
    model = FailingModel()
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=2)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await graph.ainvoke(input_state(episode_id))

    assert not client.actions
    call, = repository.calls
    assert call["status"] == "error"
    assert call["observation_event_id"] == 100
    assert call["input_messages"] == [
        {"role": message.type, "content": message.content} for message in model.messages
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```"])
async def test_invalid_decision_never_reaches_game(wrapper) -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 1))
    model = FakeModel(wrapper.format(json.dumps({
        "action": {"type": "semantic", "action": "EAT"}, "rationale": "Eat.",
    })))
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=1)

    with pytest.raises(DecisionValidationError):
        await graph.ainvoke(input_state(episode_id))
    assert not client.actions
    assert repository.calls[0]["status"] == "invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("same_episode, recovered_step", [(True, 0), (True, 2), (False, 1)])
async def test_timeout_with_unverified_state_interrupts_without_retry(same_episode, recovered_step) -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id if same_episode else uuid4(), recovered_step), timeout=True)
    model = FakeModel(json.dumps({
        "action": {"type": "semantic", "action": "REST"}, "rationale": "Wait.",
    }))
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=1)

    with pytest.raises(SessionInterrupted):
        await graph.ainvoke(input_state(episode_id))
    assert len(client.actions) == 1
    assert repository.events[0]["outcome"] == "uncertain"
    assert "observation" not in repository.events[0]


@pytest.mark.asyncio
async def test_timeout_with_advanced_step_recovers_without_retry() -> None:
    episode_id = uuid4()
    repository = FakeRepository()
    client = FakeClient(observed(episode_id, 1), timeout=True)
    model = FakeModel(json.dumps({
        "action": {"type": "semantic", "action": "REST"}, "rationale": "Wait.",
    }))
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=1)

    result = await graph.ainvoke(input_state(episode_id))
    assert result["next_observation"].api_step == 1
    assert result["next_observation_event_id"] == 1
    assert result["next_observation_outcome"] == "uncertain"
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
        await graph.ainvoke(input_state(episode_id))
    assert len(client.actions) == 1
    assert repository.events[0]["outcome"] == "failed"
    assert repository.events[0]["http_status"] == status_code


def test_context_preserves_order_and_drops_whole_oldest_pairs() -> None:
    episode_id = uuid4()
    reset = observed(episode_id, message="Welcome.", screen=["OLD RESET SCREEN"])
    after_rest = observed(episode_id, 1, message="You wait.", screen=["OLD REST SCREEN"])
    current = observed(episode_id, 2, message="You search.", screen=["@  .  ", "   %  "])
    rest = {"type": "semantic", "action": "REST"}
    search = {"type": "semantic", "action": "SEARCH"}
    history = [
        DecisionTransition(1, reset, "succeeded", 1, rest, "Wait.", 2, after_rest, "succeeded"),
        DecisionTransition(2, after_rest, "succeeded", 2, search, "Look.", 3, current, "succeeded"),
    ]
    messages = build_messages(current, history)
    assert [message.type for message in messages] == ["system", "human", "ai", "human", "ai", "human"]
    assert "Welcome." in messages[1].content
    assert json.loads(messages[2].content) == {"action": rest, "rationale": "Wait."}
    assert "You wait." in messages[3].content
    assert json.loads(messages[4].content) == {"action": search, "rationale": "Look."}
    assert "You search." in messages[5].content
    combined = "\n".join(message.content for message in messages)
    assert "OLD RESET SCREEN" not in combined
    assert "OLD REST SCREEN" not in combined
    assert combined.count("@  .  \n   %  ") == 1
    recent_only = [messages[0], *messages[3:]]
    budget = count_messages(None, recent_only).count
    assert _fit_recent(
        history, lambda selected: build_messages(current, selected),
        lambda rendered: count_messages(None, rendered).count <= budget,
    ) == recent_only


def test_selection_mode_accepts_only_matching_key() -> None:
    valid = Decision.model_validate({
        "action": {"type": "key", "key": "k"}, "rationale": "Choose north.",
    })
    validate_mode(valid, "select_direction")
    with pytest.raises(ValueError, match="invalid key"):
        validate_mode(valid, "select_hand")


@pytest.mark.asyncio
@pytest.mark.parametrize("history_limit", [0, 1, 8])
async def test_runner_replays_only_completed_decisions_after_recovery(monkeypatch, history_limit) -> None:
    episode_id = uuid4()

    class RecoveringClient(FakeClient):
        initial = observed(episode_id, message="Reset message.")

        async def action(self, episode_id, action):
            self.actions.append((episode_id, action))
            step = len(self.actions)
            if step == 2:
                raise RogueTransportError("timeout")
            return observed(
                episode_id, step, status="dead" if step == 5 else "running",
                message=f"Action result {step}.",
            )

        async def observe(self):
            return observed(episode_id, 2, message="Recovered state.")

    def decision(rationale):
        return json.dumps({
            "action": {"type": "semantic", "action": "REST"}, "rationale": rationale,
        })

    model = FakeModel("bad JSON", f"```json\n{decision('First')}\n```", *(
        decision(rationale) for rationale in ("Timed out", "Third", "Fourth", "Fifth")
    ))
    client = RecoveringClient(None)
    repository = FakeRepository()
    monkeypatch.setattr("game_runner.session.RogueClient", lambda *_: client)
    monkeypatch.setenv("MODEL_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("MAX_MODEL_CALLS_PER_ACTION", "2")
    monkeypatch.setenv("MAX_GAME_ACTIONS", "10")
    monkeypatch.setenv("HISTORY_RECENT_INTERACTIONS", str(history_limit))

    assert await run_session(Settings.from_env(), repository, model) == "dead"

    assert [call["observation_event_id"] for call in repository.calls] == [1, 1, 2, 3, 4, 5]
    assert [event.get("model_call_id") for event in repository.events] == [None, 2, 3, 4, 5, 6]
    assert [query[2] for query in repository.history_queries] == [history_limit] * 5
    assert repository.prompt_version == "rogue-v1.3"
    assert repository.context_policy == "history_only"
    for call, messages in zip(repository.calls, model.inputs, strict=True):
        assert call["input_messages"] == [
            {"role": message.type, "content": message.content} for message in messages
        ]
    before_timeout = model.inputs[2]
    assert [message.type for message in before_timeout] == (
        ["system", "human", "ai", "human"] if history_limit else ["system", "human"]
    )
    if history_limit:
        assert "Reset message." in before_timeout[1].content
        assert before_timeout[2].content.startswith('{"action":')
        assert json.loads(before_timeout[2].content)["rationale"] == "First"
    assert "Action result 1." in before_timeout[-1].content
    after_recovery = model.inputs[3]
    assert [message.type for message in after_recovery] == ["system", "human"]
    assert "recovered after an action timeout" in after_recovery[-1].content
    assert "Recovered state." in after_recovery[-1].content
    rebuilt_history = [json.loads(message.content)["rationale"] for message in model.inputs[-1] if message.type == "ai"]
    assert rebuilt_history == {0: [], 1: ["Fourth"], 8: ["Third", "Fourth"]}[history_limit]


@pytest.mark.asyncio
@pytest.mark.parametrize("initialize_model", [False, True])
async def test_runner_stops_after_terminal_observation(monkeypatch, initialize_model) -> None:
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
        model_kwargs={"temperature": 0.7, "extra_body": {"top_k": 40}},
    )

    initialized = []

    def initialize(name, **kwargs):
        initialized.append((name, kwargs))
        return model

    monkeypatch.setattr("game_runner.session.init_chat_model", initialize)
    status = await run_session(settings, repository, None if initialize_model else model)

    assert initialized == ([("test-model", {
        "model_provider": "openai", "base_url": "http://localhost:8080/v1",
        "api_key": "local", "timeout": 120, "max_retries": 0,
        "temperature": 0.7, "extra_body": {"top_k": 40}, "max_tokens": 512,
    })] if initialize_model else [])

    assert status == repository.session_status == "dead"
    assert len(client.actions) == 1
    assert [event["kind"] for event in repository.events] == ["reset", "action"]
    assert [call["observation_event_id"] for call in repository.calls] == [1]
