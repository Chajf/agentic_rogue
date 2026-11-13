from dataclasses import replace
import json
from uuid import uuid4

from langchain_core.messages import AIMessage
import pytest

from game_runner.context.budget import ContextLimitError, RETRY_RESERVE, count_messages
from game_runner.context.compaction import evidence_rows
from game_runner.context.messages import build_messages
from game_runner.context.planning import ContextPlanner, ContextSettings
from game_runner.context.progress import Progress
from game_runner.graph.builder import build_graph
from game_runner.graph.nodes.step import DecisionValidationError
from game_runner.persistence.repository import DecisionTransition

from test_agent_graph import FakeClient, FakeModel, FakeRepository, input_state, observed


def test_counts_compatible_model_and_conservative_estimate():
    class Tokenizer:
        def get_num_tokens_from_messages(self, messages):
            return 7

    messages = [AIMessage(content="é")]
    assert count_messages(Tokenizer(), messages).method == "model_tokenizer"
    assert count_messages(Tokenizer(), messages).count == 7
    assert count_messages(None, messages).count == 18


async def test_failed_compaction_keeps_uncovered_evidence_until_budget_exhaustion():
    episode_id = uuid4()
    initial = observed(episode_id, message="First observation")
    latest = observed(episode_id, 1, message="Current observation")
    trace = [DecisionTransition(
        1, initial, "succeeded", 1, {"type": "semantic", "action": "REST"},
        "Wait.", 2, latest, "succeeded",
    )]

    class Repo(FakeRepository):
        def latest_memory(self, session_id):
            return None

        def uncovered_trace(self, session_id, event_id, through_event_id):
            return trace

    repo = Repo()
    summary = FakeModel("invalid", "invalid", "invalid", "invalid")
    settings = ContextSettings(8192, history_limit=0, policy="working_memory")
    progress = Progress.initial(2, latest)
    actor = FakeModel()
    planner = ContextPlanner(actor, repo, settings, summary)

    prepared = await planner.prepare(uuid4(), latest, 2, "succeeded", progress)
    assert prepared.revision_id is None
    assert "First observation" in prepared.messages[-1].content
    assert "Uncompacted episode evidence" in prepared.messages[-1].content
    assert actor.inputs == []

    base = build_messages(latest, [], progress=progress)
    maximum = count_messages(actor, base).count + settings.actor_output_tokens + RETRY_RESERVE
    planner.settings = replace(settings, token_budget=maximum)
    with pytest.raises(ContextLimitError, match="cannot retain uncovered episode evidence"):
        await planner.prepare(uuid4(), latest, 2, "succeeded", progress)
    assert [call["status"] for call in repo.calls] == ["invalid"] * 4
    assert all(call["purpose"] == "episode_summary" for call in repo.calls)
    assert repo.events == []


async def test_oversized_base_context_stops_before_model_or_action():
    episode_id = uuid4()
    model, repository = FakeModel(), FakeRepository()
    client = FakeClient(observed(episode_id, 1))
    graph = build_graph(model, client, repository, context_token_budget=100, max_model_calls=3)
    with pytest.raises(ContextLimitError, match="current observation"):
        await graph.ainvoke(input_state(episode_id))
    assert repository.events == repository.calls == model.inputs == client.actions == []


async def test_retries_remain_bounded_and_preserve_raw_rejections():
    episode_id = uuid4()
    bad_response = json.dumps({"unexpected": "very long input " * 2000})
    model = FakeModel(bad_response, bad_response, bad_response)
    repository, client = FakeRepository(), FakeClient(observed(episode_id, 1))
    graph = build_graph(model, client, repository, context_token_budget=8192, max_model_calls=3)
    with pytest.raises(DecisionValidationError, match="after 3 attempts"):
        await graph.ainvoke(input_state(episode_id))

    original, first_retry, second_retry = model.inputs
    assert count_messages(model, first_retry).count - count_messages(model, original).count <= RETRY_RESERVE
    assert first_retry == second_retry
    assert [call["raw_response"]["content"] for call in repository.calls] == [bad_response] * 3
    assert client.actions == []


def test_recovery_gap_carries_no_action_result_into_summary():
    episode_id = uuid4()
    recovered = observed(episode_id, 1, message="Unknown origin")
    trace = [DecisionTransition(
        1, observed(episode_id), "succeeded", 1, {"type": "semantic", "action": "REST"},
        "An unconfirmed attempt", 2, recovered, "uncertain",
    )]
    row, = evidence_rows(trace)
    assert row["output_event_id"] == 2
    assert set(row) == {"output_event_id", "api_step", "gap"}
