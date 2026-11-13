import asyncio
from uuid import uuid4

import pytest

from game_runner.budget import BudgetStop, RequestBudget, TrialBudget
from game_runner.config import Settings
from game_runner.context.compaction import compact_batch, evidence_rows
from game_runner.graph.builder import build_graph
from game_runner.persistence.repository import DecisionTransition
from game_runner.session import run_session

from test_agent_graph import FakeClient, FakeModel, FakeRepository, input_state, observed

REST = {"type": "semantic", "action": "REST"}
REPLY = '{"action":{"type":"semantic","action":"REST"},"rationale":"Wait."}'


def budget(*, calls=5, actions=3, duration=10):
    return TrialBudget(RequestBudget(actions, calls, duration, "trial"), RequestBudget(100, 100, 100, "series"))


async def test_actor_retry_budget_has_no_second_call_or_dispatch():
    episode = uuid4()
    repo, client = FakeRepository(), FakeClient(observed(episode, 1))
    limits = budget(calls=1)
    graph = build_graph(FakeModel("invalid", REPLY), client, repo,
                        context_token_budget=8192, max_model_calls=2, budget=limits)
    with pytest.raises(BudgetStop, match="trial_model_call_limit"):
        await graph.ainvoke(input_state(episode))
    assert len(repo.calls) == limits.series.calls == 1
    assert not client.actions


async def test_summary_retry_counts_toward_series_limit():
    episode = uuid4()
    repo = FakeRepository()
    limits = budget()
    limits.series.call_limit = 1
    trace = [DecisionTransition(1, observed(episode), "succeeded", 1, REST, "Wait.",
                                2, observed(episode, 1), "succeeded")]
    with pytest.raises(BudgetStop, match="series_model_call_limit"):
        await compact_batch(FakeModel("invalid", "invalid"), repo, uuid4(), episode, None,
                            trace, evidence_rows(trace), token_budget=8192, output_tokens=512,
                            attempts=2, fits_actor=lambda _: True, budget=limits)
    assert len(repo.calls) == limits.trial.calls == 1
    assert repo.calls[0]["purpose"] == "episode_summary"
    assert not repo.events


async def test_dispatch_deadline_records_uncertainty_without_resending():
    episode = uuid4()

    class SlowClient(FakeClient):
        async def action(self, episode_id, action):
            self.actions.append(action)
            await asyncio.sleep(10)

    repo, client = FakeRepository(), SlowClient(observed(episode, 1))
    limits = budget(duration=0.15)
    graph = build_graph(FakeModel(REPLY), client, repo,
                        context_token_budget=8192, max_model_calls=1, budget=limits)
    with pytest.raises(BudgetStop, match="trial_time_limit"):
        await graph.ainvoke(input_state(episode))
    assert len(client.actions) == limits.series.actions == 1
    assert repo.events[0]["outcome"] == "uncertain"


async def test_seed_mismatch_stops_after_saving_reset():
    episode = uuid4()
    repo, client = FakeRepository(), FakeClient(observed(episode, 1))
    client.initial = observed(episode).model_copy(update={"seed": 7})
    settings = Settings("http://test", 10, "http://test", "fake", "local", 5, 42, 3, 2, 8192)
    stops = []
    assert await run_session(settings, repo, FakeModel(REPLY), client=client, budget=budget(),
                             on_finish=lambda status, reason: stops.append((status, reason))) == "error"
    assert stops == [("error", "seed_mismatch")]
    assert repo.events[0]["observation"]["seed"] == 7
    assert not client.actions
