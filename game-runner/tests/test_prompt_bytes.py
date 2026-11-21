"""Byte-level reference for prompts rendered by beeb899."""

from hashlib import sha256
import json
from uuid import UUID

import pytest

from game_runner.context.compaction import SummaryUnavailable, compact_batch
from game_runner.context.messages import build_messages
from game_runner.context.progress import Progress
from game_runner.context.transitions import DecisionTransition
from game_runner.graph.builder import build_graph

from test_agent_graph import FakeClient, FakeModel, FakeRepository, input_state, observed

EPISODE = UUID(int=1)
ACTION = {"type": "semantic", "action": "REST"}


def message_digest(messages):
    data = json.dumps(
        [{"role": message.type, "content": message.content} for message in messages],
        ensure_ascii=False, separators=(",", ":"),
    )
    return sha256(data.encode("utf-8")).hexdigest()


def fixed_trace():
    start = observed(EPISODE, screen=["Bienvenue".ljust(80), ".@%".ljust(80)] + [" " * 80] * 22)
    result = start.model_copy(update={"api_step": 1, "messages": ["Après le repos."]})
    transition = DecisionTransition(1, start, "succeeded", 1, ACTION, "Reprendre des forces.", 2, result, "succeeded")
    return transition


def test_legacy_replay_prompt_bytes():
    trace = fixed_trace()
    progress = Progress.initial(1, trace.input_observation).advance(
        2, trace.input_observation, trace.action, trace.output_observation, recovered=False,
    )
    messages = build_messages(trace.output_observation, [trace], progress=progress)
    assert message_digest(messages) == "b535199c5212a00b85b77296597aa6eba862956de973d38470b3e30121f3b566"


async def test_legacy_decision_retry_prompt_bytes():
    model = FakeModel("not JSON", '{"action":{"type":"semantic","action":"REST"},"rationale":"Wait."}')
    graph = build_graph(
        model, FakeClient(fixed_trace().output_observation), FakeRepository(),
        context_token_budget=8192, max_model_calls=2,
    )
    await graph.ainvoke(input_state(EPISODE))
    assert message_digest(model.inputs[1]) == "007363729bf2a92cc5baea6dd97383d9b0a27962144046cba77e5dfe89461a95"


async def test_legacy_summary_and_retry_prompt_bytes():
    model = FakeModel("bad JSON", "still bad JSON")
    with pytest.raises(SummaryUnavailable):
        await compact_batch(
            model, FakeRepository(), UUID(int=2), EPISODE, None,
            [fixed_trace()], [{"output_event_id": 2, "message": "Après le repos."}],
            token_budget=8192, output_tokens=1024, attempts=2, fits_actor=lambda _: True,
        )
    assert [message_digest(messages) for messages in model.inputs] == [
        "aca757ab7ae4868749cbeecec729cfd7e8f87574faf07c65be365c1c4b28222c",
        "63bbddd110fe29fdb691caec7d9ddb7102979b81a5127dc9caaa9c1206877713",
    ]
