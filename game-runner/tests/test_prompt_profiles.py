"""Verify the prompt-only contrast through rendering and the pilot entry point."""

from dataclasses import replace
import json
import os
import sys

import pytest

from game_runner.config import Settings
from game_runner.context.messages import build_messages
from game_runner.context.progress import Progress
from game_runner.decision import parse_decision
from game_runner.pilot import store
from game_runner.pilot.__main__ import main as pilot_main
from game_runner.pilot.manifest import Manifest
from game_runner.prompts.game_rules import GAME_RULES
from game_runner.prompts.rules_v2 import RULES_V2_INSTRUCTION
from game_runner.prompts.system import SYSTEM_INSTRUCTION
from game_runner.rule_feasibility.corpus import load
from game_runner.session import run_session

from test_agent_graph import FakeClient, FakeModel, FakeRepository
from test_pilot_store import isolated_database, make_manifest  # noqa: F401
from test_prompt_bytes import ACTION, fixed_trace

RUNTIME = Settings("http://test", 10, "http://test", "fake", "local", 5, 1234, 3, 2, 8192)
REST = json.dumps({"action": ACTION, "rationale": "Wait."})

def _prompt_pair():
    payload = make_manifest().model_dump(mode="json")
    payload["trials"][0].pop("prompt_profile")
    payload["trials"][1].update(context_policy="history_only", prompt_profile="rules-v2")
    return payload


def test_manifest_accepts_rules_v2_by_itself_and_defaults_legacy_profile():
    payload = _prompt_pair()
    manifest = Manifest.model_validate(payload)
    assert [trial.prompt_profile for trial in manifest.trials] == ["legacy", "rules-v2"]
    assert [manifest.settings(RUNTIME, trial).prompt_profile for trial in manifest.trials] == ["legacy", "rules-v2"]
    payload["trials"] = payload["trials"][1:]
    assert Manifest.model_validate(payload).trials[0].prompt_profile == "rules-v2"


@pytest.mark.parametrize(("field", "value"), [
    ("context_policy", "working_memory"),
    ("context_variant", "neighbors"),
    ("split", "pilot_evaluation"),
])
def test_manifest_rejects_confounded_prompt_trials(field, value):
    payload = _prompt_pair()
    if field == "split":
        payload[field] = value
    else:
        payload["trials"][1][field] = value
    with pytest.raises(ValueError, match="rules-v2 trials require development"):
        Manifest.model_validate(payload)


def test_rules_v2_changes_only_the_system_message():
    trace = fixed_trace()
    progress = Progress.initial(1, trace.input_observation).advance(
        2, trace.input_observation, trace.action, trace.output_observation, recovered=False,
    )
    legacy = build_messages(trace.output_observation, [trace], progress=progress)
    rules = build_messages(trace.output_observation, [trace], progress=progress, prompt_profile="rules-v2")

    assert [message.type for message in rules] == ["system", "human", "ai", "human"]
    assert legacy[1:] == rules[1:]
    assert legacy[0].content != rules[0].content
    assert GAME_RULES in rules[0].content
    label = "Current 80x24 screen (spaces are significant):\n"
    assert sum(message.content.count(label) for message in rules) == 1
    assert label + "\n".join(trace.output_observation.screen) in rules[-1].content
    assert "Observed progress" in rules[-1].content
    for mechanic in (
        "# is a traversable corridor", "+ is a doorway", "? scroll", "blank map cell",
        "zero-based [row, column]", "while your position stays the same",
        "stand on stairs and issue DESCEND", "final score can include the value",
    ):
        assert mechanic in GAME_RULES


@pytest.mark.parametrize(("mode", "action", "instruction"), [
    ("normal", {"type": "semantic", "action": "MOVE_UP_LEFT"}, "MOVE_UP_LEFT"),
    ("normal", {"type": "semantic", "action": "DESCEND"}, "DESCEND"),
    ("normal", {"type": "semantic", "action": "THROW", "item": "a", "direction": "UP"}, "THROW and ZAP require item plus direction"),
    ("normal", {"type": "semantic", "action": "PUT_ON_RING", "item": "a", "hand": "LEFT"}, "may specify hand LEFT or RIGHT"),
    ("select_item", {"type": "key", "key": "*"}, "In select_item"),
    ("select_direction", {"type": "key", "key": "u"}, "u up-right"),
    ("select_hand", {"type": "key", "key": "l"}, "In select_hand send l or r"),
    ("confirm", {"type": "semantic", "action": "CONFIRM"}, "In confirm use semantic CONFIRM or CANCEL"),
    ("normal", {"type": "key", "key": "ENTER"}, "send key ENTER, even in normal mode"),
])
def test_guide_examples_match_the_action_contract(mode, action, instruction):
    response = json.dumps({"action": action, "rationale": "Respond to the current game state."})
    assert parse_decision(response, mode).action.model_dump(mode="json", exclude_none=True) == action
    assert instruction in RULES_V2_INSTRUCTION


def test_complete_response_examples_match_the_decision_contract():
    semantic, key = (line for line in RULES_V2_INSTRUCTION.splitlines() if line.startswith("{"))
    assert parse_decision(semantic, "normal").action.action == "MOVE_LEFT"
    assert parse_decision(key, "select_item").action.key == "a"


async def test_standalone_profile_records_the_delivered_instruction():
    trace = fixed_trace()
    client = FakeClient(trace.output_observation)
    client.initial = trace.input_observation
    repo = FakeRepository()
    settings = replace(RUNTIME, game_seed=trace.input_observation.seed, max_game_actions=1, prompt_profile="rules-v2")

    await run_session(settings, repo, model=FakeModel(REST), client=client)

    assert repo.prompt_version == "rules-v2.2"
    assert repo.calls[0]["input_messages"][0] == {"role": "system", "content": RULES_V2_INSTRUCTION}
    assert client.actions[0][1].model_dump(mode="json", exclude_none=True) == ACTION


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
def test_native_prompt_pair_exports_profile_and_exact_dispatch(isolated_database, monkeypatch, tmp_path):
    monkeypatch.setattr(store, "connect", isolated_database)
    monkeypatch.setenv("MODEL_BASE_URL", "http://test")
    monkeypatch.setenv("PROMPT_PROFILE", "rules-v2")
    monkeypatch.setattr("game_runner.session.init_chat_model", lambda *_, **__: FakeModel(REST, REST, REST))
    payload = _prompt_pair()
    payload["actor"]["context_tokens"] = 16384
    payload["trials"].append({**payload["trials"][1], "trial_id": "unstarted"})
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
    versions = {"first": "rogue-v1.3", "second": "rules-v2.2", "unstarted": "rules-v2.2"}
    metadata = json.loads((output / "manifest.json").read_bytes())
    assert metadata["provenance"]["prompt_versions_by_trial"] == versions
    assert "prompt_version" not in metadata["provenance"]
    assert [
        (result["prompt_profile"], result["prompt_version"], result["outcome"])
        for result in corpus.trials.values()
    ] == [("legacy", "rogue-v1.3", "limited"), ("rules-v2", "rules-v2.2", "limited"), ("rules-v2", None, "unstarted")]

    resets = [event["observation"] for event in corpus.events.values() if event["kind"] == "reset"]
    assert len(resets) == 2
    for field in ("screen", "state", "messages", "seed"):
        assert resets[0][field] == resets[1][field]
    for trial_id, instruction in (("first", SYSTEM_INSTRUCTION), ("second", RULES_V2_INSTRUCTION)):
        calls = [call for call in corpus.calls.values() if call["trial_id"] == trial_id]
        actions = [e for e in corpus.events.values() if e["trial_id"] == trial_id and e["kind"] == "action"]
        assert len(calls) == len(actions) == 3
        for step, (call, event) in enumerate(zip(calls, actions, strict=True)):
            assert call["purpose"] == "decision"
            assert call["input_messages"][0] == {"role": "system", "content": instruction}
            assert [message["role"] for message in call["input_messages"]] == ["system", *["human", "ai"] * step, "human"]
            assert event["model_call_id"] == call["id"]
            assert (event["outcome"], event["api_step"]) == ("succeeded", step + 1)
            assert event["action_request"] == call["parsed_action"] == ACTION
