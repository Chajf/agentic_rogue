"""Verify history format and delivered window through the actor and pilot export."""

from dataclasses import replace
import json
import os
import sys

import pytest

from game_runner.context.budget import count_messages
from game_runner.pilot import store
from game_runner.pilot.__main__ import main as pilot_main
from game_runner.pilot.manifest import Manifest
from game_runner.rule_feasibility.artifacts import FeasibilityError
from game_runner.rule_feasibility.corpus import load
from game_runner.session import run_session

from test_agent_graph import FakeClient, FakeModel, FakeRepository
from test_pilot_store import isolated_database  # noqa: F401
from test_prompt_profiles import REST, RUNTIME
from test_rule_corpus import write_bundle
from test_working_state import frame
from test_working_state_pilot import STAIR_ACTIONS, STAIR_REPLIES, _state_pair

HISTORY_HEADER = "Recent completed attempts (oldest first)."


def _history_pair(context_tokens):
    payload = _state_pair()
    payload["actor"]["context_tokens"] = context_tokens
    payload["trials"][0].update(structured_state="map-v1")
    payload["trials"][0].pop("history_format")
    payload["trials"][1].update(history_format="compact")
    return payload


def test_manifest_defaults_replay_and_allows_compact_by_itself():
    payload = _history_pair(16384)
    manifest = Manifest.model_validate(payload)
    assert [manifest.settings(RUNTIME, trial).history_format for trial in manifest.trials] == ["replay", "compact"]
    payload["trials"] = payload["trials"][1:]
    assert Manifest.model_validate(payload).trials[0].history_format == "compact"

    payload["trials"][0]["structured_state"] = "off"
    with pytest.raises(ValueError, match="compact history requires map-v1"):
        Manifest.model_validate(payload)


async def test_retry_records_the_same_compact_window_and_counts_its_actual_input():
    reset = frame(".@%.", "....")

    class SteppingClient(FakeClient):
        async def action(self, episode_id, action):
            self.actions.append((episode_id, action))
            return reset.model_copy(update={"api_step": len(self.actions)})

    client = SteppingClient(reset)
    client.initial = reset
    repo = FakeRepository()
    model = FakeModel(REST, "bad JSON", REST)
    settings = replace(
        RUNTIME, game_seed=reset.seed, max_game_actions=2,
        prompt_profile="rules-v2", structured_state="map-v1", history_format="compact",
    )

    assert await run_session(settings, repo, model=model, client=client) == "limit_reached"

    initial, invalid, retry = repo.calls
    assert initial["history_window"] == {
        "format": "compact", "eligible_output_event_ids": [], "delivered_output_event_ids": [],
    }
    assert invalid["history_window"] == retry["history_window"] == {
        "format": "compact", "eligible_output_event_ids": [2], "delivered_output_event_ids": [2],
    }
    for call, messages in zip(repo.calls, model.inputs, strict=True):
        measured = count_messages(model, messages)
        assert (call["input_token_count"], call["input_token_count_method"]) == (measured.count, measured.method)
    assert retry["input_token_count"] > invalid["input_token_count"]
    assert "previous response failed validation" in retry["input_messages"][-1]["content"]


def _trial_rows(rows, trial_id):
    return [row for row in rows.values() if row["trial_id"] == trial_id]


@pytest.mark.skipif(not os.getenv("TEST_ROGUE_BINARY"), reason="TEST_ROGUE_BINARY is not set")
@pytest.mark.parametrize("context_tokens", [16384, 8192], ids=["matched-window", "tight-window"])
def test_native_history_comparison_tracks_delivered_windows(isolated_database, monkeypatch, tmp_path, context_tokens):
    monkeypatch.setattr(store, "connect", isolated_database)
    monkeypatch.setenv("MODEL_BASE_URL", "http://test")
    monkeypatch.setenv("PROMPT_PROFILE", "rules-v2")
    monkeypatch.setenv("STRUCTURED_STATE", "map-v1")
    monkeypatch.setenv("HISTORY_FORMAT", "compact")
    monkeypatch.setattr("game_runner.session.init_chat_model", lambda *_, **__: FakeModel(*STAIR_REPLIES))
    payload = _history_pair(context_tokens)
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
    assert [(row["history_format"], row["outcome"]) for row in corpus.trials.values()] == [
        ("replay", "limited"), ("compact", "limited"),
    ]
    was_truncated = False
    calls_by_format = []
    for trial_id, history_format in (("first", "replay"), ("second", "compact")):
        calls = _trial_rows(corpus.calls, trial_id)
        events = _trial_rows(corpus.events, trial_id)
        calls_by_format.append(calls)
        reset, *actions = events
        assert [row["action_request"] for row in actions] == STAIR_ACTIONS
        assert actions[-1]["observation"]["state"]["dungeon_level"] == 2
        for step, (call, event) in enumerate(zip(calls, actions, strict=True)):
            assert call["parsed_action"] == event["action_request"]
            assert call["observation_event_id"] == events[step]["id"]
            window = call["history_window"]
            assert window["format"] == history_format
            assert window["eligible_output_event_ids"] == [row["id"] for row in actions[:step]]
            delivered = window["delivered_output_event_ids"]
            if context_tokens == 16384:
                assert delivered == window["eligible_output_event_ids"]
            was_truncated |= len(delivered) < step
            messages = call["input_messages"]
            assert messages[-1]["role"] == "human"
            assert sum(row["content"].count("Current 80x24 screen") for row in messages) == 1
            if history_format == "compact":
                current = messages[-1]["content"]
                assert len(messages) == 2
                assert (HISTORY_HEADER in current) == bool(delivered)
                if delivered:
                    rows = current.split("\n\nCurrent API step", 1)[0].splitlines()[1:]
                    assert [int(row.split()[0]) for row in rows] == [corpus.events[i]["api_step"] for i in delivered]
            else:
                assert len(messages) == 2 + 2 * len(delivered)
            assert call["input_token_count_method"] == "utf8_byte_estimate"
            assert call["input_token_count"] == 8 + sum(8 + len(row["content"].encode("utf-8")) for row in messages)

    replay, compact = calls_by_format
    assert all(a["input_messages"][0] == b["input_messages"][0] for a, b in zip(replay, compact, strict=True))
    if context_tokens == 8192:
        assert was_truncated
    else:
        assert not was_truncated
        assert compact[-1]["input_token_count"] < replay[-1]["input_token_count"]

        # Checksum the altered bundle to verify the window check itself rejects future evidence.
        manifest_data = json.loads((output / "manifest.json").read_bytes())
        rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
        first_call = next(row for row in rows if row["record_type"] == "model_call")
        future = next(row["id"] for row in rows if row["record_type"] == "event" and row["kind"] == "action")
        first_call["history_window"]["eligible_output_event_ids"] = [future]
        tampered = tmp_path / "tampered"
        write_bundle(tampered, manifest_data, list(corpus.trials.values()), rows)
        with pytest.raises(FeasibilityError, match="history window is not linked"):
            load(tampered, "development")
