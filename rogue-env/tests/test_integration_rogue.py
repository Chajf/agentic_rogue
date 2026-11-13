import os
from pathlib import Path
import sys
import tempfile

import pytest

from rogue_api.config import Settings
from rogue_api.environment import RogueEnv
from rogue_api.models import SemanticAction, SemanticActionRequest


ROGUE_BINARY = os.environ.get("TEST_ROGUE_BINARY")


@pytest.mark.parametrize("draw_screen", [True, False], ids=["delayed-screen", "startup-timeout"])
async def test_reset_waits_for_initial_screen(tmp_path, draw_screen) -> None:
    greeting = b"Hello rogo-agent, just a moment while I dig the dungeon #12345..."
    screen = b"\x1b[2J\x1b[H\x1b[2;1H@\x1b[24;1HLevel: 1 Gold: 0 Hp: 12(12) Str: 16(16) Arm: 4 Exp: 1/0"
    binary = tmp_path / "rogue"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        f"sys.stdout.buffer.write({greeting!r})\n"
        "sys.stdout.flush()\n"
        "time.sleep(0.2)\n"
        f"sys.stdout.buffer.write({screen if draw_screen else b''!r})\n"
        "sys.stdout.flush()\n"
        "time.sleep(10)\n"
    )
    binary.chmod(0o700)
    environment = RogueEnv(Settings(
        rogue_binary=binary,
        data_dir=tmp_path,
        quiet_window_seconds=0.025,
        action_timeout_seconds=3.0 if draw_screen else 0.5,
    ))
    try:
        observation = await environment.reset(12345)
        if draw_screen:
            assert observation.status == "running", observation.error
            assert observation.api_step == 0
            assert observation.state.dungeon_level == 1
            assert observation.screen[1].startswith("@")
            assert observation.raw_output == (greeting + screen).decode()
        else:
            assert observation.status == "error"
            assert observation.terminated
            assert "did not draw its initial screen" in observation.error
            assert not environment.process_running
    finally:
        await environment.close()


@pytest.mark.skipif(not ROGUE_BINARY, reason="TEST_ROGUE_BINARY is not set")
async def test_seeded_native_episode() -> None:
    with tempfile.TemporaryDirectory(prefix="rogue-", dir="/tmp") as data_dir:
        settings = Settings(
            rogue_binary=Path(ROGUE_BINARY),
            data_dir=Path(data_dir),
            quiet_window_seconds=0.05,
            action_timeout_seconds=3.0,
        )
        environment = RogueEnv(settings)
        try:
            initial = await environment.reset(12345)
            assert initial.status == "running", initial.error
            assert initial.api_step == 0
            assert initial.state.dungeon_level == 1
            assert any("@" in row for row in initial.screen)

            repeated = await environment.reset(12345)
            assert repeated.screen == initial.screen
            assert repeated.episode_id != initial.episode_id
            initial = repeated

            moved = await environment.step(
                initial.episode_id,
                SemanticActionRequest(type="semantic", action=SemanticAction.REST),
            )
            assert moved.status == "running"
            assert moved.api_step == 1
            assert moved.episode_id == initial.episode_id

            quit_prompt = await environment.step(
                initial.episode_id,
                SemanticActionRequest(type="semantic", action=SemanticAction.QUIT),
            )
            assert quit_prompt.mode == "confirm"
            assert quit_prompt.api_step == 2

            resumed = await environment.step(
                initial.episode_id,
                SemanticActionRequest(type="semantic", action=SemanticAction.CANCEL),
            )
            assert resumed.status == "running"
            assert resumed.mode == "normal"
            assert resumed.api_step == 3
        finally:
            await environment.close()


@pytest.mark.skipif(not ROGUE_BINARY, reason="TEST_ROGUE_BINARY is not set")
async def test_collects_messages_across_more_prompt() -> None:
    with tempfile.TemporaryDirectory(prefix="rogue-", dir="/tmp") as data_dir:
        environment = RogueEnv(
            Settings(
                rogue_binary=Path(ROGUE_BINARY),
                data_dir=Path(data_dir),
                quiet_window_seconds=0.05,
                action_timeout_seconds=3.0,
            )
        )
        try:
            initial = await environment.reset(412399380)
            combat = await environment.step(
                initial.episode_id,
                SemanticActionRequest(
                    type="semantic",
                    action=SemanticAction.MOVE_RIGHT,
                ),
            )

            assert len(combat.messages) == 2
            assert "hobgoblin" in combat.messages[0].lower()
            assert "hobgoblin" in combat.messages[1].lower()
            assert "--More--" not in combat.messages[0]
        finally:
            await environment.close()


@pytest.mark.skipif(not ROGUE_BINARY, reason="TEST_ROGUE_BINARY is not set")
async def test_scripted_inventory_selection_rest_and_movement() -> None:
    from rogue_api.models import KeyActionRequest

    with tempfile.TemporaryDirectory(prefix="rogue-", dir="/tmp") as data_dir:
        environment = RogueEnv(Settings(
            rogue_binary=Path(ROGUE_BINARY), data_dir=Path(data_dir), quiet_window_seconds=0.08,
        ))
        try:
            runs = []
            for _ in range(2):
                initial = await environment.reset(12345)
                observations = [initial]
                for key in ("i", "e", "a", ".", "h", "h"):
                    observations.append(await environment.step(
                        initial.episode_id, KeyActionRequest(type="key", key=key),
                    ))
                runs.append(observations)
            initial, inventory, selection, eaten, rested, _, moved = runs[0]
            assert any("a) Some food" in message for message in inventory.messages)
            assert selection.mode == "select_item"
            assert any("tasted good" in message.lower() for message in eaten.messages)
            assert rested.state.hp == eaten.state.hp == initial.state.max_hp
            assert moved.screen != rested.screen
            assert [observation.api_step for observation in runs[0]] == list(range(7))
            assert [observation.screen for observation in runs[0]] == [observation.screen for observation in runs[1]]
        finally:
            await environment.close()


@pytest.mark.skipif(not ROGUE_BINARY, reason="TEST_ROGUE_BINARY is not set")
async def test_scripted_stair_descent_reaches_second_level() -> None:
    from rogue_api.models import KeyActionRequest

    with tempfile.TemporaryDirectory(prefix="rogue-", dir="/tmp") as data_dir:
        environment = RogueEnv(Settings(
            rogue_binary=Path(ROGUE_BINARY), data_dir=Path(data_dir), quiet_window_seconds=0.08,
        ))
        try:
            initial = await environment.reset(3)
            assert initial.state.dungeon_level == 1
            for key in "hkk>":
                current = await environment.step(initial.episode_id, KeyActionRequest(type="key", key=key))
            assert current.state.dungeon_level == 2
            assert current.api_step == 4
            assert current.episode_id == initial.episode_id
        finally:
            await environment.close()
