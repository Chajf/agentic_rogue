import os
from pathlib import Path
import tempfile

import pytest

from rogue_api.config import Settings
from rogue_api.environment import RogueEnv
from rogue_api.models import SemanticAction, SemanticActionRequest


ROGUE_BINARY = os.environ.get("TEST_ROGUE_BINARY")


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
