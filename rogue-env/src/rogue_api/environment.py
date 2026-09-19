"""Episode lifecycle and action orchestration."""

import asyncio
from pathlib import Path
import secrets
from uuid import UUID, uuid4

from .actions import EncodedAction, InputSegment, InvalidAction, encode_action
from .config import Settings
from .models import (
    Action,
    GameStatus,
    InteractionMode,
    Observation,
    SemanticAction,
    SemanticActionRequest,
)
from .process import RogueProcess, RogueProcessError
from .terminal import TerminalObserver


class NoActiveEpisode(RuntimeError):
    pass


class StaleEpisode(RuntimeError):
    pass


class IncompatibleMode(RuntimeError):
    pass


class RogueEnv:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.process: RogueProcess | None = None
        self.terminal: TerminalObserver | None = None
        self.episode_id: UUID | None = None
        self.seed: int | None = None
        self.api_step = 0
        self.error: str | None = None
        self.messages: list[str] = []
        self._lock = asyncio.Lock()

    @property
    def has_episode(self) -> bool:
        return self.episode_id is not None

    @property
    def process_running(self) -> bool:
        return self.process is not None and self.process.running

    async def reset(self, seed: int | None = None) -> Observation:
        async with self._lock:
            await self._close_unlocked()
            self._remove_stale_runtime_files()
            self.seed = seed if seed is not None else secrets.randbits(32)
            self.episode_id = uuid4()
            self.api_step = 0
            self.error = None
            self.messages = []
            self.terminal = TerminalObserver(
                self.settings.terminal_columns,
                self.settings.terminal_rows,
            )
            self.process = self._new_process()
            try:
                output = await self.process.start(self.seed)
                self.terminal.feed(output)
                output, page_messages = await self._consume_more(output)
                final_message = self.terminal.message()
                if final_message:
                    page_messages.append(final_message)
                self.messages = self._deduplicate_adjacent(page_messages)
                return self._observation(output)
            except Exception as exc:
                self.error = str(exc)
                if self.process is not None:
                    await self.process.close()
                return self._observation(b"", forced_status=GameStatus.ERROR)

    async def observe(self) -> Observation:
        async with self._lock:
            self._require_episode()
            return self._observation(b"")

    async def step(self, episode_id: UUID, action: Action) -> Observation:
        async with self._lock:
            self._require_episode()
            if episode_id != self.episode_id:
                raise StaleEpisode("episode_id does not match the active episode")
            if self.process is None or self.terminal is None or not self.process.running:
                self.error = "Rogue process is not running"
                return self._observation(b"", forced_status=GameStatus.ERROR)
            self._validate_mode(action)
            encoded = self._encode_for_current_mode(action)
            self.api_step += 1
            try:
                chunks: list[bytes] = []
                page_messages: list[str] = []
                message_before = self.terminal.message()
                for segment in encoded.segments:
                    if (
                        segment.required_mode is not None
                        and self.terminal.mode() is not segment.required_mode
                    ):
                        break
                    output = await self.process.exchange(segment.data)
                    self.terminal.feed(output)
                    consumed, consumed_messages = await self._consume_more(output)
                    chunks.append(consumed)
                    page_messages.extend(consumed_messages)
                output = b"".join(chunks)
            except (RogueProcessError, TimeoutError) as exc:
                self.error = str(exc)
                return self._observation(b"", forced_status=GameStatus.ERROR)
            final_message = self.terminal.message()
            if final_message and (final_message != message_before or page_messages):
                page_messages.append(final_message)
            self.messages = self._deduplicate_adjacent(page_messages)
            return self._observation(output)

    async def close(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        if self.process is not None:
            await self.process.close()
        self.process = None
        self.terminal = None
        self.episode_id = None
        self.seed = None
        self.api_step = 0
        self.error = None
        self.messages = []

    async def _consume_more(self, initial: bytes) -> tuple[bytes, list[str]]:
        assert self.process is not None
        assert self.terminal is not None
        chunks = [initial]
        messages: list[str] = []
        for _ in range(100):
            if self.terminal.mode() is not InteractionMode.MORE:
                return b"".join(chunks), messages
            message = self.terminal.message().removesuffix(" --More--").rstrip()
            if message:
                messages.append(message)
            output = await self.process.exchange(b" ")
            chunks.append(output)
            self.terminal.feed(output)
        raise RogueProcessError("too many consecutive --More-- pages")

    def _observation(
        self,
        raw_output: bytes,
        forced_status: GameStatus | None = None,
    ) -> Observation:
        self._require_episode()
        assert self.terminal is not None
        assert self.seed is not None
        status = forced_status or self.terminal.status()
        if forced_status is None and self.process is not None and not self.process.running:
            status = (
                status
                if status in {GameStatus.DEAD, GameStatus.WON, GameStatus.QUIT}
                else GameStatus.ERROR
            )
        return Observation(
            episode_id=self.episode_id,
            seed=self.seed,
            api_step=self.api_step,
            status=status,
            mode=(
                InteractionMode.GAME_OVER
                if status
                in {
                    GameStatus.DEAD,
                    GameStatus.WON,
                    GameStatus.QUIT,
                    GameStatus.ERROR,
                }
                else self.terminal.mode()
            ),
            messages=self.messages,
            screen=self.terminal.display(),
            cursor=self.terminal.cursor(),
            state=self.terminal.state(),
            raw_output=raw_output.decode("utf-8", errors="replace"),
            terminated=status in {GameStatus.DEAD, GameStatus.WON, GameStatus.QUIT, GameStatus.ERROR},
            error=self.error,
        )

    @staticmethod
    def _deduplicate_adjacent(messages: list[str]) -> list[str]:
        result: list[str] = []
        for message in messages:
            if not result or result[-1] != message:
                result.append(message)
        return result

    def _require_episode(self) -> None:
        if self.episode_id is None or self.terminal is None:
            raise NoActiveEpisode("no active Rogue episode")

    def _new_process(self) -> RogueProcess:
        return RogueProcess(
            binary=self.settings.rogue_binary,
            data_dir=self.settings.data_dir,
            rows=self.settings.terminal_rows,
            columns=self.settings.terminal_columns,
            quiet_window=self.settings.quiet_window_seconds,
            timeout=self.settings.action_timeout_seconds,
            player_name=self.settings.player_name,
            rogue_options=self.settings.rogue_options,
        )

    def _validate_mode(self, action: Action) -> None:
        assert self.terminal is not None
        mode = self.terminal.mode()
        if mode is InteractionMode.CONFIRM:
            allowed = isinstance(action, SemanticActionRequest) and action.action in {
                SemanticAction.CONFIRM,
                SemanticAction.CANCEL,
            }
            if not allowed:
                raise IncompatibleMode("the game is awaiting CONFIRM or CANCEL")
        elif mode in {
            InteractionMode.SELECT_ITEM,
            InteractionMode.SELECT_DIRECTION,
            InteractionMode.SELECT_HAND,
        }:
            allowed = not isinstance(action, SemanticActionRequest) or action.action is SemanticAction.CANCEL
            if not allowed:
                raise IncompatibleMode(f"the game is awaiting raw input for mode {mode}")

    def _encode_for_current_mode(self, action: Action) -> EncodedAction:
        assert self.terminal is not None
        if (
            self.terminal.mode() is InteractionMode.CONFIRM
            and isinstance(action, SemanticActionRequest)
            and action.action is SemanticAction.CANCEL
        ):
            return EncodedAction((InputSegment(b"n"),))
        return encode_action(action)

    def _remove_stale_runtime_files(self) -> None:
        for name in (".rogue.save", ".rogue.lck"):
            path = Path(self.settings.data_dir, name)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
