import asyncio
import errno
import fcntl
import os
from pathlib import Path
import select
import struct
import subprocess
import termios
import time


class RogueProcessError(RuntimeError):
    pass


class RogueProcess:
    def __init__(
        self,
        binary: Path,
        data_dir: Path,
        rows: int,
        columns: int,
        quiet_window: float,
        timeout: float,
        player_name: str,
        rogue_options: str,
    ) -> None:
        self.binary = binary
        self.data_dir = data_dir
        self.rows = rows
        self.columns = columns
        self.quiet_window = quiet_window
        self.timeout = timeout
        self.player_name = player_name
        self.rogue_options = rogue_options
        self.process: subprocess.Popen[bytes] | None = None
        self.fd: int | None = None

    @property
    def running(self) -> bool:
        if self.process is None:
            return False
        if self.process.poll() is None:
            return True
        self.process = None
        self._close_fd()
        return False

    async def start(self, seed: int) -> bytes:
        if self.running:
            raise RogueProcessError("Rogue is already running")
        if not self.binary.is_file():
            raise RogueProcessError(f"Rogue binary not found: {self.binary}")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        master_fd, slave_fd = os.openpty()
        window_size = struct.pack("HHHH", self.rows, self.columns, 0, 0)
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, window_size)
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(self.data_dir),
                "TERM": "xterm",
                "ROGOSEED": str(seed),
                "ROGUEOPTS": self._runtime_options(),
            }
        )
        try:
            self.process = subprocess.Popen(
                [str(self.binary)],
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=self.data_dir,
                env=env,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            os.close(slave_fd)

        self.fd = master_fd
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        output = await self.read_until_quiet()
        if not self.running:
            raise RogueProcessError(f"Rogue exited during startup: {output.decode(errors='replace')}")
        return output

    def _runtime_options(self) -> str:
        paths = {
            "file": self.data_dir / ".rogue.save",
            "score": self.data_dir / ".rogue.scr",
            "lock": self.data_dir / ".rogue.lck",
        }
        path_options = ",".join(f"{name}={path}" for name, path in paths.items())
        return f"{self.rogue_options},name={self.player_name},{path_options}"

    async def exchange(self, data: bytes) -> bytes:
        if not self.running or self.fd is None:
            raise RogueProcessError("Rogue is not running")
        try:
            os.write(self.fd, data)
        except OSError as exc:
            raise RogueProcessError(f"failed to write to Rogue PTY: {exc}") from exc
        return await self.read_until_quiet()

    async def read_until_quiet(self) -> bytes:
        return await asyncio.to_thread(self._read_until_quiet_sync)

    def _read_until_quiet_sync(self) -> bytes:
        if self.fd is None:
            return b""
        chunks: list[bytes] = []
        started = time.monotonic()
        last_output = started
        while True:
            now = time.monotonic()
            if now - started >= self.timeout:
                raise TimeoutError(f"Rogue output did not settle within {self.timeout} seconds")
            wait = min(self.quiet_window - (now - last_output), self.timeout - (now - started))
            if wait <= 0:
                return b"".join(chunks)
            readable, _, _ = select.select([self.fd], [], [], wait)
            if not readable:
                return b"".join(chunks)
            while True:
                try:
                    data = os.read(self.fd, 65536)
                except BlockingIOError:
                    break
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        return b"".join(chunks)
                    raise
                if not data:
                    return b"".join(chunks)
                chunks.append(data)
                last_output = time.monotonic()

    async def close(self) -> None:
        process = self.process
        if process is None:
            self._close_fd()
            return
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 0.2)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait)
        self.process = None
        self._close_fd()

    def _close_fd(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
