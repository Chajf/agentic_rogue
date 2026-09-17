import os
import pty
import select
import sys
import termios
import tty

pid, fd = pty.fork()

if pid == 0:
    os.execv("../rogue5.4/rogue", ["./rogue"])

stdin_fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(stdin_fd)

try:
    tty.setraw(stdin_fd)

    while True:
        readable, _, _ = select.select([fd, stdin_fd], [], [])

        if fd in readable:
            try:
                data = os.read(fd, 4096)
            except OSError:
                break

            if not data:
                break

            # pokaż output Rogue dokładnie w naszym terminalu
            os.write(sys.stdout.fileno(), data)

        if stdin_fd in readable:
            key = os.read(stdin_fd, 1)

            if key == b"\x1d":  # Ctrl+]
                break

            os.write(fd, key)

finally:
    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)