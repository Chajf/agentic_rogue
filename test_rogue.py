import os
import pty
import select
import time

pid, fd = pty.fork()

if pid == 0:
    # Proces potomny
    os.execv("../rogue5.4/rogue", ["./rogue"])

# Proces nadrzędny — kontroler Rogue
time.sleep(1)

while True:
    readable, _, _ = select.select([fd], [], [], 0.2)

    if fd in readable:
        try:
            data = os.read(fd, 4096)
        except OSError:
            break

        if not data:
            break

        print(repr(data))

    # na początek wykonaj jeden ruch w prawo
    os.write(fd, b"l")

    time.sleep(1)
    break