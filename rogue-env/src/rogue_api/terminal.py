"""Terminal reconstruction and observation parsing."""

import re

import pyte

from .models import Cursor, GameStatus, InteractionMode, RogueState


STATUS_RE = re.compile(
    r"Level:\s*(?P<level>\d+)\s+Gold:\s*(?P<gold>\d+)\s+"
    r"Hp:\s*(?P<hp>-?\d+)\(\s*(?P<max_hp>\d+)\)\s+"
    r"Str:\s*(?P<strength>\d+)\((?P<max_strength>\d+)\)\s+"
    r"Arm:\s*(?P<armor>-?\d+)\s+Exp:\s*(?P<experience_level>\d+)/(?P<experience>\d+)"
    r"(?:\s+(?P<hunger>\S+))?",
    re.IGNORECASE,
)


class TerminalObserver:
    def __init__(self, columns: int = 80, rows: int = 24) -> None:
        self.columns = columns
        self.rows = rows
        self.screen = pyte.Screen(columns, rows)
        self.stream = pyte.Stream(self.screen)

    def feed(self, output: bytes) -> None:
        self.stream.feed(output.decode("utf-8", errors="replace"))

    def display(self) -> list[str]:
        return [line[: self.columns].ljust(self.columns) for line in self.screen.display[: self.rows]]

    def cursor(self) -> Cursor:
        return Cursor(row=self.screen.cursor.y, column=self.screen.cursor.x)

    def message(self) -> str:
        return self.display()[0].rstrip()

    def state(self) -> RogueState:
        for line in reversed(self.display()):
            match = STATUS_RE.search(line)
            if match:
                values = match.groupdict()
                return RogueState(
                    dungeon_level=int(values["level"]),
                    gold=int(values["gold"]),
                    hp=int(values["hp"]),
                    max_hp=int(values["max_hp"]),
                    strength=int(values["strength"]),
                    max_strength=int(values["max_strength"]),
                    armor=int(values["armor"]),
                    experience_level=int(values["experience_level"]),
                    experience=int(values["experience"]),
                    hunger=values.get("hunger"),
                )
        return RogueState()

    def mode(self) -> InteractionMode:
        text = "\n".join(line.rstrip() for line in self.display())
        lowered = text.lower()
        if "--more--" in lowered:
            return InteractionMode.MORE
        if any(prompt in lowered for prompt in ("which object", "which item", "(* for list)", "item: ")):
            return InteractionMode.SELECT_ITEM
        if any(prompt in lowered for prompt in ("which direction", "what direction")):
            return InteractionMode.SELECT_DIRECTION
        if any(prompt in lowered for prompt in ("left hand or right hand", "left or right ring")):
            return InteractionMode.SELECT_HAND
        if any(prompt in lowered for prompt in ("really quit?", "are you sure", "[yn]", "(y/n)")):
            return InteractionMode.CONFIRM
        if self.status() in {GameStatus.DEAD, GameStatus.WON, GameStatus.QUIT}:
            return InteractionMode.GAME_OVER
        return InteractionMode.NORMAL

    def status(self) -> GameStatus:
        text = "\n".join(self.display()).lower()
        if "congratulations, you have made it to the light of day" in text:
            return GameStatus.WON
        if "rest in peace" in text or "killed by" in text:
            return GameStatus.DEAD
        if "really quit?" in text:
            return GameStatus.RUNNING
        if "top ten rogueists" in text or "quit" in text and "score" in text:
            return GameStatus.QUIT
        return GameStatus.RUNNING
