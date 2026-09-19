from rogue_api.models import InteractionMode
from rogue_api.terminal import TerminalObserver


def test_parses_status_line() -> None:
    terminal = TerminalObserver()
    terminal.feed(
        b"\x1b[24;1HLevel: 3  Gold: 42     Hp: 11(12)  Str: 16(16)  Arm: 4   Exp: 2/17  Hungry"
    )

    state = terminal.state()

    assert state.dungeon_level == 3
    assert state.gold == 42
    assert state.hp == 11
    assert state.max_hp == 12
    assert state.experience_level == 2
    assert state.experience == 17
    assert state.hunger == "Hungry"


def test_detects_more_prompt() -> None:
    terminal = TerminalObserver()
    terminal.feed(b"A long message --More--")

    assert terminal.mode() is InteractionMode.MORE


def test_detects_hand_prompt() -> None:
    terminal = TerminalObserver()
    terminal.feed(b"left hand or right hand? ")

    assert terminal.mode() is InteractionMode.SELECT_HAND


def test_screen_is_fixed_size() -> None:
    terminal = TerminalObserver()
    terminal.feed(b"hello")

    assert len(terminal.display()) == 24
    assert all(len(row) == 80 for row in terminal.display())


def test_expands_linux_ncurses_repeat_character_sequence() -> None:
    terminal = TerminalObserver()
    terminal.feed(
        b"\x1b[10;40H-----+------"
        b"\x1b[11;40H+..*.\x1b[6b+"
        b"\x1b[12;40H|.\x1b[9b|"
        b"\x1b[13;40H|.\x1b[7b@H|"
    )

    assert terminal.display()[10][39:51] == "+..*.......+"
    assert terminal.display()[11][39:51] == "|..........|"
    assert terminal.display()[12][39:51] == "|........@H|"
