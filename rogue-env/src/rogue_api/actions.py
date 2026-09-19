"""Translate API actions into prompt-aware Rogue input segments."""

from dataclasses import dataclass

from .models import (
    Direction,
    Hand,
    InteractionMode,
    KeyActionRequest,
    SemanticAction,
    SemanticActionRequest,
)


class InvalidAction(ValueError):
    pass


DIRECTION_KEYS = {
    Direction.UP: "k",
    Direction.DOWN: "j",
    Direction.LEFT: "h",
    Direction.RIGHT: "l",
    Direction.UP_LEFT: "y",
    Direction.UP_RIGHT: "u",
    Direction.DOWN_LEFT: "b",
    Direction.DOWN_RIGHT: "n",
}

MOVEMENT_KEYS = {
    SemanticAction.MOVE_UP: "k",
    SemanticAction.MOVE_DOWN: "j",
    SemanticAction.MOVE_LEFT: "h",
    SemanticAction.MOVE_RIGHT: "l",
    SemanticAction.MOVE_UP_LEFT: "y",
    SemanticAction.MOVE_UP_RIGHT: "u",
    SemanticAction.MOVE_DOWN_LEFT: "b",
    SemanticAction.MOVE_DOWN_RIGHT: "n",
}

SIMPLE_KEYS = {
    SemanticAction.REST: ".",
    SemanticAction.SEARCH: "s",
    SemanticAction.ASCEND: "<",
    SemanticAction.DESCEND: ">",
    SemanticAction.PICK_UP: ",",
    SemanticAction.INVENTORY: "i",
    SemanticAction.TAKE_OFF_ARMOR: "T",
    SemanticAction.CONFIRM: "y",
    SemanticAction.CANCEL: "\x1b",
    SemanticAction.QUIT: "Q",
}

ITEM_COMMANDS = {
    SemanticAction.EAT: "e",
    SemanticAction.QUAFF: "q",
    SemanticAction.READ: "r",
    SemanticAction.WIELD: "w",
    SemanticAction.WEAR: "W",
    SemanticAction.DROP: "d",
}

NAMED_KEYS = {"ESCAPE": "\x1b", "ENTER": "\n", "SPACE": " "}
DISALLOWED_RAW_KEYS = {"!"}


@dataclass(frozen=True)
class InputSegment:
    data: bytes
    required_mode: InteractionMode | None = None


@dataclass(frozen=True)
class EncodedAction:
    segments: tuple[InputSegment, ...]

    @property
    def data(self) -> bytes:
        return b"".join(segment.data for segment in self.segments)


def _plan(*segments: tuple[str, InteractionMode | None]) -> EncodedAction:
    return EncodedAction(tuple(InputSegment(data.encode(), mode) for data, mode in segments))


def _item(request: SemanticActionRequest) -> str:
    if request.item is None:
        raise InvalidAction(f"{request.action} requires item")
    return request.item.lower()


def _direction(request: SemanticActionRequest) -> str:
    if request.direction is None:
        raise InvalidAction(f"{request.action} requires direction")
    return DIRECTION_KEYS[request.direction]


def encode_semantic(request: SemanticActionRequest) -> EncodedAction:
    action = request.action
    if action in MOVEMENT_KEYS:
        return _plan((MOVEMENT_KEYS[action], None))
    if action in SIMPLE_KEYS:
        return _plan((SIMPLE_KEYS[action], None))
    if action in ITEM_COMMANDS:
        return _plan(
            (ITEM_COMMANDS[action], None),
            (_item(request), InteractionMode.SELECT_ITEM),
        )
    if action is SemanticAction.THROW:
        return _plan(
            ("t", None),
            (_direction(request), InteractionMode.SELECT_DIRECTION),
            (_item(request), InteractionMode.SELECT_ITEM),
        )
    if action is SemanticAction.ZAP:
        return _plan(
            ("z", None),
            (_direction(request), InteractionMode.SELECT_DIRECTION),
            (_item(request), InteractionMode.SELECT_ITEM),
        )
    if action is SemanticAction.PUT_ON_RING:
        segments = [("P", None), (_item(request), InteractionMode.SELECT_ITEM)]
        if request.hand is not None:
            segments.append(("l" if request.hand is Hand.LEFT else "r", InteractionMode.SELECT_HAND))
        return _plan(*segments)
    if action is SemanticAction.REMOVE_RING:
        if request.hand is None:
            return _plan(("R", None))
        return _plan(
            ("R", None),
            ("l" if request.hand is Hand.LEFT else "r", InteractionMode.SELECT_HAND),
        )
    raise InvalidAction(f"unsupported semantic action: {action}")


def encode_key(request: KeyActionRequest) -> EncodedAction:
    value = NAMED_KEYS.get(request.key.upper())
    if value is None:
        if len(request.key) != 1 or not request.key.isascii() or not request.key.isprintable():
            raise InvalidAction("key must be one printable ASCII character or ESCAPE, ENTER, or SPACE")
        value = request.key
    if value in DISALLOWED_RAW_KEYS:
        raise InvalidAction("the Rogue shell escape key is not allowed")
    return _plan((value, None))


def encode_action(request: SemanticActionRequest | KeyActionRequest) -> EncodedAction:
    if isinstance(request, SemanticActionRequest):
        return encode_semantic(request)
    return encode_key(request)
