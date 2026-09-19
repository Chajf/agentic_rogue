import pytest

from rogue_api.actions import InvalidAction, encode_action
from rogue_api.models import Direction, InteractionMode, KeyActionRequest, SemanticAction, SemanticActionRequest


def semantic(action: SemanticAction, **kwargs: object) -> SemanticActionRequest:
    return SemanticActionRequest(type="semantic", action=action, **kwargs)


@pytest.mark.parametrize(
    ("action_request", "expected"),
    [
        (semantic(SemanticAction.MOVE_UP), b"k"),
        (semantic(SemanticAction.REST), b"."),
        (semantic(SemanticAction.QUAFF, item="B"), b"qb"),
        (semantic(SemanticAction.THROW, item="a", direction=Direction.UP_RIGHT), b"tua"),
        (semantic(SemanticAction.ZAP, item="c", direction=Direction.LEFT), b"zhc"),
    ],
)
def test_encodes_semantic_actions(action_request: SemanticActionRequest, expected: bytes) -> None:
    assert encode_action(action_request).data == expected


def test_requires_action_parameters() -> None:
    with pytest.raises(InvalidAction, match="requires item"):
        encode_action(semantic(SemanticAction.EAT))


def test_blocks_shell_escape() -> None:
    with pytest.raises(InvalidAction, match="shell escape"):
        encode_action(KeyActionRequest(type="key", key="!"))


def test_allows_named_escape() -> None:
    assert encode_action(KeyActionRequest(type="key", key="ESCAPE")).data == b"\x1b"


def test_item_action_is_staged_by_prompt() -> None:
    encoded = encode_action(semantic(SemanticAction.QUAFF, item="b"))

    assert encoded.segments[0].required_mode is None
    assert encoded.segments[1].required_mode is InteractionMode.SELECT_ITEM
