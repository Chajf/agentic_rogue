"""Public API request and response models."""

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class GameStatus(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    DEAD = "dead"
    WON = "won"
    QUIT = "quit"
    ERROR = "error"


class InteractionMode(StrEnum):
    NORMAL = "normal"
    MORE = "more"
    SELECT_ITEM = "select_item"
    SELECT_DIRECTION = "select_direction"
    SELECT_HAND = "select_hand"
    CONFIRM = "confirm"
    INVENTORY = "inventory"
    GAME_OVER = "game_over"
    UNKNOWN = "unknown"


class SemanticAction(StrEnum):
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_UP_LEFT = "MOVE_UP_LEFT"
    MOVE_UP_RIGHT = "MOVE_UP_RIGHT"
    MOVE_DOWN_LEFT = "MOVE_DOWN_LEFT"
    MOVE_DOWN_RIGHT = "MOVE_DOWN_RIGHT"
    REST = "REST"
    SEARCH = "SEARCH"
    ASCEND = "ASCEND"
    DESCEND = "DESCEND"
    PICK_UP = "PICK_UP"
    INVENTORY = "INVENTORY"
    EAT = "EAT"
    QUAFF = "QUAFF"
    READ = "READ"
    WIELD = "WIELD"
    WEAR = "WEAR"
    TAKE_OFF_ARMOR = "TAKE_OFF_ARMOR"
    PUT_ON_RING = "PUT_ON_RING"
    REMOVE_RING = "REMOVE_RING"
    DROP = "DROP"
    THROW = "THROW"
    ZAP = "ZAP"
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    QUIT = "QUIT"


class Direction(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


class Hand(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class ResetRequest(BaseModel):
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class SemanticActionRequest(BaseModel):
    type: Literal["semantic"]
    action: SemanticAction
    item: str | None = None
    direction: Direction | None = None
    hand: Hand | None = None

    @field_validator("item")
    @classmethod
    def validate_item(cls, value: str | None) -> str | None:
        if value is not None and (len(value) != 1 or not value.isascii() or not value.isalpha()):
            raise ValueError("item must be one ASCII letter")
        return value


class KeyActionRequest(BaseModel):
    type: Literal["key"]
    key: str


Action = Annotated[SemanticActionRequest | KeyActionRequest, Field(discriminator="type")]


class SemanticActionEnvelope(SemanticActionRequest):
    episode_id: UUID


class KeyActionEnvelope(KeyActionRequest):
    episode_id: UUID


ActionEnvelope = Annotated[SemanticActionEnvelope | KeyActionEnvelope, Field(discriminator="type")]


class Cursor(BaseModel):
    row: int
    column: int


class RogueState(BaseModel):
    dungeon_level: int | None = None
    gold: int | None = None
    hp: int | None = None
    max_hp: int | None = None
    strength: int | None = None
    max_strength: int | None = None
    armor: int | None = None
    experience_level: int | None = None
    experience: int | None = None
    hunger: str | None = None


class Observation(BaseModel):
    episode_id: UUID
    seed: int
    api_step: int
    status: GameStatus
    mode: InteractionMode
    message: str
    screen: list[str]
    cursor: Cursor
    state: RogueState
    raw_output: str
    terminated: bool
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    game: Literal["not_started", "running", "stopped", "failed"]
