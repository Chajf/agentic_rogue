"""Typed HTTP client for the existing Rogue environment API."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

import httpx
from pydantic import BaseModel, field_validator, model_validator


class Direction(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP_LEFT = "UP_LEFT"
    UP_RIGHT = "UP_RIGHT"
    DOWN_LEFT = "DOWN_LEFT"
    DOWN_RIGHT = "DOWN_RIGHT"


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


class ActionRequest(BaseModel):
    type: Literal["semantic"] = "semantic"
    action: SemanticAction
    item: str | None = None
    direction: Direction | None = None
    hand: Literal["LEFT", "RIGHT"] | None = None

    @field_validator("item")
    @classmethod
    def valid_item(cls, value: str | None) -> str | None:
        if value is not None and (len(value) != 1 or not value.isascii() or not value.isalpha()):
            raise ValueError("item must be one ASCII letter")
        return value

    @model_validator(mode="after")
    def required_parameters(self) -> "ActionRequest":
        if self.action in {
            SemanticAction.EAT, SemanticAction.QUAFF, SemanticAction.READ,
            SemanticAction.WIELD, SemanticAction.WEAR, SemanticAction.DROP,
            SemanticAction.PUT_ON_RING, SemanticAction.THROW, SemanticAction.ZAP,
        } and self.item is None:
            raise ValueError(f"{self.action} requires item")
        if self.action in {SemanticAction.THROW, SemanticAction.ZAP} and self.direction is None:
            raise ValueError(f"{self.action} requires direction")
        return self


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


class Cursor(BaseModel):
    row: int
    column: int


class Observation(BaseModel):
    episode_id: UUID
    seed: int
    api_step: int
    status: Literal["starting", "running", "dead", "won", "quit", "error"]
    mode: Literal[
        "normal", "more", "select_item", "select_direction", "select_hand",
        "confirm", "inventory", "game_over", "unknown",
    ]
    messages: list[str]
    screen: list[str]
    cursor: Cursor
    state: RogueState
    raw_output: str
    terminated: bool
    error: str | None = None


class RogueAPIError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Rogue API returned {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class RogueTransportError(RuntimeError):
    """The request may have reached the game; inspect state before retrying."""


class RogueClient:
    def __init__(self, base_url: str, timeout: float = 10, *, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self._owns_client = client is None

    async def __aenter__(self) -> "RogueClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def reset(self, seed: int | None = None) -> Observation:
        return await self._request("POST", "/game/reset", json={"seed": seed})

    async def observe(self) -> Observation:
        return await self._request("GET", "/game/state")

    async def action(self, episode_id: UUID, action: ActionRequest) -> Observation:
        return await self._request(
            "POST", "/game/action",
            json={"episode_id": str(episode_id), **action.model_dump(mode="json", exclude_none=True)},
        )

    async def _request(self, method: str, path: str, **kwargs: object) -> Observation:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise RogueTransportError(str(exc)) from exc
        if response.is_error:
            try:
                detail = str(response.json().get("detail", response.text))
            except ValueError:
                detail = response.text
            raise RogueAPIError(response.status_code, detail)
        return Observation.model_validate(response.json())
