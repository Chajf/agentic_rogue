"""Validated runtime settings for one game session."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    rogue_api_url: str
    rogue_timeout_seconds: float
    model_base_url: str
    model_name: str
    model_api_key: str
    model_timeout_seconds: float
    game_seed: int | None
    max_game_actions: int
    max_model_calls_per_action: int
    context_token_budget: int

    @classmethod
    def from_env(cls) -> "Settings":
        seed = os.getenv("GAME_SEED", "").strip()
        settings = cls(
            rogue_api_url=os.getenv("ROGUE_API_URL", "http://localhost:8000"),
            rogue_timeout_seconds=float(os.getenv("ROGUE_TIMEOUT_SECONDS", "10")),
            model_base_url=os.environ["MODEL_BASE_URL"],
            model_name=os.environ["MODEL_NAME"],
            model_api_key=os.getenv("MODEL_API_KEY", "local"),
            model_timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "120")),
            game_seed=int(seed) if seed else None,
            max_game_actions=int(os.getenv("MAX_GAME_ACTIONS", "10000")),
            max_model_calls_per_action=int(os.getenv("MAX_MODEL_CALLS_PER_ACTION", "3")),
            context_token_budget=int(os.getenv("CONTEXT_TOKEN_BUDGET", "8192")),
        )
        if settings.game_seed is not None and not 0 <= settings.game_seed <= 2**32 - 1:
            raise ValueError("GAME_SEED must be between 0 and 2^32 - 1")
        if min(
            settings.rogue_timeout_seconds, settings.model_timeout_seconds,
            settings.max_game_actions, settings.max_model_calls_per_action,
            settings.context_token_budget,
        ) <= 0:
            raise ValueError("timeouts and limits must be positive")
        return settings
