"""Validated runtime settings for one game session."""

import json
import os
from dataclasses import dataclass, field
from typing import Any


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
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    history_recent_interactions: int = 8
    context_policy: str = "history_only"
    summary_context_token_budget: int = 8192
    summary_output_tokens: int = 1024
    max_summary_attempts: int = 2

    def __post_init__(self) -> None:
        if self.context_policy not in {"history_only", "working_memory"}:
            raise ValueError("CONTEXT_POLICY must be history_only or working_memory")
        if self.summary_output_tokens <= 0 or self.summary_context_token_budget <= self.summary_output_tokens:
            raise ValueError("SUMMARY_CONTEXT_TOKEN_BUDGET must exceed positive SUMMARY_OUTPUT_TOKENS")
        if self.max_summary_attempts <= 0:
            raise ValueError("MAX_SUMMARY_ATTEMPTS must be positive")
        _ = self.actor_output_tokens

    @property
    def actor_output_tokens(self) -> int:
        caps = [self.model_kwargs[key] for key in ("max_tokens", "max_completion_tokens") if key in self.model_kwargs]
        if len(caps) > 1 or (caps and (type(caps[0]) is not int or caps[0] <= 0)):
            raise ValueError("MODEL_KWARGS requires a single positive integer output cap")
        return caps[0] if caps else 512

    @classmethod
    def from_env(cls) -> "Settings":
        seed = os.getenv("GAME_SEED", "").strip()
        try:
            model_kwargs = json.loads(os.getenv("MODEL_KWARGS", "").strip() or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("MODEL_KWARGS must be a valid JSON object") from exc
        if not isinstance(model_kwargs, dict):
            raise ValueError("MODEL_KWARGS must be a JSON object")
        if "model" in model_kwargs:
            raise ValueError("Use MODEL_NAME to set the model, not MODEL_KWARGS")
        try:
            history_recent_interactions = int(os.getenv("HISTORY_RECENT_INTERACTIONS", "8"))
        except ValueError as exc:
            raise ValueError("HISTORY_RECENT_INTERACTIONS must be a non-negative integer") from exc
        if history_recent_interactions < 0:
            raise ValueError("HISTORY_RECENT_INTERACTIONS must be a non-negative integer")
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
            model_kwargs=model_kwargs,
            history_recent_interactions=history_recent_interactions,
            context_policy=os.getenv("CONTEXT_POLICY", "history_only"),
            summary_context_token_budget=int(os.getenv("SUMMARY_CONTEXT_TOKEN_BUDGET", "8192")),
            summary_output_tokens=int(os.getenv("SUMMARY_OUTPUT_TOKENS", "1024")),
            max_summary_attempts=int(os.getenv("MAX_SUMMARY_ATTEMPTS", "2")),
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
