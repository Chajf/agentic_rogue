"""Validated runtime settings for one game session."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from game_runner.prompts.system import ACTOR_PROMPTS, PromptProfile, actor_prompt_version

ContextVariant = Literal["baseline", "neighbors"]
StructuredStateProfile = Literal["off", "map-v1"]
HistoryFormat = Literal["replay", "compact"]
GoalManagement = Literal["off", "managed-v1"]
ActorType = Literal["chat", "system_one"]


class GoalReviewThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    same_edge_returns: int = Field(default=2, strict=True, gt=0)
    movement_stall_actions: int = Field(default=6, strict=True, gt=0)


def validate_goal_management(mode: GoalManagement, structured_state: StructuredStateProfile) -> None:
    if mode not in {"off", "managed-v1"}:
        raise ValueError("GOAL_MANAGEMENT must be off or managed-v1")
    if mode == "managed-v1" and structured_state != "map-v1":
        raise ValueError("managed-v1 requires map-v1 structured state")


def validate_structured_state(
    profile: StructuredStateProfile, prompt_profile: PromptProfile, context_policy: str, context_variant: ContextVariant,
) -> None:
    if profile not in {"off", "map-v1"}:
        raise ValueError("STRUCTURED_STATE must be off or map-v1")
    if profile == "map-v1" and (
        prompt_profile != "rules-v2" or context_policy != "history_only" or context_variant != "baseline"
    ):
        raise ValueError("map-v1 requires rules-v2, history_only, and the baseline context variant")


def validate_history_format(history_format: HistoryFormat, structured_state: StructuredStateProfile) -> None:
    if history_format not in {"replay", "compact"}:
        raise ValueError("HISTORY_FORMAT must be replay or compact")
    if history_format == "compact" and structured_state != "map-v1":
        raise ValueError("compact history requires map-v1 structured state")


def validate_actor_type(actor_type: ActorType, prompt: PromptProfile, policy: str, goals: GoalManagement) -> None:
    if actor_type not in {"chat", "system_one"}:
        raise ValueError("ACTOR_TYPE must be chat or system_one")
    if actor_type == "system_one" and (prompt, policy, goals) != ("rules-v2", "history_only", "off"):
        raise ValueError("System One requires rules-v2, history_only, and goals disabled")


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
    context_variant: ContextVariant = "baseline"
    prompt_profile: PromptProfile = "legacy"
    structured_state: StructuredStateProfile = "off"
    history_format: HistoryFormat = "replay"
    goal_management: GoalManagement = "off"
    goal_review: GoalReviewThresholds = field(default_factory=GoalReviewThresholds)
    actor_type: ActorType = "chat"

    def __post_init__(self) -> None:
        if self.prompt_profile not in ACTOR_PROMPTS:
            raise ValueError("PROMPT_PROFILE must be legacy or rules-v2")
        if self.context_policy not in {"history_only", "working_memory"}:
            raise ValueError("CONTEXT_POLICY must be history_only or working_memory")
        if self.context_variant != "baseline" and (
            self.context_variant != "neighbors" or self.context_policy != "history_only"
        ):
            raise ValueError("neighbors requires history_only")
        validate_structured_state(self.structured_state, self.prompt_profile, self.context_policy, self.context_variant)
        validate_history_format(self.history_format, self.structured_state)
        validate_goal_management(self.goal_management, self.structured_state)
        validate_actor_type(self.actor_type, self.prompt_profile, self.context_policy, self.goal_management)
        if self.summary_output_tokens <= 0 or self.summary_context_token_budget <= self.summary_output_tokens:
            raise ValueError("SUMMARY_CONTEXT_TOKEN_BUDGET must exceed positive SUMMARY_OUTPUT_TOKENS")
        if self.max_summary_attempts <= 0:
            raise ValueError("MAX_SUMMARY_ATTEMPTS must be positive")
        _ = self.actor_output_tokens

    @property
    def prompt_version(self) -> str:
        return actor_prompt_version(
            self.context_variant, prompt_profile=self.prompt_profile, system_one=self.actor_type == "system_one",
        )

    @property
    def actor_output_tokens(self) -> int:
        caps = [self.model_kwargs[key] for key in ("max_tokens", "max_completion_tokens") if key in self.model_kwargs]
        if len(caps) > 1 or (caps and (type(caps[0]) is not int or caps[0] <= 0)):
            raise ValueError("MODEL_KWARGS requires a single positive integer output cap")
        return caps[0] if caps else 512

    @classmethod
    def from_env(
        cls, *, prompt_profile: PromptProfile | None = None, structured_state: StructuredStateProfile | None = None,
        history_format: HistoryFormat | None = None, goal_management: GoalManagement | None = None,
        actor_type: ActorType | None = None,
    ) -> "Settings":
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
            prompt_profile=prompt_profile if prompt_profile is not None else os.getenv("PROMPT_PROFILE", "legacy"),
            structured_state=structured_state if structured_state is not None else os.getenv("STRUCTURED_STATE", "off"),
            history_format=history_format if history_format is not None else os.getenv("HISTORY_FORMAT", "replay"),
            goal_management=goal_management if goal_management is not None else os.getenv("GOAL_MANAGEMENT", "off"),
            actor_type=actor_type if actor_type is not None else os.getenv("ACTOR_TYPE", "chat"),
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
