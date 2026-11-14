"""Validated, serializable settings shared across a series."""

from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from game_runner.config import Settings

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Policy = Literal["history_only", "working_memory"]


class ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Limits(ManifestModel):
    actions: PositiveInt
    model_calls: PositiveInt
    seconds: PositiveInt


class Actor(ManifestModel):
    model_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0
    reasoning_enabled: Annotated[bool, Field(strict=True)] | None = None
    output_tokens: PositiveInt = 512
    context_tokens: PositiveInt = 8192
    history_interactions: Annotated[int, Field(strict=True, ge=0)] = 8
    max_attempts: PositiveInt = 3
    summary_context_tokens: PositiveInt = 8192
    summary_output_tokens: PositiveInt = 1024
    max_summary_attempts: PositiveInt = 2

    @model_validator(mode="after")
    def validate_budgets(self):
        if self.context_tokens <= self.output_tokens + 256:
            raise ValueError("actor context must leave room for input and retry feedback")
        if self.summary_context_tokens <= self.summary_output_tokens:
            raise ValueError("summary context must leave room for input")
        return self


class Trial(ManifestModel):
    trial_id: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")]
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
    context_policy: Policy


class Manifest(ManifestModel):
    schema_version: Literal[1]
    series_id: UUID
    split: Literal["development", "pilot_evaluation"]
    actor: Actor
    trial_limits: Limits
    series_limits: Limits
    trials: Annotated[tuple[Trial, ...], Field(min_length=1)]

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        return cls.model_validate_json(path.read_bytes())

    @model_validator(mode="after")
    def validate_trials(self):
        ids = [trial.trial_id for trial in self.trials]
        if len(set(ids)) != len(ids):
            raise ValueError("trial IDs must be unique")
        if self.split == "pilot_evaluation":
            pairs: dict[int, list[str]] = {}
            for trial in self.trials:
                pairs.setdefault(trial.seed, []).append(trial.context_policy)
            if any(sorted(policies) != ["history_only", "working_memory"] for policies in pairs.values()):
                raise ValueError("evaluation requires one trial per policy for each seed")
        return self

    def settings(self, runtime: Settings, trial: Trial) -> Settings:
        actor = self.actor
        # Only transport headers come from MODEL_KWARGS. Inference settings
        # come from the frozen manifest and are shared across policies.
        kwargs = {key: runtime.model_kwargs[key] for key in ("default_headers",) if key in runtime.model_kwargs}
        kwargs.update(temperature=actor.temperature, max_tokens=actor.output_tokens, max_retries=0)
        if actor.reasoning_enabled is not None:
            kwargs["extra_body"] = {"reasoning": {"enabled": actor.reasoning_enabled}}
        return Settings(
            rogue_api_url=runtime.rogue_api_url, rogue_timeout_seconds=runtime.rogue_timeout_seconds,
            model_base_url=runtime.model_base_url, model_api_key=runtime.model_api_key,
            model_name=actor.model_name, model_timeout_seconds=runtime.model_timeout_seconds,
            game_seed=trial.seed, max_game_actions=self.trial_limits.actions,
            max_model_calls_per_action=actor.max_attempts, context_token_budget=actor.context_tokens,
            model_kwargs=kwargs, history_recent_interactions=actor.history_interactions,
            context_policy=trial.context_policy, summary_context_token_budget=actor.summary_context_tokens,
            summary_output_tokens=actor.summary_output_tokens, max_summary_attempts=actor.max_summary_attempts,
        )
