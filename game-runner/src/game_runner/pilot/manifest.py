"""Validated, serializable settings shared across a series."""

from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from game_runner.config import (
    ContextVariant, GoalManagement, GoalReviewThresholds, HistoryFormat, Settings, StructuredStateProfile,
    validate_goal_management, validate_history_format, validate_structured_state,
)
from game_runner.pilot.diagnostics import FrozenDiagnostic
from game_runner.prompts.system import PromptProfile

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Policy = Literal["history_only", "working_memory"]
Split = Literal["development", "pilot_evaluation", "held_out"]
Condition = Literal["A", "B", "C", "C-compact", "D"]
CONDITION_PROFILES = {
    "A": ("legacy", "off", "replay", "off"),
    "B": ("rules-v2", "off", "replay", "off"),
    "C": ("rules-v2", "map-v1", "replay", "off"),
    "C-compact": ("rules-v2", "map-v1", "compact", "off"),
}
MATCHED_CONTRASTS = {("A", "B"), ("B", "C"), ("C", "C-compact"), ("C", "D"), ("C-compact", "D")}


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


class Contrast(ManifestModel):
    reference: Condition
    treatment: Condition


class Timeouts(ManifestModel):
    model_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    rogue_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class Trial(ManifestModel):
    trial_id: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")]
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
    context_policy: Policy
    context_variant: ContextVariant = "baseline"
    prompt_profile: PromptProfile = "legacy"
    structured_state: StructuredStateProfile = "off"
    history_format: HistoryFormat = "replay"
    goal_management: GoalManagement = "off"
    goal_review: GoalReviewThresholds = Field(default_factory=GoalReviewThresholds)
    condition: Condition | None = None
    diagnostic: FrozenDiagnostic | None = None


class Manifest(ManifestModel):
    schema_version: Literal[1, 2]
    series_id: UUID
    split: Split
    contrasts: tuple[Contrast, ...] = ()
    timeouts: Timeouts | None = None
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
        if self.schema_version == 2:
            return self._validate_decision_loop()
        if self.split == "held_out" or self.contrasts or self.timeouts is not None or any(
            trial.condition is not None or trial.diagnostic is not None for trial in self.trials
        ):
            raise ValueError("decision-loop evaluation fields require manifest version 2")
        for trial in self.trials:
            validate_structured_state(trial.structured_state, trial.prompt_profile, trial.context_policy, trial.context_variant)
            validate_history_format(trial.history_format, trial.structured_state)
            validate_goal_management(trial.goal_management, trial.structured_state)
            if trial.structured_state == "map-v1" and self.split != "development":
                raise ValueError("map-v1 trials require the development split")
        if any(trial.prompt_profile == "rules-v2" and (
            self.split != "development" or trial.context_policy != "history_only" or trial.context_variant != "baseline"
        ) for trial in self.trials):
            raise ValueError("rules-v2 trials require development, history_only, and the baseline context variant")
        if any(trial.context_variant == "neighbors" for trial in self.trials):
            if self.split != "development" or any(trial.context_policy != "history_only" for trial in self.trials):
                raise ValueError("neighbor pilots require development and history_only for all trials")
            by_seed: dict[int, list[ContextVariant]] = {}
            for trial in self.trials:
                by_seed.setdefault(trial.seed, []).append(trial.context_variant)
            if any(group.count("neighbors") != 1 or "baseline" not in group for group in by_seed.values()):
                raise ValueError("neighbor pilots require one neighbors trial and at least one baseline per seed")
        if self.split == "pilot_evaluation":
            pairs: dict[int, list[str]] = {}
            for trial in self.trials:
                pairs.setdefault(trial.seed, []).append(trial.context_policy)
            if any(sorted(policies) != ["history_only", "working_memory"] for policies in pairs.values()):
                raise ValueError("evaluation requires one trial per policy for each seed")
        return self

    def _validate_decision_loop(self):
        if self.split == "pilot_evaluation" or self.timeouts is None:
            raise ValueError("decision-loop evaluation requires a development or held_out split and explicit timeouts")
        contrasts = [(entry.reference, entry.treatment) for entry in self.contrasts]
        if not contrasts or len(set(contrasts)) != len(contrasts) or any(pair not in MATCHED_CONTRASTS for pair in contrasts):
            raise ValueError("declare unique decision-loop contrasts")
        goal_references = [reference for reference, treatment in contrasts if treatment == "D"]
        if len(goal_references) > 1:
            raise ValueError("declare one common history format for the goal contrast")
        profiles = CONDITION_PROFILES.copy()
        if goal_references:
            profiles["D"] = (*profiles[goal_references[0]][:3], "managed-v1")

        diagnostics = [trial.diagnostic for trial in self.trials if trial.diagnostic is not None]
        if diagnostics and (self.split != "development" or len(diagnostics) != len(self.trials)):
            raise ValueError("diagnostics require a separate development series")
        scenarios = {}
        for spec in diagnostics:
            if scenarios.setdefault(spec.scenario_id, spec) != spec:
                raise ValueError("a scenario ID must use one frozen diagnostic specification")

        expected = {condition for pair in contrasts for condition in pair}
        blocks: dict[tuple[int, str | None], set[str]] = {}
        for trial in self.trials:
            profile = (trial.prompt_profile, trial.structured_state, trial.history_format, trial.goal_management)
            if (
                trial.condition not in expected or profiles.get(trial.condition) != profile
                or trial.context_policy != "history_only" or trial.context_variant != "baseline"
            ):
                raise ValueError("trial settings do not match the declared condition")
            spec = trial.diagnostic
            if spec is not None and spec.seed != trial.seed:
                raise ValueError("diagnostic seed must match its trial seed")
            block = blocks.setdefault((trial.seed, spec.scenario_id if spec else None), set())
            if trial.condition in block:
                raise ValueError("a matched block can contain only one trial per condition")
            block.add(trial.condition)
        if any(block != expected for block in blocks.values()):
            raise ValueError("each matched block must contain all declared conditions")
        return self

    def snapshot(self) -> dict:
        if self.schema_version == 1:
            return self.model_dump(mode="json", exclude={
                "contrasts": True, "timeouts": True, "trials": {"__all__": {"condition", "diagnostic"}},
            })
        return self.model_dump(mode="json")

    def settings(self, runtime: Settings, trial: Trial) -> Settings:
        actor = self.actor
        # Only transport headers come from MODEL_KWARGS. Inference settings
        # come from the frozen manifest and are shared across policies.
        kwargs = {key: runtime.model_kwargs[key] for key in ("default_headers",) if key in runtime.model_kwargs}
        kwargs.update(temperature=actor.temperature, max_tokens=actor.output_tokens, max_retries=0)
        if actor.reasoning_enabled is not None:
            kwargs["extra_body"] = {"reasoning": {"enabled": actor.reasoning_enabled}}
        return Settings(
            rogue_api_url=runtime.rogue_api_url,
            rogue_timeout_seconds=self.timeouts.rogue_seconds if self.timeouts else runtime.rogue_timeout_seconds,
            model_base_url=runtime.model_base_url, model_api_key=runtime.model_api_key,
            model_name=actor.model_name,
            model_timeout_seconds=self.timeouts.model_seconds if self.timeouts else runtime.model_timeout_seconds,
            game_seed=trial.seed, max_game_actions=self.trial_limits.actions,
            max_model_calls_per_action=actor.max_attempts, context_token_budget=actor.context_tokens,
            model_kwargs=kwargs, history_recent_interactions=actor.history_interactions,
            context_policy=trial.context_policy, summary_context_token_budget=actor.summary_context_tokens,
            summary_output_tokens=actor.summary_output_tokens, max_summary_attempts=actor.max_summary_attempts,
            context_variant=trial.context_variant, prompt_profile=trial.prompt_profile,
            structured_state=trial.structured_state, history_format=trial.history_format,
            goal_management=trial.goal_management, goal_review=trial.goal_review,
        )
