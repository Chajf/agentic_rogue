"""Prepare and replay native diagnostic starts using linked public observations."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from game_runner.budget import BudgetStop, TrialBudget, within_budget
from game_runner.clients.rogue import ActionRequest, KeyActionRequest, Observation, RogueAPIError, RogueClient
from game_runner.context.progress import Progress, visible_position
from game_runner.context.transitions import DecisionTransition
from game_runner.context.working_state import WorkingState
from game_runner.decision import Decision, validate_mode
from game_runner.persistence.repository import GameRepository
from game_runner.rule_feasibility.artifacts import canonical

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ScriptAction = Annotated[ActionRequest | KeyActionRequest, Field(discriminator="type")]


class DiagnosticObjective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["new_position", "reach_target", "descend_here", "same_position_combat", "observe_current_map"]
    target: tuple[Annotated[int, Field(ge=1, le=22)], Annotated[int, Field(ge=0, le=79)]] | None = None

    @model_validator(mode="after")
    def check_target(self):
        if (self.target is not None) != (self.kind == "reach_target"):
            raise ValueError("only reach_target requires coordinates")
        return self


class DiagnosticSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    scenario_id: Identifier
    seed: Annotated[int, Field(strict=True, ge=0, le=2**32 - 1)]
    prefix: Annotated[tuple[ScriptAction, ...], Field(max_length=32)]
    objective: DiagnosticObjective | None = None


class FrozenDiagnostic(DiagnosticSpec):
    binary_sha256: Digest
    trace_sha256: Digest
    anchor_sha256: Digest


class DiagnosticMismatch(ValueError):
    """A diagnostic start lacks its frozen evidence."""


def action_payload(action: ScriptAction) -> dict:
    return action.model_dump(mode="json", exclude_none=True)


def public_frame(observation: Observation) -> dict:
    return observation.model_dump(mode="json", include={"screen", "mode", "messages", "state", "api_step", "status"})


def fingerprints(frames: list[Observation], actions: list[dict]) -> tuple[str, str]:
    public = [public_frame(frame) for frame in frames]
    trace = {"reset": public[0], "setup": [
        {"action": action, "observation": frame} for action, frame in zip(actions, public[1:], strict=True)
    ]}
    return sha256(canonical(trace)).hexdigest(), sha256(canonical(public[-1])).hexdigest()


def _require_reset(observation: Observation, seed: int) -> None:
    if observation.api_step != 0 or observation.seed != seed or observation.status != "running" or observation.terminated:
        raise DiagnosticMismatch("diagnostic reset is not a running observation for the requested seed")


def _require_link(before: Observation, after: Observation) -> None:
    if (after.api_step != before.api_step + 1 or after.episode_id != before.episode_id
            or after.seed != before.seed or after.status != "running" or after.terminated):
        raise DiagnosticMismatch("diagnostic setup action did not return a linked running observation")


def _require_anchor(observation: Observation) -> None:
    if visible_position(observation) is None:
        raise DiagnosticMismatch("diagnostic anchor requires a confirmed map position")


async def prepare_diagnostic(client: RogueClient, spec: DiagnosticSpec, binary_sha256: str) -> FrozenDiagnostic:
    initial = await client.reset(spec.seed)
    _require_reset(initial, spec.seed)
    frames = [initial]
    for action in spec.prefix:
        before = frames[-1]
        validate_mode(Decision(action=action, rationale="Scripted diagnostic setup."), before.mode)
        after = await client.action(before.episode_id, action)
        _require_link(before, after)
        frames.append(after)
    _require_anchor(frames[-1])
    trace, anchor = fingerprints(frames, [action_payload(action) for action in spec.prefix])
    return FrozenDiagnostic(**spec.model_dump(), binary_sha256=binary_sha256, trace_sha256=trace, anchor_sha256=anchor)


@dataclass(frozen=True)
class DiagnosticStart:
    observation: Observation
    progress: Progress
    evidence: list[dict]


async def run_diagnostic_setup(
    client: RogueClient, repository: GameRepository, session_id: UUID, spec: FrozenDiagnostic, binary_sha256: str,
    initial: Observation, progress: Progress, working_state: WorkingState | None, budget: TrialBudget | None,
) -> DiagnosticStart:
    _require_reset(initial, spec.seed)
    frames = [initial]
    actions = []
    if binary_sha256 != spec.binary_sha256:
        trace, anchor = fingerprints(frames, actions)
        repository.record_diagnostic_anchor(session_id, progress.event_id, _anchor_record(spec, trace, anchor, False))
        raise DiagnosticMismatch("Rogue binary does not match the frozen diagnostic")

    for action in spec.prefix:
        before = frames[-1]
        try:
            validate_mode(Decision(action=action, rationale="Scripted diagnostic setup."), before.mode)
        except ValueError as exc:
            raise DiagnosticMismatch("scripted action does not match the observed interaction mode") from exc
        payload = action_payload(action)
        input_event_id = progress.event_id
        event_id = repository.start_game_event(
            session_id, "action", action_request=payload, scripted_input_event_id=input_event_id,
        )
        try:
            after = await within_budget(budget, lambda: client.action(before.episode_id, action))
            _require_link(before, after)
        except RogueAPIError as exc:
            repository.finish_game_event(event_id, "failed", http_status=exc.status_code, error=exc.detail)
            raise DiagnosticMismatch("Rogue rejected a diagnostic setup action") from exc
        except Exception as exc:
            repository.finish_game_event(event_id, "uncertain", error=type(exc).__name__)
            if isinstance(exc, BudgetStop):
                raise
            raise DiagnosticMismatch("diagnostic setup action could not be confirmed") from exc

        progress = progress.advance(event_id, before, payload, after, recovered=False)
        persisted = asdict(progress)
        if working_state is not None:
            transition = DecisionTransition(
                input_event_id, before, "succeeded", None, payload, "", event_id, after, "succeeded",
            )
            persisted["working_state_delta"] = working_state.advance(transition, progress)
        repository.finish_game_event(
            event_id, "succeeded", http_status=200, api_step=after.api_step,
            observation=after.model_dump(mode="json"), progress=persisted,
        )
        frames.append(after)
        actions.append(payload)

    trace, anchor = fingerprints(frames, actions)
    verified = trace == spec.trace_sha256 and anchor == spec.anchor_sha256 and visible_position(frames[-1]) is not None
    repository.record_diagnostic_anchor(session_id, progress.event_id, _anchor_record(spec, trace, anchor, verified))
    if not verified:
        raise DiagnosticMismatch("diagnostic start does not match its frozen trace and anchor")
    return DiagnosticStart(frames[-1], progress, setup_evidence(frames, actions))


def _anchor_record(spec: FrozenDiagnostic, trace: str, anchor: str, verified: bool) -> dict:
    return {"scenario_id": spec.scenario_id, "verified": verified, "trace_sha256": trace, "anchor_sha256": anchor}


def setup_evidence(frames: list[Observation], actions: list[dict]) -> list[dict]:
    """Build shared prompt evidence independent of the trial's memory settings."""
    evidence = []
    for frame, action in list(zip(frames, [None, *actions], strict=True))[-8:]:
        position = visible_position(frame)
        glyphs = [
            {"glyph": glyph, "position": [row, col]}
            for row, line in enumerate(frame.screen[1:-1], 1)
            for col, glyph in enumerate(line) if glyph in {"%", "+"}
        ] if position is not None else []
        evidence.append({
            "api_step": frame.api_step, "action": action, "mode": frame.mode,
            "position": position, "depth": frame.state.dungeon_level, "hp": frame.state.hp,
            "messages": [" ".join(message.split())[:120] for message in frame.messages[:2]],
            "visible_landmarks": glyphs[:8],
        })
    return evidence
