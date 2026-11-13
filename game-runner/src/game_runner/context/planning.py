"""Select replay, compact evicted evidence, and enforce prompt budgets."""

from dataclasses import dataclass
import logging
from uuid import UUID

from langchain_core.messages import BaseMessage

from game_runner.budget import TrialBudget
from game_runner.clients.rogue import Observation
from game_runner.context.budget import ContextLimitError, RETRY_RESERVE, count_messages
from game_runner.context.compaction import SummaryUnavailable, compact_batch, evidence_rows, summary_messages
from game_runner.context.memory import MemoryRevision
from game_runner.context.messages import build_messages
from game_runner.context.progress import Progress
from game_runner.context.transitions import DecisionTransition, ObservationOutcome
from game_runner.persistence.repository import GameRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextSettings:
    token_budget: int
    history_limit: int = 8
    actor_output_tokens: int = 512
    policy: str = "history_only"
    summary_token_budget: int = 8192
    summary_output_tokens: int = 1024
    summary_attempts: int = 2


@dataclass(frozen=True)
class PreparedContext:
    messages: list[BaseMessage]
    revision_id: int | None = None


@dataclass
class ContextPlanner:
    model: object
    repository: GameRepository
    settings: ContextSettings
    summary_model: object | None = None
    budget: TrialBudget | None = None

    async def prepare(
        self, session_id: UUID, observation: Observation, event_id: int,
        outcome: ObservationOutcome, progress: Progress,
    ) -> PreparedContext:
        settings = self.settings

        def render(history, revision=None, uncovered=None):
            return build_messages(
                observation, history, observation_outcome=outcome, progress=progress,
                memory=revision, uncovered=uncovered,
            )

        def fits(messages):
            reserved = settings.actor_output_tokens + RETRY_RESERVE
            return count_messages(self.model, messages).count + reserved <= settings.token_budget

        if not fits(render([])):
            raise ContextLimitError("current observation and actor reserves exceed context budget")

        if settings.policy == "history_only":
            history = self.repository.recent_transitions(session_id, event_id, settings.history_limit)
            return PreparedContext(_fit_recent(history, render, fits))
        if self.summary_model is None:
            raise SummaryUnavailable("working_memory requires a summary model")

        revision = self.repository.latest_memory(session_id)
        trace = self.repository.uncovered_trace(session_id, event_id, revision.through_event_id if revision else 0)
        _validate_trace(trace, revision, event_id, observation)
        recent = _replay_suffix(trace, settings.history_limit)
        if len(recent) == len(trace) and fits(render(recent, revision)):
            return PreparedContext(render(recent, revision), revision.id if revision else None)

        # Leave room for several actions before the next compaction. The retained
        # interactions must be an uninterrupted suffix after the latest recovery.
        retain = min(len(recent), settings.history_limit // 2)
        while True:
            while retain and not fits(render(trace[-retain:], revision)):
                retain -= 1
            for_compaction = len(trace) - retain
            if for_compaction == 0:
                messages = render(trace, revision)
                if fits(messages):
                    return PreparedContext(messages, revision.id if revision else None)
                raise ContextLimitError("episode memory and required actor context exceed the budget")

            prefix = trace[:for_compaction]
            rows = evidence_rows(prefix)
            # Limit what is sent to the provider, including room for a small
            # summary correction. Prefer the longest prefix that fits.
            size = 0
            for length in range(1, len(rows) + 1):
                messages = summary_messages(revision, rows[:length])
                estimated_total = count_messages(self.summary_model, messages).count + settings.summary_output_tokens
                if estimated_total + 96 > settings.summary_token_budget:
                    break
                size = length
            try:
                if size == 0:
                    raise ContextLimitError("oldest uncovered event cannot fit in the summary budget")
                revision = await compact_batch(
                    self.summary_model, self.repository, session_id, observation.episode_id,
                    revision, prefix[:size], rows[:size], token_budget=settings.summary_token_budget,
                    output_tokens=settings.summary_output_tokens, attempts=settings.summary_attempts,
                    fits_actor=lambda candidate: fits(render([], candidate)),
                    budget=self.budget,
                )
            except (SummaryUnavailable, ContextLimitError) as error:
                logger.warning("Episode %s could not compact through an uncovered prefix: %s", session_id, error)
                # Preserve uncovered evidence in the actor request if the whole
                # trace still fits. The next decision will retry this boundary.
                replay = _replay_suffix(trace, settings.history_limit)
                while True:
                    evicted = trace[:len(trace) - len(replay)]
                    messages = render(replay, revision, evidence_rows(evicted))
                    if fits(messages):
                        return PreparedContext(messages, revision.id if revision else None)
                    if not replay:
                        raise ContextLimitError(f"cannot retain uncovered episode evidence: {error}") from error
                    replay = replay[1:]
            trace = trace[size:]


def _fit_recent(history, render, fits):
    for offset in range(len(history)):
        messages = render(history[offset:])
        if fits(messages):
            return messages
    return render([])


def _replay_suffix(trace: list[DecisionTransition], limit: int) -> list[DecisionTransition]:
    if limit == 0:
        return []
    replay = []
    for transition in reversed(trace[-limit:]):
        if transition.output_outcome == "uncertain":
            break
        replay.append(transition)
    return list(reversed(replay))


def _validate_trace(
    trace: list[DecisionTransition], revision: MemoryRevision | None, event_id: int, current: Observation,
) -> None:
    expected = revision.through_event_id if revision else (trace[0].input_event_id if trace else event_id)
    step = trace[0].input_observation.api_step if trace else current.api_step
    if revision is None and step != 0:
        raise SummaryUnavailable("working-memory trace is missing its reset observation")
    for transition in trace:
        before, after = transition.input_observation, transition.output_observation
        if (
            transition.input_event_id != expected or before.api_step != step
            or before.episode_id != current.episode_id or after.episode_id != current.episode_id
            or after.api_step != step + 1
        ):
            raise SummaryUnavailable("working-memory trace has an unlinked observation")
        expected, step = transition.output_event_id, after.api_step
    if expected != event_id or step != current.api_step:
        raise SummaryUnavailable("working-memory trace does not reach the current observation")
