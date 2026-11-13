"""Run one Rogue episode by invoking the graph once per game action."""

import asyncio
from dataclasses import asdict
from collections.abc import Callable
import json
import re
from uuid import UUID, uuid4

from langchain.chat_models import init_chat_model

from game_runner.budget import BudgetStop, TrialBudget, within_budget
from game_runner.clients.rogue import RogueAPIError, RogueClient, RogueTransportError
from game_runner.config import Settings
from game_runner.context.planning import ContextSettings
from game_runner.context.progress import Progress
from game_runner.graph.builder import build_graph
from game_runner.graph.nodes.step import SessionInterrupted
from game_runner.persistence.repository import GameRepository, ObservationOutcome
from game_runner.prompts.system import PROMPT_VERSION


async def run_session(
    settings: Settings, repository: GameRepository, model=None, *, summary_model=None,
    client: RogueClient | None = None, session_id: UUID | None = None, budget: TrialBudget | None = None,
    on_finish: Callable[[str, str], None] | None = None,
) -> str:
    if session_id is None:
        session_id = uuid4()
        repository.create_session(session_id, settings.model_name, PROMPT_VERSION, context_policy=settings.context_policy)

    def finish(status: str, error: str | None = None, *, reason: str | None = None) -> str:
        if on_finish is None:
            repository.update_session(session_id, status, error=error)
        else:
            on_finish(status, reason or status)
        print(json.dumps({
            "session_id": str(session_id), "final_status": status,
            "reason": reason or status, "error": error,
        }), flush=True)
        return status

    model_kwargs = {
        "model_provider": "openai",
        "base_url": settings.model_base_url,
        "api_key": settings.model_api_key,
        "timeout": settings.model_timeout_seconds,
        "max_retries": 0,
        "temperature": 0,
        **settings.model_kwargs,
    }
    try:
        if model is None:
            if "max_tokens" not in model_kwargs and "max_completion_tokens" not in model_kwargs:
                model_kwargs["max_tokens"] = settings.actor_output_tokens
            model = init_chat_model(settings.model_name, **model_kwargs)
        if settings.context_policy == "working_memory" and summary_model is None:
            summary_kwargs = {**model_kwargs, "max_retries": 0, "max_tokens": settings.summary_output_tokens}
            summary_kwargs.pop("max_completion_tokens", None)
            summary_model = init_chat_model(settings.model_name, **summary_kwargs)
    except Exception as exc:
        return finish("error", _safe_error(exc, budget), reason="model_initialization")
    print(json.dumps({
        "session_id": str(session_id), "status": "starting", "context_policy": settings.context_policy,
    }), flush=True)

    rogue = client if client is not None else RogueClient(settings.rogue_api_url, settings.rogue_timeout_seconds)
    async with rogue:
        if budget:
            try:
                budget.check(action=True, call=True)
            except BudgetStop as exc:
                return finish("limit_reached", reason=str(exc))
        event_id = repository.start_game_event(session_id, "reset")
        try:
            observation = await within_budget(budget, lambda: rogue.reset(settings.game_seed))
        except BudgetStop as exc:
            repository.finish_game_event(event_id, "uncertain", error=str(exc))
            return finish("limit_reached", reason=str(exc))
        except RogueAPIError as exc:
            repository.finish_game_event(event_id, "failed", http_status=exc.status_code, error=exc.detail)
            return finish("error", _safe_error(exc, budget), reason="reset_api_error")
        except RogueTransportError as exc:
            repository.finish_game_event(event_id, "uncertain", error=_safe_error(exc, budget))
            return finish("interrupted", _safe_error(exc, budget), reason="reset_transport_error")
        except Exception as exc:
            repository.finish_game_event(event_id, "uncertain", error=_safe_error(exc, budget))
            return finish("error", _safe_error(exc, budget), reason="reset_error")

        progress = Progress.initial(event_id, observation)
        repository.finish_game_event(
            event_id, "succeeded", api_step=observation.api_step,
            http_status=200, observation=observation.model_dump(mode="json"), progress=asdict(progress),
        )
        repository.update_session(
            session_id, "running",
            episode_id=observation.episode_id, seed=observation.seed,
            error=observation.error,
        )
        if budget and observation.seed != settings.game_seed:
            return finish("error", reason="seed_mismatch")
        if observation.terminated:
            return finish(observation.status, observation.error)

        try:
            graph = build_graph(
                model, rogue, repository,
                context_token_budget=settings.context_token_budget,
                max_model_calls=settings.max_model_calls_per_action,
                context_settings=ContextSettings(
                    token_budget=settings.context_token_budget, history_limit=settings.history_recent_interactions,
                    actor_output_tokens=settings.actor_output_tokens, policy=settings.context_policy,
                    summary_token_budget=settings.summary_context_token_budget,
                    summary_output_tokens=settings.summary_output_tokens, summary_attempts=settings.max_summary_attempts,
                ),
                summary_model=summary_model,
                budget=budget,
            )
        except Exception as exc:
            return finish("error", _safe_error(exc, budget), reason="graph_initialization")
        actions = 0
        observation_event_id = event_id
        observation_outcome: ObservationOutcome = "succeeded"
        while not observation.terminated and actions < settings.max_game_actions:
            try:
                if budget:
                    budget.check(action=True, call=True)
                result = await graph.ainvoke({
                    "session_id": session_id, "observation": observation,
                    "observation_event_id": observation_event_id, "observation_outcome": observation_outcome,
                    "attempts": 0,
                    "progress": progress,
                })
            except BudgetStop as exc:
                return finish("limit_reached", reason=str(exc))
            except SessionInterrupted as exc:
                return finish("interrupted", _safe_error(exc, budget), reason="action_unresolved")
            except RogueAPIError as exc:
                status = "interrupted" if exc.status_code == 409 else "error"
                return finish(status, _safe_error(exc, budget), reason="action_api_error")
            except Exception as exc:
                return finish("error", _safe_error(exc, budget), reason=_error_reason(exc))
            observation = result["next_observation"]
            observation_event_id = result["next_observation_event_id"]
            observation_outcome = result["next_observation_outcome"]
            progress = result["next_progress"]
            actions += 1
            print(
                json.dumps({
                    "session_id": str(session_id), "action_count": actions,
                    "api_step": observation.api_step, "status": observation.status,
                    "no_movement_streak": progress.no_movement_streak,
                }),
                flush=True,
            )

        status = observation.status if observation.terminated else "limit_reached"
        return finish(status, observation.error, reason="trial_action_limit" if status == "limit_reached" else None)


def _safe_error(error: Exception, budget: TrialBudget | None) -> str:
    if budget is None:
        return str(error)
    return _error_reason(error)


def _error_reason(error: Exception) -> str:
    return re.sub(r"(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])", "_", type(error).__name__).lower()


def main() -> None:
    status = asyncio.run(run_session(Settings.from_env(), GameRepository()))
    if status not in {"won", "dead", "quit"}:
        raise SystemExit(1)
