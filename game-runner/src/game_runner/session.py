"""Run one Rogue episode by invoking the graph once per game action."""

import asyncio
import json
from uuid import uuid4

from langchain.chat_models import init_chat_model

from game_runner.clients.rogue import RogueAPIError, RogueClient, RogueTransportError
from game_runner.config import Settings
from game_runner.graph.builder import build_graph
from game_runner.graph.nodes.step import SessionInterrupted
from game_runner.persistence.repository import GameRepository
from game_runner.prompts.system import PROMPT_VERSION


async def run_session(settings: Settings, repository: GameRepository, model=None) -> str:
    if model is None:
        model = init_chat_model(
            settings.model_name,
            model_provider="openai",
            base_url=settings.model_base_url,
            api_key=settings.model_api_key,
            timeout=settings.model_timeout_seconds,
            max_retries=0,
            temperature=0,
        )

    repository.interrupt_open_sessions()
    session_id = uuid4()
    repository.create_session(session_id, settings.model_name, PROMPT_VERSION)
    print(json.dumps({"session_id": str(session_id), "status": "starting"}), flush=True)

    def finish(status: str, error: str | None = None) -> str:
        repository.update_session(session_id, status, error=error)
        print(json.dumps({"session_id": str(session_id), "final_status": status, "error": error}), flush=True)
        return status

    async with RogueClient(settings.rogue_api_url, settings.rogue_timeout_seconds) as client:
        event_id = repository.start_game_event(session_id, "reset")
        try:
            observation = await client.reset(settings.game_seed)
        except RogueAPIError as exc:
            repository.finish_game_event(event_id, "failed", http_status=exc.status_code, error=exc.detail)
            return finish("error", str(exc))
        except RogueTransportError as exc:
            repository.finish_game_event(event_id, "uncertain", error=str(exc))
            return finish("interrupted", str(exc))
        except Exception as exc:
            repository.finish_game_event(event_id, "uncertain", error=str(exc))
            return finish("error", str(exc))

        repository.finish_game_event(
            event_id, "succeeded", api_step=observation.api_step,
            http_status=200, observation=observation.model_dump(mode="json"),
        )
        repository.update_session(
            session_id, "running" if not observation.terminated else observation.status,
            episode_id=observation.episode_id, seed=observation.seed,
            error=observation.error,
        )
        if observation.terminated:
            return finish(observation.status, observation.error)

        try:
            graph = build_graph(
                model, client, repository,
                context_token_budget=settings.context_token_budget,
                max_model_calls=settings.max_model_calls_per_action,
            )
        except Exception as exc:
            return finish("error", str(exc))
        actions = 0
        while not observation.terminated and actions < settings.max_game_actions:
            try:
                result = await graph.ainvoke(
                    {"session_id": session_id, "observation": observation, "attempts": 0}
                )
            except SessionInterrupted as exc:
                return finish("interrupted", str(exc))
            except RogueAPIError as exc:
                status = "interrupted" if exc.status_code == 409 else "error"
                return finish(status, str(exc))
            except Exception as exc:
                return finish("error", str(exc))
            observation = result["next_observation"]
            actions += 1
            print(
                json.dumps({
                    "session_id": str(session_id), "action_count": actions,
                    "api_step": observation.api_step, "status": observation.status,
                }),
                flush=True,
            )

        status = observation.status if observation.terminated else "limit_reached"
        return finish(status, observation.error)


def main() -> None:
    status = asyncio.run(run_session(Settings.from_env(), GameRepository()))
    if status not in {"won", "dead", "quit"}:
        raise SystemExit(1)
