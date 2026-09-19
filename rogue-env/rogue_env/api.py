from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.params import Body

from .actions import InvalidAction
from .config import Settings
from .environment import IncompatibleMode, NoActiveEpisode, RogueEnv, StaleEpisode
from .models import ActionEnvelope, HealthResponse, Observation, ResetRequest


settings = Settings()
environment = RogueEnv(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await environment.close()


app = FastAPI(title="Rogue Environment", version="0.1.0", lifespan=lifespan)


@app.post("/game/reset", response_model=Observation)
async def reset_game(request: Annotated[ResetRequest, Body()] = ResetRequest()) -> Observation:
    return await environment.reset(request.seed)


@app.get("/game/state", response_model=Observation)
async def game_state() -> Observation:
    try:
        return await environment.observe()
    except NoActiveEpisode as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/game/action", response_model=Observation)
async def game_action(
    request: Annotated[ActionEnvelope, Body(discriminator="type")],
) -> Observation:
    try:
        return await environment.step(request.episode_id, request)
    except NoActiveEpisode as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except StaleEpisode as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except IncompatibleMode as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidAction as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/game", status_code=204)
async def delete_game() -> None:
    await environment.close()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if not environment.has_episode:
        return HealthResponse(status="healthy", game="not_started")
    if environment.process_running:
        return HealthResponse(status="healthy", game="running")
    return HealthResponse(status="degraded", game="failed" if environment.error else "stopped")
