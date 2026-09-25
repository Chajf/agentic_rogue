from uuid import uuid4

import httpx
import pytest

from game_runner.clients.rogue import (
    ActionRequest,
    Direction,
    RogueAPIError,
    RogueClient,
    RogueTransportError,
    SemanticAction,
)
from rogue_api.models import SemanticAction as ServiceAction


def observation(episode_id: str, api_step: int) -> dict:
    return {
        "episode_id": episode_id,
        "seed": 42,
        "api_step": api_step,
        "status": "running",
        "mode": "normal",
        "messages": [],
        "screen": ["@"],
        "cursor": {"row": 0, "column": 0},
        "state": {"hp": 12},
        "raw_output": "",
        "terminated": False,
        "error": None,
    }


def test_semantic_actions_match_service_contract() -> None:
    assert {action.value for action in SemanticAction} == {action.value for action in ServiceAction}


@pytest.mark.asyncio
async def test_reset_observe_and_action_use_existing_contract() -> None:
    episode_id = uuid4()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        step = 1 if request.url.path == "/game/action" else 0
        return httpx.Response(200, json=observation(str(episode_id), step))

    async with httpx.AsyncClient(
        base_url="http://rogue-api:8000", transport=httpx.MockTransport(handler)
    ) as http_client:
        async with RogueClient("http://rogue-api:8000", client=http_client) as client:
            assert (await client.reset(42)).api_step == 0
            assert (await client.observe()).episode_id == episode_id
            result = await client.action(
                episode_id, ActionRequest(action=SemanticAction.THROW, item="a", direction=Direction.UP)
            )

    assert result.api_step == 1
    assert [request.url.path for request in requests] == [
        "/game/reset", "/game/state", "/game/action"
    ]
    assert requests[2].read().decode() == (
        f'{{"episode_id":"{episode_id}","type":"semantic","action":"THROW",'
        '"item":"a","direction":"UP"}'
    )


@pytest.mark.asyncio
async def test_http_conflict_and_transport_failure_are_distinct() -> None:
    def conflict(_: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "stale episode"})

    async with httpx.AsyncClient(
        base_url="http://rogue-api:8000", transport=httpx.MockTransport(conflict)
    ) as http_client:
        async with RogueClient("http://rogue-api:8000", client=http_client) as client:
            with pytest.raises(RogueAPIError) as error:
                await client.observe()
            assert error.value.status_code == 409

    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no response", request=request)

    async with httpx.AsyncClient(
        base_url="http://rogue-api:8000", transport=httpx.MockTransport(timeout)
    ) as http_client:
        async with RogueClient("http://rogue-api:8000", client=http_client) as client:
            with pytest.raises(RogueTransportError):
                await client.action(uuid4(), ActionRequest(action=SemanticAction.REST))
