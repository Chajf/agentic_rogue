from fastapi.testclient import TestClient

from rogue_api.api import app


def test_health_without_game() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "game": "not_started"}


def test_state_without_game_is_not_found() -> None:
    with TestClient(app) as client:
        response = client.get("/game/state")

    assert response.status_code == 404


def test_action_contract_is_flat() -> None:
    schema = app.openapi()["components"]["schemas"]

    assert "episode_id" in schema["SemanticActionEnvelope"]["properties"]
    assert "action" in schema["SemanticActionEnvelope"]["properties"]
    assert "episode_id" in schema["KeyActionEnvelope"]["properties"]
    assert "key" in schema["KeyActionEnvelope"]["properties"]
