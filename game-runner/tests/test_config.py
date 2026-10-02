import pytest

from game_runner.config import Settings


@pytest.fixture
def model_env(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "http://localhost:8080/v1")
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.delenv("MODEL_KWARGS", raising=False)


@pytest.mark.parametrize("value", [None, "", "  ", "{}"])
def test_model_kwargs_default_to_empty_object(model_env, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("MODEL_KWARGS", value)
    assert Settings.from_env().model_kwargs == {}


def test_model_kwargs_preserve_json_values(model_env, monkeypatch):
    monkeypatch.setenv("MODEL_KWARGS", '{"temperature":0.7,"extra_body":{"enabled":true,"stop":null}}')
    assert Settings.from_env().model_kwargs == {
        "temperature": 0.7, "extra_body": {"enabled": True, "stop": None},
    }


@pytest.mark.parametrize("value", ["invalid", "[]", "null", "42", '"text"'])
def test_model_kwargs_reject_invalid_objects(model_env, monkeypatch, value):
    monkeypatch.setenv("MODEL_KWARGS", value)
    with pytest.raises(ValueError, match="MODEL_KWARGS must be"):
        Settings.from_env()


def test_model_name_uses_dedicated_variable(model_env, monkeypatch):
    monkeypatch.setenv("MODEL_KWARGS", '{"model":"other-model"}')
    with pytest.raises(ValueError, match="Use MODEL_NAME"):
        Settings.from_env()
