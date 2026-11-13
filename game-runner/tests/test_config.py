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


@pytest.mark.parametrize("value, expected", [(None, 8), ("0", 0), ("3", 3)])
def test_history_recent_interactions(model_env, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("HISTORY_RECENT_INTERACTIONS", raising=False)
    else:
        monkeypatch.setenv("HISTORY_RECENT_INTERACTIONS", value)
    assert Settings.from_env().history_recent_interactions == expected


@pytest.mark.parametrize("value", ["-1", "1.5", "text"])
def test_history_recent_interactions_rejects_invalid_values(model_env, monkeypatch, value):
    monkeypatch.setenv("HISTORY_RECENT_INTERACTIONS", value)
    with pytest.raises(ValueError, match="HISTORY_RECENT_INTERACTIONS"):
        Settings.from_env()


@pytest.mark.parametrize("policy", ["history_only", "working_memory"])
def test_context_policy_and_summary_limits(model_env, monkeypatch, policy):
    monkeypatch.setenv("CONTEXT_POLICY", policy)
    monkeypatch.setenv("SUMMARY_CONTEXT_TOKEN_BUDGET", "4096")
    monkeypatch.setenv("SUMMARY_OUTPUT_TOKENS", "750")
    monkeypatch.setenv("MAX_SUMMARY_ATTEMPTS", "1")
    monkeypatch.setenv("MODEL_KWARGS", '{"max_completion_tokens":700}')
    settings = Settings.from_env()
    assert settings.context_policy == policy
    assert settings.summary_context_token_budget == 4096
    assert settings.summary_output_tokens == 750
    assert settings.max_summary_attempts == 1
    assert settings.actor_output_tokens == 700


@pytest.mark.parametrize("key,value", [
    ("CONTEXT_POLICY", "other"), ("SUMMARY_OUTPUT_TOKENS", "0"),
    ("SUMMARY_CONTEXT_TOKEN_BUDGET", "512"), ("MAX_SUMMARY_ATTEMPTS", "0"),
    ("MODEL_KWARGS", '{"max_tokens":false}'),
])
def test_rejects_invalid_context_settings(model_env, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        Settings.from_env()
