from uuid import uuid4

import pytest

import game_runner.pilot.store as store
from game_runner import migrate
from game_runner.config import Settings
from game_runner.pilot.manifest import Manifest

from test_persistence import isolated_database  # noqa: F401


def make_manifest(*, split="development"):
    return Manifest.model_validate({
        "schema_version": 1, "series_id": str(uuid4()), "split": split,
        "actor": {"model_name": "test-model"}, "trial_limits": {"actions": 3, "model_calls": 5, "seconds": 10},
        "series_limits": {"actions": 6, "model_calls": 10, "seconds": 20},
        "trials": [
            {"trial_id": "first", "seed": 1234, "context_policy": "history_only"},
            {"trial_id": "second", "seed": 1234, "context_policy": "working_memory"},
        ],
    })


async def test_series_lock_seed_split_and_abandonment(isolated_database, monkeypatch, tmp_path):
    from test_persistence import MIGRATIONS

    for migration in sorted(MIGRATIONS.glob("00[123]_*.sql")):
        (tmp_path / migration.name).write_text(migration.read_text())
    with monkeypatch.context() as patch:
        patch.setenv("MIGRATIONS_DIR", str(tmp_path))
        migrate.main()
    unrelated, memory = uuid4(), uuid4()
    with isolated_database() as connection:
        connection.execute(
            "INSERT INTO game_sessions(id, episode_id, status, model_name, prompt_version, context_policy) "
            "VALUES (%s, %s, 'running', 'p2', 'rogue-v1.3', 'working_memory')", (unrelated, memory),
        )
    migrate.main()
    monkeypatch.setattr(store, "connect", isolated_database)
    manifest = make_manifest()
    with store.series_lock(manifest.series_id) as connection:
        store.create_series(connection, manifest, {"revision": "abc"})
        with pytest.raises(store.PilotError, match="active"):
            with store.series_lock(manifest.series_id):
                pass
        with pytest.raises(store.PilotError, match="already exists"):
            store.create_series(connection, manifest, {})
        store.claim_trial(connection, manifest, manifest.trials[0], uuid4())

    evaluation = make_manifest(split="pilot_evaluation")
    with store.series_lock(evaluation.series_id) as connection:
        with pytest.raises(store.PilotError, match="development split"):
            store.create_series(connection, evaluation, {})
        assert connection.execute("SELECT count(*) FROM pilot_series").fetchone() == (1,)

    with store.series_lock(manifest.series_id) as connection:
        store.finalize_series(connection, manifest.series_id, "abandoned")
        assert connection.execute(
            "SELECT trial_id, outcome, reason FROM pilot_trials ORDER BY ordinal",
        ).fetchall() == [("first", "interrupted", "abandoned"), ("second", "unstarted", "abandoned")]
        assert connection.execute("SELECT status FROM game_sessions WHERE id = %s", (unrelated,)).fetchone() == ("running",)


def test_manifest_rejects_duplicate_trials_unmatched_evaluation_and_secrets():
    manifest = make_manifest()
    payload = manifest.model_dump(mode="json")
    payload["trials"][1]["trial_id"] = "first"
    with pytest.raises(ValueError, match="trial IDs"):
        Manifest.model_validate(payload)
    payload["trials"] = payload["trials"][:1]
    payload["split"] = "pilot_evaluation"
    with pytest.raises(ValueError, match="one trial per policy"):
        Manifest.model_validate(payload)
    payload["actor"]["api_key"] = "secret"
    with pytest.raises(ValueError, match="extra_forbidden"):
        Manifest.model_validate(payload)


def test_manifest_applies_explicit_reasoning_setting_to_both_policies():
    runtime = Settings(
        "http://test", 10, "http://test", "fake", "local", 5, 42, 3, 2, 8192,
        model_kwargs={"extra_body": {"reasoning": {"enabled": True}}},
    )
    original = make_manifest()
    payload = original.model_dump(mode="json")
    payload["actor"]["reasoning_enabled"] = False
    manifest = Manifest.model_validate(payload)
    assert manifest.model_dump(mode="json")["actor"]["reasoning_enabled"] is False

    for trial in manifest.trials:
        assert "extra_body" not in original.settings(runtime, trial).model_kwargs
        assert manifest.settings(runtime, trial).model_kwargs["extra_body"] == {
            "reasoning": {"enabled": False},
        }


async def test_series_continues_in_order_and_leaves_budget_skips_unstarted(isolated_database, monkeypatch, tmp_path):
    from contextlib import asynccontextmanager

    from game_runner.pilot.runner import run_series

    migrate.main()
    monkeypatch.setattr(store, "connect", isolated_database)
    payload = make_manifest().model_dump(mode="json")
    payload["series_limits"]["actions"] = 1
    payload["trials"].append({"trial_id": "last", "seed": 1235, "context_policy": "history_only"})
    manifest = Manifest.model_validate(payload)
    binary = tmp_path / "rogue"
    binary.write_bytes(b"fake")
    order = []

    @asynccontextmanager
    async def fake_rogue(*_):
        async def close():
            order.append("close")
        order.append("open")
        yield object(), close

    async def execute(settings, repository, *, session_id, client, budget, on_finish):
        order.append(settings.context_policy)
        if settings.context_policy == "history_only":
            on_finish("error", "provider_error")
        else:
            budget.start_action()
            on_finish("dead", "dead")

    runtime = Settings("http://test", 10, "http://test", "fake", "local", 5, 42, 3, 2, 8192)
    assert await run_series(manifest, runtime, binary=binary, revision="0" * 40,
                            rogue_factory=fake_rogue, execute_trial=execute) == "series_action_limit"
    assert order == ["open", "history_only", "close", "working_memory", "close"]
    with isolated_database() as connection:
        assert connection.execute(
            "SELECT trial_id, outcome, reason FROM pilot_trials ORDER BY ordinal",
        ).fetchall() == [
            ("first", "failed", "provider_error"), ("second", "ended", "dead"),
            ("last", "unstarted", "series_action_limit"),
        ]
