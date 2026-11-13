"""Exercise the runner migrations and history against an isolated PostgreSQL schema."""

import os
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.errors import ForeignKeyViolation
from psycopg.types.json import Jsonb

from game_runner import migrate
from game_runner.persistence import repository
from game_runner.persistence.repository import GameRepository

MIGRATIONS = Path(__file__).resolve().parents[2] / "postgres" / "migrations"
ACTION = {"type": "semantic", "action": "REST"}


@pytest.fixture
def isolated_database(monkeypatch):
    dsn = os.getenv("TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("TEST_POSTGRES_DSN is not set")
    schema = f"test_{uuid4().hex}"
    identifier = sql.Identifier(schema)

    with psycopg.connect(dsn) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(identifier))

    def connect():
        return psycopg.connect(dsn, options=f"-c search_path={schema}")

    monkeypatch.setattr(migrate, "connect", connect)
    monkeypatch.setattr(repository, "connect", connect)
    monkeypatch.setenv("MIGRATIONS_DIR", str(MIGRATIONS))
    try:
        yield connect
    finally:
        with psycopg.connect(dsn) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(identifier))


@pytest.mark.parametrize("legacy", [False, True], ids=["empty-db", "legacy-db"])
def test_migration_preserves_sessions_and_enforces_observation_link(isolated_database, tmp_path, monkeypatch, legacy):
    session_id = uuid4()
    input_messages = [{"role": "human", "content": "Original observation"}]
    observation = {"api_step": 0}

    if legacy:
        initial = MIGRATIONS / "001_initial.sql"
        (tmp_path / initial.name).write_text(initial.read_text())
        with monkeypatch.context() as patch:
            patch.setenv("MIGRATIONS_DIR", str(tmp_path))
            migrate.main()
        with isolated_database() as connection:
            connection.execute(
                "INSERT INTO game_sessions(id, status, model_name, prompt_version) "
                "VALUES (%s, 'dead', 'model', 'rogue-v1.1')",
                (session_id,),
            )
            legacy_call_id = connection.execute(
                "INSERT INTO model_calls(session_id, input_messages, status) "
                "VALUES (%s, %s, 'valid') RETURNING id",
                (session_id, Jsonb(input_messages)),
            ).fetchone()[0]
            legacy_event_id = connection.execute(
                "INSERT INTO game_events(session_id, kind, action_request, model_call_id, outcome, observation) "
                "VALUES (%s, 'action', %s, %s, 'succeeded', %s) RETURNING id",
                (session_id, Jsonb(ACTION), legacy_call_id, Jsonb(observation)),
            ).fetchone()[0]

    migrate.main()
    migrate.main()
    repo = GameRepository()
    if legacy:
        with isolated_database() as connection:
            assert connection.execute(
                "SELECT observation_event_id, input_messages FROM model_calls WHERE id = %s",
                (legacy_call_id,),
            ).fetchone() == (None, input_messages)
            assert connection.execute(
                "SELECT model_call_id, observation FROM game_events WHERE id = %s",
                (legacy_event_id,),
            ).fetchone() == (legacy_call_id, observation)
        assert repo.recent_transitions(session_id, legacy_event_id, 8) == []
    else:
        repo.create_session(session_id, "model", "rogue-v1.2")

    other_session = uuid4()
    repo.create_session(other_session, "model", "rogue-v1.2")
    input_id = repo.start_game_event(session_id, "reset")
    repo.finish_game_event(input_id, "succeeded", api_step=0, observation=observation)
    call_id = repo.start_model_call(session_id, input_messages, observation_event_id=input_id)
    with isolated_database() as connection:
        assert connection.execute(
            "SELECT session_id, observation_event_id, input_messages FROM model_calls WHERE id = %s",
            (call_id,),
        ).fetchone() == (session_id, input_id, input_messages)

    with pytest.raises(ForeignKeyViolation):
        repo.start_model_call(other_session, input_messages, observation_event_id=input_id)
    with pytest.raises(ForeignKeyViolation):
        repo.start_model_call(session_id, input_messages, observation_event_id=input_id + 1)


def test_reads_only_the_bounded_same_session_chain(isolated_database):
    migrate.main()
    repo = GameRepository()
    session_id, other_session, episode_id = uuid4(), uuid4(), uuid4()
    repo.create_session(session_id, "model", "rogue-v1.2")
    repo.create_session(other_session, "model", "rogue-v1.2")

    def record_observation(event_id, step, outcome):
        repo.finish_game_event(event_id, outcome, api_step=step, observation={
            "episode_id": str(episode_id), "seed": 42, "api_step": step,
            "status": "running", "mode": "normal", "messages": [f"Step {step}"],
            "screen": ["@  ."], "cursor": {"row": 0, "column": 0}, "state": {"hp": 12},
            "raw_output": "", "terminated": False,
        })

    def record_action(input_id, step, *, outcome="succeeded", call_status="valid"):
        call_id = repo.start_model_call(session_id, [], observation_event_id=input_id)
        repo.finish_model_call(call_id, call_status, parsed_action=ACTION, decision_rationale=f"Decision {step}")
        event_id = repo.start_game_event(session_id, "action", model_call_id=call_id, action_request=ACTION)
        if outcome in {"succeeded", "uncertain"}:
            record_observation(event_id, step, outcome)
        elif outcome == "failed":
            repo.finish_game_event(event_id, outcome)
        return event_id, call_id

    reset = repo.start_game_event(session_id, "reset")
    record_observation(reset, 0, "succeeded")
    invalid_call = repo.start_model_call(session_id, [], observation_event_id=reset)
    repo.finish_model_call(invalid_call, "invalid", raw_response={"content": "bad JSON"})
    first, first_call = record_action(reset, 1)
    second, second_call = record_action(first, 2)
    pending, _ = record_action(second, 3, outcome="pending")
    failed, _ = record_action(second, 3, outcome="failed")
    invalid, _ = record_action(second, 3, call_status="invalid")
    recovered, _ = record_action(second, 3, outcome="uncertain")
    fourth, fourth_call = record_action(recovered, 4)
    fifth, _ = record_action(fourth, 5)
    foreign_action = repo.start_game_event(other_session, "action", model_call_id=first_call, action_request=ACTION)
    record_observation(foreign_action, 1, "succeeded")

    before_recovery = repo.recent_transitions(session_id, second, 8)
    assert [(t.input_event_id, t.model_call_id, t.output_event_id) for t in before_recovery] == [
        (reset, first_call, first), (first, second_call, second),
    ]
    assert [(t.input_observation.api_step, t.output_observation.api_step) for t in before_recovery] == [
        (0, 1), (1, 2),
    ]
    assert before_recovery[0].action == ACTION
    assert before_recovery[0].rationale == "Decision 1"
    assert before_recovery[0].output_outcome == "succeeded"
    assert repo.recent_transitions(session_id, second, 1) == before_recovery[-1:]
    assert repo.recent_transitions(session_id, second, 0) == []
    for event_id in (reset, pending, failed, invalid, recovered, foreign_action):
        assert repo.recent_transitions(session_id, event_id, 8) == []
    assert repo.recent_transitions(other_session, second, 8) == []
    assert repo.recent_transitions(other_session, foreign_action, 8) == []

    rebuilt = repo.recent_transitions(session_id, fifth, 8)
    assert [t.output_event_id for t in rebuilt] == [fourth, fifth]
    assert rebuilt[0].input_event_id == recovered
    assert rebuilt[0].input_outcome == "uncertain"
    assert rebuilt[0].model_call_id == fourth_call
