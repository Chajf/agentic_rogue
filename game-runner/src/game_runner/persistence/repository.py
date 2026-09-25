"""Write-ahead records for model calls and Rogue API requests."""

from uuid import UUID

from psycopg.types.json import Jsonb

from .database import connect


class GameRepository:
    def create_session(self, session_id: UUID, model_name: str, prompt_version: str) -> None:
        with connect() as connection:
            connection.execute(
                "INSERT INTO game_sessions(id, status, model_name, prompt_version) "
                "VALUES (%s, 'starting', %s, %s)",
                (session_id, model_name, prompt_version),
            )

    def update_session(
        self, session_id: UUID, status: str, *, episode_id: UUID | None = None,
        seed: int | None = None, error: str | None = None,
    ) -> None:
        with connect() as connection:
            connection.execute(
                "UPDATE game_sessions SET status = %s, "
                "episode_id = COALESCE(%s, episode_id), seed = COALESCE(%s, seed), "
                "error = %s, ended_at = CASE WHEN %s IN ('starting', 'running') "
                "THEN NULL ELSE now() END "
                "WHERE id = %s",
                (status, episode_id, seed, error, status, session_id),
            )

    def start_model_call(self, session_id: UUID, input_messages: list[dict]) -> int:
        with connect() as connection:
            row = connection.execute(
                "INSERT INTO model_calls(session_id, input_messages, status) "
                "VALUES (%s, %s, 'pending') RETURNING id",
                (session_id, Jsonb(input_messages)),
            ).fetchone()
            assert row is not None
            return row[0]

    def finish_model_call(
        self, call_id: int, status: str, *, raw_response: dict | None = None,
        parsed_action: dict | None = None, decision_rationale: str | None = None,
        validation_error: str | None = None, response_metadata: dict | None = None,
        usage_metadata: dict | None = None, duration_ms: int | None = None,
    ) -> None:
        with connect() as connection:
            connection.execute(
                "UPDATE model_calls SET status = %s, raw_response = %s, parsed_action = %s, "
                "decision_rationale = %s, validation_error = %s, response_metadata = %s, "
                "usage_metadata = %s, duration_ms = %s WHERE id = %s",
                (
                    status, Jsonb(raw_response) if raw_response is not None else None,
                    Jsonb(parsed_action) if parsed_action is not None else None,
                    decision_rationale, validation_error,
                    Jsonb(response_metadata) if response_metadata is not None else None,
                    Jsonb(usage_metadata) if usage_metadata is not None else None,
                    duration_ms, call_id,
                ),
            )

    def start_game_event(
        self, session_id: UUID, kind: str, *, model_call_id: int | None = None,
        action_request: dict | None = None,
    ) -> int:
        with connect() as connection:
            row = connection.execute(
                "INSERT INTO game_events(session_id, model_call_id, kind, action_request, outcome) "
                "VALUES (%s, %s, %s, %s, 'pending') RETURNING id",
                (
                    session_id, model_call_id, kind,
                    Jsonb(action_request) if action_request is not None else None,
                ),
            ).fetchone()
            assert row is not None
            return row[0]

    def finish_game_event(
        self, event_id: int, outcome: str, *, api_step: int | None = None,
        http_status: int | None = None, observation: dict | None = None,
        error: str | None = None,
    ) -> None:
        with connect() as connection:
            connection.execute(
                "UPDATE game_events SET outcome = %s, api_step = %s, http_status = %s, "
                "observation = %s, error = %s, completed_at = now() WHERE id = %s",
                (
                    outcome, api_step, http_status,
                    Jsonb(observation) if observation is not None else None,
                    error, event_id,
                ),
            )
