"""Write-ahead records for model calls and Rogue API requests."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from game_runner.clients.rogue import Observation

from .database import connect

ObservationOutcome = Literal["succeeded", "uncertain"]


@dataclass(frozen=True)
class DecisionTransition:
    input_event_id: int
    input_observation: Observation
    input_outcome: ObservationOutcome
    model_call_id: int
    action: dict
    rationale: str
    output_event_id: int
    output_observation: Observation
    output_outcome: Literal["succeeded"]


class GameRepository:
    def interrupt_open_sessions(self) -> None:
        with connect() as connection:
            connection.execute(
                "UPDATE game_sessions SET status = 'interrupted', ended_at = now(), "
                "error = 'runner restarted before session completion' "
                "WHERE status IN ('starting', 'running')"
            )

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

    def start_model_call(
        self, session_id: UUID, input_messages: list[dict], *, observation_event_id: int,
    ) -> int:
        with connect() as connection:
            row = connection.execute(
                "INSERT INTO model_calls(session_id, input_messages, observation_event_id, status) "
                "VALUES (%s, %s, %s, 'pending') RETURNING id",
                (session_id, Jsonb(input_messages), observation_event_id),
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

    def recent_transitions(
        self, session_id: UUID, observation_event_id: int, limit: int,
    ) -> list[DecisionTransition]:
        if limit == 0:
            return []
        with connect() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    """
                    WITH RECURSIVE completed AS NOT MATERIALIZED (
                        SELECT input.id AS input_event_id, input.observation AS input_observation,
                               input.outcome AS input_outcome, call.id AS model_call_id,
                               call.parsed_action AS action, call.decision_rationale AS rationale,
                               output.id AS output_event_id, output.observation AS output_observation,
                               output.outcome AS output_outcome
                        FROM game_events AS output
                        JOIN model_calls AS call
                            ON call.id = output.model_call_id AND call.session_id = output.session_id
                        JOIN game_events AS input
                            ON input.id = call.observation_event_id AND input.session_id = call.session_id
                        WHERE output.session_id = %s AND output.kind = 'action'
                          AND output.outcome = 'succeeded' AND output.observation IS NOT NULL
                          AND call.status = 'valid' AND call.parsed_action IS NOT NULL
                          AND call.decision_rationale IS NOT NULL
                          AND input.id < output.id AND input.observation IS NOT NULL
                          AND (input.outcome = 'succeeded'
                               OR (input.kind = 'action' AND input.outcome = 'uncertain'))
                    ), history AS (
                        SELECT completed.*, 1 AS depth FROM completed WHERE output_event_id = %s
                        UNION ALL
                        SELECT completed.*, history.depth + 1
                        FROM history JOIN completed ON completed.output_event_id = history.input_event_id
                        WHERE history.depth < %s
                    )
                    SELECT input_event_id, input_observation, input_outcome, model_call_id, action,
                           rationale, output_event_id, output_observation, output_outcome
                    FROM history ORDER BY depth DESC
                    """,
                    (session_id, observation_event_id, limit),
                )
                return [
                    DecisionTransition(**{
                        **row,
                        "input_observation": Observation.model_validate(row["input_observation"]),
                        "output_observation": Observation.model_validate(row["output_observation"]),
                    })
                    for row in cursor.fetchall()
                ]
