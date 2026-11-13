"""Write-ahead records for model calls and Rogue API requests."""

from uuid import UUID

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from game_runner.clients.rogue import Observation
from game_runner.context.memory import EpisodeMemory, MemoryRevision, POLICY_VERSION, SCHEMA_VERSION

from game_runner.context.transitions import DecisionTransition

from .database import connect

class GameRepository:
    def create_session(
        self, session_id: UUID, model_name: str, prompt_version: str, *, context_policy: str | None = None,
    ) -> None:
        with connect() as connection:
            connection.execute(
                "INSERT INTO game_sessions(id, status, model_name, prompt_version, context_policy) "
                "VALUES (%s, 'starting', %s, %s, %s)",
                (session_id, model_name, prompt_version, context_policy),
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
        purpose: str = "decision", memory_revision_id: int | None = None,
        input_token_count: int | None = None, input_token_count_method: str | None = None,
    ) -> int:
        with connect() as connection:
            row = connection.execute(
                "INSERT INTO model_calls(session_id, input_messages, observation_event_id, purpose, "
                "memory_revision_id, input_token_count, input_token_count_method, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending') RETURNING id",
                (session_id, Jsonb(input_messages), observation_event_id, purpose, memory_revision_id,
                 input_token_count, input_token_count_method),
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
        error: str | None = None, progress: dict | None = None,
    ) -> None:
        with connect() as connection:
            connection.execute(
                "UPDATE game_events SET outcome = %s, api_step = %s, http_status = %s, "
                "observation = %s, error = %s, progress = %s, completed_at = now() WHERE id = %s",
                (
                    outcome, api_step, http_status,
                    Jsonb(observation) if observation is not None else None,
                    error, Jsonb(progress) if progress is not None else None, event_id,
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
                          AND call.purpose = 'decision' AND call.status = 'valid' AND call.parsed_action IS NOT NULL
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
                return [_transition(row) for row in cursor.fetchall()]

    def uncovered_trace(
        self, session_id: UUID, observation_event_id: int, through_event_id: int,
    ) -> list[DecisionTransition]:
        with connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT input.id AS input_event_id, input.observation AS input_observation,
                       input.outcome AS input_outcome, call.id AS model_call_id,
                       call.parsed_action AS action, call.decision_rationale AS rationale,
                       output.id AS output_event_id, output.observation AS output_observation,
                       output.outcome AS output_outcome, output.progress
                FROM game_events AS output
                JOIN model_calls AS call
                    ON call.id = output.model_call_id AND call.session_id = output.session_id
                JOIN game_events AS input
                    ON input.id = call.observation_event_id AND input.session_id = call.session_id
                WHERE output.session_id = %s AND output.id > %s AND output.id <= %s
                  AND output.kind = 'action' AND output.outcome IN ('succeeded', 'uncertain')
                  AND output.observation IS NOT NULL
                  AND input.id < output.id AND input.observation IS NOT NULL
                  AND call.purpose = 'decision' AND call.status = 'valid'
                  AND call.parsed_action IS NOT NULL AND call.decision_rationale IS NOT NULL
                ORDER BY output.id
                """,
                (session_id, through_event_id, observation_event_id),
            )
            return [_transition(row) for row in cursor.fetchall()]

    def latest_memory(self, session_id: UUID) -> MemoryRevision | None:
        with connect() as connection:
            return _read_memory(connection, session_id)

    def publish_memory(
        self, session_id: UUID, episode_id: UUID, *, previous: MemoryRevision | None,
        through_event_id: int, call_id: int, content: EpisodeMemory, gap_event_ids: frozenset[int],
        raw_response: dict, duration_ms: int,
    ) -> MemoryRevision:
        with connect() as connection:
            if connection.execute(
                "SELECT 1 FROM game_sessions WHERE id = %s AND episode_id = %s FOR UPDATE",
                (session_id, episode_id),
            ).fetchone() is None:
                raise ValueError("episode does not belong to this session")
            existing = _read_memory(connection, session_id, through_event_id)
            latest = _read_memory(connection, session_id)
            if existing is None and (latest != previous or (latest and latest.through_event_id >= through_event_id)):
                raise ValueError("memory predecessor or coverage changed before publication")
            previous_id = previous.id if previous else None
            updated = connection.execute(
                "UPDATE model_calls SET status = 'valid', raw_response = %s, duration_ms = %s, "
                "response_metadata = %s, usage_metadata = %s "
                "WHERE session_id = %s AND id = %s AND purpose = 'episode_summary' AND status = 'pending' "
                "AND observation_event_id = %s AND memory_revision_id IS NOT DISTINCT FROM %s",
                (Jsonb(raw_response), duration_ms, Jsonb(raw_response.get("response_metadata")),
                 Jsonb(raw_response.get("usage_metadata")), session_id, call_id, through_event_id, previous_id),
            )
            if updated.rowcount != 1:
                if existing is None or connection.execute(
                    "SELECT 1 FROM episode_memory_revisions WHERE id = %s AND model_call_id = %s",
                    (existing.id, call_id),
                ).fetchone() is None:
                    raise ValueError("invalid generating summary call")
            if existing is not None:
                return existing
            row = connection.execute(
                "INSERT INTO episode_memory_revisions(session_id, episode_id, previous_revision_id, "
                "through_event_id, schema_version, policy_version, content, model_call_id, gap_event_ids) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                (session_id, episode_id, previous_id, through_event_id, SCHEMA_VERSION, POLICY_VERSION,
                 Jsonb(content.model_dump(mode="json")), call_id, sorted(gap_event_ids)),
            ).fetchone()
            assert row is not None
            return MemoryRevision(row[0], through_event_id, content, gap_event_ids)


def _read_memory(connection, session_id: UUID, boundary: int | None = None) -> MemoryRevision | None:
    query = "SELECT id, through_event_id, content, gap_event_ids FROM episode_memory_revisions WHERE session_id = %s"
    row = (
        connection.execute(query + " ORDER BY through_event_id DESC LIMIT 1", (session_id,))
        if boundary is None else
        connection.execute(query + " AND through_event_id = %s", (session_id, boundary))
    ).fetchone()
    if row is None:
        return None
    return MemoryRevision(row[0], row[1], EpisodeMemory.model_validate(row[2]), frozenset(row[3]))


def _transition(row: dict) -> DecisionTransition:
    return DecisionTransition(**{
        **row,
        "input_observation": Observation.model_validate(row["input_observation"]),
        "output_observation": Observation.model_validate(row["output_observation"]),
    })
