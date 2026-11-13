"""Persist the frozen trial schedule and coordinate access to a series."""

from contextlib import contextmanager
from uuid import UUID

from psycopg.types.json import Jsonb

from game_runner.persistence.database import connect
from game_runner.pilot.manifest import Manifest, Trial
from game_runner.prompts.system import PROMPT_VERSION


class PilotError(RuntimeError):
    pass


@contextmanager
def series_lock(series_id: UUID):
    # Session-level lock deliberately lives on its own connection. A crashed
    # coordinator releases it when PostgreSQL closes that connection.
    with connect() as connection:
        connection.autocommit = True
        acquired, = connection.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
            (f"rogue-pilot:{series_id}",),
        ).fetchone()
        if not acquired:
            raise PilotError("series already has an active coordinator or exporter")
        yield connection


def create_series(connection, manifest: Manifest, provenance: dict) -> None:
    with connection.transaction():
        inserted = connection.execute(
            "INSERT INTO pilot_series(id, split, manifest) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING RETURNING id",
            (manifest.series_id, manifest.split, Jsonb({
                **manifest.model_dump(mode="json"), "provenance": provenance,
            })),
        ).fetchone()
        if inserted is None:
            raise PilotError("series ID already exists")
        # Sort lock acquisition for concurrent series with overlapping seeds.
        for seed in sorted({trial.seed for trial in manifest.trials}):
            connection.execute(
                "INSERT INTO pilot_seeds(seed, split) VALUES (%s, %s) ON CONFLICT (seed) DO NOTHING",
                (seed, manifest.split),
            )
            split, = connection.execute("SELECT split FROM pilot_seeds WHERE seed = %s", (seed,)).fetchone()
            if split != manifest.split:
                raise PilotError(f"seed {seed} already belongs to the {split} split")
        with connection.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO pilot_trials(series_id, trial_id, ordinal, split, requested_seed, context_policy) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [(manifest.series_id, trial.trial_id, index, manifest.split, trial.seed, trial.context_policy)
                 for index, trial in enumerate(manifest.trials)],
            )


def claim_trial(connection, manifest: Manifest, trial: Trial, session_id: UUID) -> None:
    with connection.transaction():
        connection.execute(
            "INSERT INTO game_sessions(id, status, model_name, prompt_version, context_policy) "
            "VALUES (%s, 'starting', %s, %s, %s)",
            (session_id, manifest.actor.model_name, PROMPT_VERSION, trial.context_policy),
        )
        updated = connection.execute(
            "UPDATE pilot_trials SET session_id = %s, outcome = 'running', started_at = now() "
            "WHERE series_id = %s AND trial_id = %s AND outcome = 'planned'",
            (session_id, manifest.series_id, trial.trial_id),
        )
        if updated.rowcount != 1:
            raise PilotError("trial is unavailable for execution")


def finalize_trial(connection, series_id: UUID, trial_id: str, status: str, reason: str) -> None:
    outcome = {
        "dead": "ended", "won": "ended", "quit": "ended", "limit_reached": "limited",
        "error": "failed", "interrupted": "interrupted",
    }[status]
    with connection.transaction():
        record = connection.execute(
            "UPDATE pilot_trials SET outcome = %s, reason = %s, ended_at = now() "
            "WHERE series_id = %s AND trial_id = %s AND outcome = 'running' RETURNING session_id",
            (outcome, reason, series_id, trial_id),
        ).fetchone()
        if record is None:
            raise PilotError("trial is unavailable for finalization")
        connection.execute(
            "UPDATE game_sessions SET status = %s, ended_at = COALESCE(ended_at, now()), "
            "error = COALESCE(error, %s) WHERE id = %s",
            (status, reason if status in {"error", "interrupted"} else None, record[0]),
        )


def finalize_series(connection, series_id: UUID, reason: str) -> None:
    with connection.transaction():
        connection.execute(
            "UPDATE game_sessions SET status = 'interrupted', ended_at = now(), error = %s "
            "WHERE id IN (SELECT session_id FROM pilot_trials WHERE series_id = %s AND outcome = 'running') "
            "AND status IN ('starting', 'running')",
            (reason, series_id),
        )
        connection.execute(
            "UPDATE pilot_trials SET outcome = CASE WHEN outcome = 'running' THEN 'interrupted' ELSE 'unstarted' END, "
            "reason = %s, ended_at = now() WHERE series_id = %s AND outcome IN ('running', 'planned')",
            (reason, series_id),
        )
        connection.execute(
            "UPDATE pilot_series SET ended_at = now(), stop_reason = %s WHERE id = %s AND ended_at IS NULL",
            (reason, series_id),
        )
