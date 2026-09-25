"""Apply ordered SQL migrations exactly once per database."""

import os
from pathlib import Path

from .persistence.database import connect


def migration_directory() -> Path:
    configured = os.environ.get("MIGRATIONS_DIR")
    return Path(configured) if configured else Path.cwd() / "postgres" / "migrations"


def main() -> None:
    migrations = sorted(migration_directory().glob("[0-9][0-9][0-9]_*.sql"))
    if not migrations:
        raise RuntimeError(f"no SQL migrations found in {migration_directory()}")
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(1479174313)")
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            cursor.execute("SELECT version FROM schema_migrations")
            applied = {row[0] for row in cursor.fetchall()}
            known = {path.stem for path in migrations}
            if unknown := applied - known:
                raise RuntimeError(f"database contains unknown migrations: {sorted(unknown)}")
            for path in migrations:
                if path.stem in applied:
                    continue
                cursor.execute(path.read_text())
                cursor.execute("INSERT INTO schema_migrations(version) VALUES (%s)", (path.stem,))
                print(f"Applied migration {path.name}")


if __name__ == "__main__":
    main()
