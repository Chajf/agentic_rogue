FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.17 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY rogue-env/pyproject.toml ./rogue-env/pyproject.toml
COPY game-runner/pyproject.toml ./game-runner/pyproject.toml
COPY game-runner/src ./game-runner/src
COPY postgres/migrations ./postgres/migrations
RUN uv sync --locked --package game-runner --no-dev --no-editable \
    && useradd --create-home --uid 10001 runner \
    && chown -R runner:runner /app

USER runner
CMD ["sh", "-c", "/app/.venv/bin/python -m game_runner.migrate && exec /app/.venv/bin/python -m game_runner"]
