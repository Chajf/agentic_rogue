FROM python:3.12-slim-bookworm

WORKDIR /app
COPY game-runner/pyproject.toml ./
COPY game-runner/src ./src
COPY postgres/migrations ./postgres/migrations
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 runner

USER runner
CMD ["sh", "-c", "python -m game_runner.migrate && exec python -m game_runner"]
