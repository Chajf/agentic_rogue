FROM python:3.12-slim-bookworm

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir . \
    && useradd --create-home --uid 10001 runner

USER runner
CMD ["python", "-m", "game_runner"]
