# agentic_rogue
Agentic player adapted to Rogue5.4

## Repository layout

- `rogue-env/` contains the complete Rogue game container and its HTTP API.
- `experiments/pty/` contains the original standalone PTY prototypes.
- `docker-compose.yml` is the shared stack entrypoint for current and future services.

Python dependencies are managed as a uv workspace from the repository root.

```bash
uv sync --all-packages --group test
uv run --all-packages --group test pytest
docker compose up --build
```

## Sequential pilot

Install both workspace packages and build Rogue from the source revision in
`rogue-env/Dockerfile`. Set PostgreSQL connection variables and the model
endpoint and credentials in the process environment. The pilot applies migrations
and uses its own in-process Rogue API with a temporary game directory.

```bash
uv run --locked --all-packages python -m game_runner.pilot run \
  --manifest pilot.json --rogue-binary /path/to/rogue
uv run --locked --all-packages python -m game_runner.pilot export \
  --series-id 4fb2a4cf-5a57-4dc1-8e27-d300efcd3a97 --output reports/pilot-development
```

A minimal development manifest follows. Use a fresh series UUID for each run.
Evaluation manifests require a history-only and a working-memory trial for every
seed, with a seed set disjoint from development in the same database.

```json
{
  "schema_version": 1,
  "series_id": "4fb2a4cf-5a57-4dc1-8e27-d300efcd3a97",
  "split": "development",
  "actor": {"model_name": "your-provider-model", "history_interactions": 4},
  "trial_limits": {"actions": 20, "model_calls": 35, "seconds": 180},
  "series_limits": {"actions": 40, "model_calls": 70, "seconds": 360},
  "trials": [
    {"trial_id": "seed-314159-history", "seed": 314159, "context_policy": "history_only"},
    {"trial_id": "seed-314159-memory", "seed": 314159, "context_policy": "working_memory"}
  ]
}
```

The export includes one result per planned trial and rows linked by trial,
session, and source IDs. `SHA256SUMS` covers the manifest and JSONL files.
`bundle.sha256` is the SHA-256 of that checksum file.
