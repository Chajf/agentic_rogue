# Rogue environment

Single-episode HTTP environment for [Rogue 5.4](https://github.com/lcn2/rogue5.4).
The service keeps Rogue alive in a fixed `80x24` PTY, reconstructs its terminal
with `pyte`, and exposes semantic or raw-key actions through FastAPI.

## Run

```bash
docker compose up --build
```

OpenAPI documentation is available at <http://localhost:8000/docs>. Rogue save,
score, and lock files live in the `rogue-data` named volume. A reset always starts
a fresh episode and preserves only the score file.

## API flow

Start a deterministic episode:

```bash
curl -X POST http://localhost:8000/game/reset \
  -H 'Content-Type: application/json' \
  -d '{"seed": 12345}'
```

Send an action using the returned `episode_id`:

```bash
curl -X POST http://localhost:8000/game/action \
  -H 'Content-Type: application/json' \
  -d '{"episode_id":"<episode-id>","type":"semantic","action":"MOVE_UP"}'
```

Other endpoints are `GET /game/state`, `DELETE /game`, and `GET /health`.

## Test

```bash
uv run --extra test pytest
```

To include the native integration test, point it at a compiled Rogue binary:

```bash
TEST_ROGUE_BINARY=/absolute/path/to/rogue uv run --extra test pytest
```

