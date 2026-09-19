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
