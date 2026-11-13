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

## Offline rule feasibility

P4 reads one finalized P3 bundle per run. Create the output directory first
and use fresh output filenames for each stage. The natural development bundle
can be reported even if it has zero eligible cases.

```bash
mkdir -p reports/rule-feasibility
uv run --locked --all-packages python -m game_runner.rule_feasibility prepare \
  --corpus reports/pilot-p3-development --split development --origin natural \
  --output reports/rule-feasibility/cases.json
uv run --locked --all-packages python -m game_runner.rule_feasibility report \
  --corpus reports/pilot-p3-development --split development \
  --cases reports/rule-feasibility/cases.json \
  --output reports/rule-feasibility/report.json \
  --markdown reports/rule-feasibility/report.md
```

For a corpus with eligible source cases, use these additional stages:

1. Run `extract` with the same corpus and split, `--cases cases.json`,
   `--model-config model.json`, `--max-calls N`, `--seconds N`, and
   `--output candidates.json`. A minimal model file is
   `{"model_name":"your-provider-model","temperature":0,"output_tokens":2048}`.
   Optional input and output rates are expressed in USD per million tokens.
   Set the provider endpoint and credentials in the process environment.
2. Prepare a human review and run `freeze` with the same corpus and split,
   `--cases cases.json`, `--candidates candidates.json`, `--review review.json`,
   `--model-config model.json`, and `--output frozen.json`.
3. Run `probe --frozen frozen.json --max-calls N --seconds N --output probes.json`.
   Keep the generated checksum sidecar with the frozen specification. Reserve
   three calls per probe state.
4. Pass `--candidates candidates.json --frozen frozen.json --results probes.json`
   to `report` to include the audit and paired results.

The review file uses `schema_version: 1`, `corpus_sha256`, `candidates_sha256`,
`reviewer`, `approved: true`, `audits` and `probes`. For each audited
`candidate_id`, supply `factual_correctness` (correct, incorrect, unknown),
`evidential_support` (supported, refuted, insufficient, conflicting), `scope`
(appropriate, too_broad, too_narrow, unknown), `applicability` (applicable,
conditional, inapplicable, unknown), and `utility` (useful, harmful, mixed,
unknown). Record a `rationale`, the verbatim `original` claim, a `corrected`
claim, and `corrected_scope` (map_instance, episode, conditional, general).
Include `corrected_conditions`, `corrected_exceptions`, `corroborating_event_ids`,
and `counterevidence_event_ids` where applicable. For each probe, record the
`candidate_id`, `pre_event_id`, `applicability` (matching or out_of_scope),
the `acceptable`, `harmful`, and `indeterminate` action lists in the existing
action format, and a `rationale`. Actions absent from the rubric are scored
indeterminate. The reviewer can include `cost_viable_families` to record a
follow-on budget judgment.

Use separate `scripted` or `injected` case artifacts for harness checks.
Reports keep those denominators separate from `natural` evidence.
