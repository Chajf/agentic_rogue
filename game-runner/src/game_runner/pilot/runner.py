"""Run the frozen schedule with one private Rogue environment."""

from contextlib import asynccontextmanager
import asyncio
from hashlib import file_digest
from pathlib import Path
import tempfile
from uuid import uuid4

import httpx

from game_runner.budget import BudgetStop, RequestBudget, TrialBudget
from game_runner.clients.rogue import RogueClient
from game_runner.config import Settings
from game_runner.context.memory import POLICY_VERSION
from game_runner.persistence.repository import GameRepository
from game_runner.pilot.manifest import Manifest
from game_runner.pilot.store import claim_trial, create_series, finalize_series, finalize_trial, series_lock
from game_runner.prompts.system import PROMPT_VERSION
from game_runner.session import run_session

ROGUE_COMMIT = "631bdd551cee75e0afc340a71dd5a207a3006204"


@asynccontextmanager
async def private_rogue(binary: Path, timeout: float):
    # Load the workspace's Rogue package only for pilot execution.
    from rogue_api.api import create_app
    from rogue_api.config import Settings as RogueSettings
    from rogue_api.environment import RogueEnv

    # Rogue 5.4 truncates runtime paths, so keep the temporary pathname short.
    with tempfile.TemporaryDirectory(prefix="rp-", dir="/tmp") as data_dir:
        environment = RogueEnv(RogueSettings(
            rogue_binary=binary, data_dir=Path(data_dir), quiet_window_seconds=0.08,
            action_timeout_seconds=timeout,
        ))
        app = create_app(environment)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://private-rogue",
            ) as http:
                yield RogueClient("http://private-rogue", timeout, client=http), environment.close


async def run_series(
    manifest: Manifest, runtime: Settings, *, binary: Path, revision: str,
    rogue_factory=private_rogue, execute_trial=run_session,
) -> str:
    with binary.open("rb") as source:
        digest = file_digest(source, "sha256").hexdigest()
    provenance = {
        "runner_revision": revision, "rogue_source_pin": ROGUE_COMMIT, "rogue_binary_sha256": digest,
        "prompt_version": PROMPT_VERSION, "context_policy_version": POLICY_VERSION,
        "model_timeout_seconds": runtime.model_timeout_seconds, "rogue_timeout_seconds": runtime.rogue_timeout_seconds,
    }
    repository = GameRepository()
    series = manifest.series_limits
    reason = "schedule_complete"
    with series_lock(manifest.series_id) as connection:
        create_series(connection, manifest, provenance)
        budget = RequestBudget(series.actions, series.model_calls, series.seconds, "series")
        try:
            async with rogue_factory(binary, runtime.rogue_timeout_seconds) as (client, close_episode):
                for trial in manifest.trials:
                    budget.check(action=True, call=True)
                    session_id = uuid4()
                    settings = manifest.settings(runtime, trial)
                    limits = manifest.trial_limits
                    trial_budget = TrialBudget(
                        RequestBudget(limits.actions, limits.model_calls, limits.seconds, "trial"), budget,
                    )
                    claim_trial(connection, manifest, trial, session_id)

                    def on_finish(status: str, detail: str) -> None:
                        finalize_trial(connection, manifest.series_id, trial.trial_id, status, detail)

                    try:
                        await execute_trial(
                            settings, repository, session_id=session_id, client=client,
                            budget=trial_budget, on_finish=on_finish,
                        )
                    finally:
                        await close_episode()
        except BudgetStop as exc:
            reason = str(exc)
        except (KeyboardInterrupt, asyncio.CancelledError):
            reason = "coordinator_interrupted"
            raise
        except BaseException:
            reason = "coordinator_error"
            raise
        finally:
            finalize_series(connection, manifest.series_id, reason)
    return reason
