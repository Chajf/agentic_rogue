"""Dispatch an accepted action and record the observed outcome."""

from dataclasses import asdict

from game_runner.budget import BudgetStop, TrialBudget, within_budget
from game_runner.clients.rogue import Observation, RogueAPIError, RogueClient, RogueTransportError
from game_runner.context.transitions import DecisionTransition, ObservationOutcome
from game_runner.context.working_state import UnlinkedObservation
from game_runner.graph.state import StepState
from game_runner.persistence.repository import GameRepository


class SessionInterrupted(RuntimeError):
    pass


def make_action_node(client: RogueClient, repository: GameRepository, *, budget: TrialBudget | None = None):
    def record_observation(
        state: StepState, event_id: int, action_payload: dict, observation: Observation,
        outcome: ObservationOutcome, *, error: str | None = None,
    ) -> dict:
        before = state["observation"]
        progress = state["progress"].advance(
            event_id, before, action_payload, observation, recovered=outcome == "uncertain",
        )
        persisted_progress = asdict(progress)
        working_state = state.get("working_state")
        transition = None
        if working_state is not None:
            transition = DecisionTransition(
                input_event_id=state["observation_event_id"],
                input_observation=before,
                input_outcome=state["observation_outcome"],
                model_call_id=state["call_id"],
                action=action_payload,
                rationale=state["rationale"],
                output_event_id=event_id,
                output_observation=observation,
                output_outcome=outcome,
            )
            try:
                persisted_progress["working_state_delta"] = working_state.advance(transition, progress)
            except UnlinkedObservation as exc:
                repository.finish_game_event(event_id, "uncertain", error=str(exc))
                raise SessionInterrupted("working-state observation could not be linked") from exc
        repository.finish_game_event(
            event_id, outcome, api_step=observation.api_step,
            http_status=200 if outcome == "succeeded" else None,
            observation=observation.model_dump(mode="json"), progress=persisted_progress, error=error,
        )
        loop = state.get("goal_loop")
        if loop is not None:
            assert transition is not None and working_state is not None
            goal_delta = loop.advance(
                transition, working_state, persisted_progress["working_state_delta"],
                review_requested=state.get("request_goal_review", False),
            )
            if goal_delta is not None:
                repository.record_goal_audit(state["session_id"], event_id, "goal_transition", goal_delta)
        return {
            "next_observation": observation, "next_observation_event_id": event_id,
            "next_observation_outcome": outcome, "next_progress": progress, "next_working_state": working_state,
        }

    async def perform_action(state: StepState) -> dict:
        action = state["action"]
        assert action is not None
        action_payload = action.model_dump(mode="json", exclude_none=True)
        if budget:
            budget.start_action()
        event_id = repository.start_game_event(
            state["session_id"], "action", model_call_id=state["call_id"],
            action_request=action_payload,
        )
        before = state["observation"]
        try:
            observed = await within_budget(budget, lambda: client.action(before.episode_id, action))
        except RogueAPIError as exc:
            repository.finish_game_event(event_id, "failed", http_status=exc.status_code, error=exc.detail)
            raise
        except RogueTransportError as exc:
            try:
                recovered = await within_budget(budget, client.observe)
            except BudgetStop as stop:
                repository.finish_game_event(event_id, "uncertain", error=str(stop))
                raise
            except (RogueAPIError, RogueTransportError, ValueError) as recovery_error:
                repository.finish_game_event(event_id, "uncertain", error=f"{exc}; reconciliation failed: {recovery_error}")
                raise SessionInterrupted("cannot reconcile timed-out action") from recovery_error
            if recovered.episode_id != before.episode_id or recovered.api_step != before.api_step + 1:
                repository.finish_game_event(
                    event_id, "uncertain", api_step=recovered.api_step,
                    error=f"{exc}; episode or step mismatch during reconciliation",
                )
                raise SessionInterrupted("action outcome cannot be confirmed") from exc
            return record_observation(
                state, event_id, action_payload, recovered, "uncertain",
                error=f"{exc}; state recovered but action output was lost",
            )
        except Exception as exc:
            repository.finish_game_event(event_id, "uncertain", error=str(exc))
            raise
        return record_observation(state, event_id, action_payload, observed, "succeeded")

    return perform_action
