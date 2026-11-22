"""Goal review and model selection, scoped to the current saved observation."""

from dataclasses import replace
import time

from langchain_core.messages import AIMessage

from game_runner.budget import BudgetStop, TrialBudget, within_budget
from game_runner.context.budget import count_messages
from game_runner.context.goals import parse_goal_reply
from game_runner.context.messages import build_goal_selection_messages
from game_runner.context.planning import ContextSettings
from game_runner.context.working_state import UnlinkedObservation
from game_runner.context.working_views import actor_view
from game_runner.graph.state import StepState
from game_runner.persistence.repository import GameRepository


def make_goal_nodes(
    model, repository: GameRepository, *, context_settings: ContextSettings, budget: TrialBudget | None,
):
    def assess_goal(state: StepState) -> dict:
        working = state.get("working_state")
        loop = state.get("goal_loop")
        if working is None or loop is None:
            raise ValueError("managed goals require episode working state")
        observation = state["observation"]
        if (
            working.through.event_id != state["observation_event_id"]
            or working.through.api_step != observation.api_step
            or working.episode_id != observation.episode_id
            or working.outcome != state["observation_outcome"]
        ):
            raise UnlinkedObservation("working state is outside the selector's input boundary")
        return {"goal_review": loop.assess(observation, working)}

    async def select_goal(state: StepState) -> dict:
        review, loop, working = state["goal_review"], state["goal_loop"], state["working_state"]
        assert review is not None and loop is not None and working is not None
        observation = state["observation"]
        session_id, event_id = state["session_id"], state["observation_event_id"]
        audit = {
            "observation_event_id": event_id, "map_instance": working.map_instance,
            "reasons": review.reasons, "status": "requested",
            "goal_before": loop.goal.record() if loop.goal else None,
            "candidates": [
                {"candidate_id": item["candidate_id"], "kind": item["kind"], "position": item["position"],
                 "source_event_id": item["source"]["event_id"]}
                for item in review.candidates
            ],
        }

        def save_audit(status: str, **details) -> None:
            audit.update(status=status, **details)
            repository.record_goal_audit(session_id, event_id, "goal_review", audit.copy())

        save_audit("requested")
        try:
            if budget:
                budget.check(action=True)
                if not budget.has_call_capacity(reserve=1):
                    save_audit("skipped", detail="reserved_actor_call")
                    loop.record_review(observation, working, review)
                    return {}

            tiles = (actor_view(working) or {}).get("tiles", [])
            messages = None
            count = None
            delivered = review
            for limit in (None, 3, 1, 0):
                candidates = review.candidates
                if limit is not None:
                    candidates = [
                        {key: value for key, value in item.items() if key != "route"}
                        for item in candidates[:limit]
                    ]
                view = {**review.view, "opportunities": candidates}
                evidence = {
                    **view, "review_reasons": audit["reasons"], "goal": audit["goal_before"],
                    "nearby_tiles": tiles[:6] if limit is None else tiles[:3],
                }
                messages = build_goal_selection_messages(observation, evidence)
                count = count_messages(model, messages)
                if count.count + context_settings.actor_output_tokens <= context_settings.token_budget:
                    delivered = replace(review, view=view)
                    break
            else:
                save_audit("skipped", detail="selector_context_limit")
                loop.record_review(observation, working, review)
                return {}

            if budget:
                budget.start_call()
        except BudgetStop as exc:
            save_audit("budget_stop", detail=str(exc))
            raise

        assert messages is not None and count is not None
        audit["delivered_candidate_ids"] = [item["candidate_id"] for item in delivered.candidates]
        call_id = repository.start_model_call(
            session_id, [{"role": message.type, "content": message.content} for message in messages],
            observation_event_id=event_id, purpose="goal_selection", decision_metadata=audit.copy(),
            input_token_count=count.count, input_token_count_method=count.method,
        )
        audit["selector_call_id"] = call_id
        started = time.monotonic()
        try:
            response = await within_budget(budget, lambda: model.ainvoke(messages))
            if not isinstance(response, AIMessage):
                raise TypeError("goal selector did not return AIMessage")
        except Exception as exc:
            reason = str(exc) if isinstance(exc, BudgetStop) else type(exc).__name__
            save_audit("budget_stop" if isinstance(exc, BudgetStop) else "error", detail=reason)
            repository.finish_model_call(
                call_id, "error", validation_error=reason, decision_metadata=audit.copy(),
                duration_ms=round((time.monotonic() - started) * 1000),
            )
            if isinstance(exc, BudgetStop):
                raise
        else:
            elapsed = round((time.monotonic() - started) * 1000)
            raw = response.model_dump(mode="json")
            error = None
            try:
                selection = loop.choose(parse_goal_reply(response), working, delivered)
            except ValueError as exc:
                error = str(exc)
                save_audit("invalid")
            else:
                save_audit("resolved", selection=selection)
            repository.finish_model_call(
                call_id, "invalid" if error else "valid", raw_response=raw, validation_error=error,
                response_metadata=raw.get("response_metadata"), usage_metadata=raw.get("usage_metadata"),
                duration_ms=elapsed, decision_metadata=audit.copy(),
            )

        loop.record_review(observation, working, review)
        return {}

    return assess_goal, select_goal
