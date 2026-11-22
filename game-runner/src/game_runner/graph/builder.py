"""Compile a graph that performs exactly one Rogue action."""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from game_runner.budget import TrialBudget
from game_runner.clients.rogue import RogueClient
from game_runner.context.planning import ContextSettings
from game_runner.graph.nodes.goals import make_goal_nodes
from game_runner.graph.nodes.step import make_nodes
from game_runner.graph.state import StepState
from game_runner.persistence.repository import GameRepository


def build_graph(
    model: BaseChatModel, client: RogueClient, repository: GameRepository,
    *, context_token_budget: int, max_model_calls: int, history_recent_interactions: int = 8,
    context_settings: ContextSettings | None = None, summary_model=None,
    budget: TrialBudget | None = None,
    managed_goals: bool = False,
):
    settings = context_settings or ContextSettings(context_token_budget, history_recent_interactions)
    context, invoke_model, validate, perform_action = make_nodes(
        model, client, repository,
        max_model_calls=max_model_calls,
        context_settings=settings,
        summary_model=summary_model,
        budget=budget,
    )
    graph = StateGraph(StepState)
    graph.add_node("context", context)
    graph.add_node("invoke_model", invoke_model)
    graph.add_node("validate", validate)
    graph.add_node("perform_action", perform_action)
    if managed_goals:
        assess_goal, select_goal = make_goal_nodes(model, repository, context_settings=settings, budget=budget)
        graph.add_node("assess_goal", assess_goal)
        graph.add_node("select_goal", select_goal)
        graph.add_edge(START, "assess_goal")
        graph.add_conditional_edges(
            "assess_goal", lambda state: "select_goal" if state.get("goal_review") is not None else "context",
        )
        graph.add_edge("select_goal", "context")
    else:
        graph.add_edge(START, "context")
    graph.add_edge("context", "invoke_model")
    graph.add_edge("invoke_model", "validate")
    graph.add_conditional_edges(
        "validate", lambda state: "perform_action" if state.get("action") is not None else "invoke_model",
    )
    graph.add_edge("perform_action", END)
    return graph.compile()
