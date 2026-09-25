"""Compile a graph that performs exactly one Rogue action."""

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from game_runner.clients.rogue import RogueClient
from game_runner.graph.nodes.step import make_nodes
from game_runner.graph.state import StepState
from game_runner.persistence.repository import GameRepository


def build_graph(
    model: BaseChatModel, client: RogueClient, repository: GameRepository,
    *, context_token_budget: int, max_model_calls: int,
):
    context, invoke_model, validate, perform_action = make_nodes(
        model, client, repository,
        context_token_budget=context_token_budget,
        max_model_calls=max_model_calls,
    )
    graph = StateGraph(StepState)
    graph.add_node("context", context)
    graph.add_node("invoke_model", invoke_model)
    graph.add_node("validate", validate)
    graph.add_node("perform_action", perform_action)
    graph.add_edge(START, "context")
    graph.add_edge("context", "invoke_model")
    graph.add_edge("invoke_model", "validate")
    graph.add_conditional_edges(
        "validate", lambda state: "perform_action" if state.get("action") is not None else "invoke_model",
    )
    graph.add_edge("perform_action", END)
    return graph.compile()
