# src/graph.py

from typing import Literal

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from states import TravelChatBotState
from agents import (
    master_agent,
    master_response_agent,
    activities_agent,
    logistics_agent,
    update_history_summary,
)

checkpointer = MemorySaver()


def route_from_master(
    state: TravelChatBotState,
) -> Literal["activities", "logistics", "master_response"]:
    """
    Decide where to go after the master_agent.

    Uses the 'master_route' field set by the master_agent, which will be
    one of: "activities", "logistics", "master_response".
    """
    route = state.get("master_route") or "master_response"
    if route not in ("activities", "logistics", "master_response"):
        route = "master_response"
    return route  # type: ignore[return-value]


def _next_specialist_or_master(state: TravelChatBotState) -> Literal[
    "activities", "logistics", "master_response"
]:
    """
    Helper to move from one specialist to the next (for full plans),
    or back to master_response when done.

    It reads:
      - metadata["specialist_targets"]: list of strings like ["activities", "logistics"]
      - metadata["specialist_index"]: integer index of the last specialist run

    After each specialist node finishes, we bump the index:
      - If there is another specialist in the list, go there.
      - Otherwise, go to "master_response".
    """
    metadata = state.get("metadata") or {}
    targets = metadata.get("specialist_targets") or []
    idx = metadata.get("specialist_index", 0)

    # Move to the next specialist
    idx += 1
    metadata["specialist_index"] = idx
    state["metadata"] = metadata

    if idx < len(targets):
        next_target = targets[idx]
        if next_target in ("activities", "logistics"):
            return next_target  # type: ignore[return-value]

    return "master_response"  # type: ignore[return-value]


def route_from_activities(
    state: TravelChatBotState,
) -> Literal["activities", "logistics", "master_response"]:
    """
    Decide where to go after the activities_agent.
    Typically:
      - If plan_full and logistics is still remaining -> "logistics"
      - Else -> "master_response"
    """
    return _next_specialist_or_master(state)


def route_from_logistics(
    state: TravelChatBotState,
) -> Literal["activities", "logistics", "master_response"]:
    """
    Decide where to go after the logistics_agent.
    Typically:
      - If plan_full and activities is still remaining (rare) -> "activities"
      - Else -> "master_response"
    """
    return _next_specialist_or_master(state)


def build_graph():
    """
    Build and compile the LangGraph application for the travel assistant.
    """
    builder = StateGraph(TravelChatBotState)

    # Register nodes
    builder.add_node("master", master_agent)
    builder.add_node("activities", activities_agent)
    builder.add_node("logistics", logistics_agent)
    builder.add_node("master_response", master_response_agent)
    builder.add_node("update_history", update_history_summary)

    # Entry point
    builder.add_edge(START, "master")

    # After master: route based on master_route
    builder.add_conditional_edges(
        "master",
        route_from_master,
        path_map={
            "activities": "activities",
            "logistics": "logistics",
            "master_response": "master_response",
        },
    )

    # After activities: maybe go to logistics (for full plan) or finish
    builder.add_conditional_edges(
        "activities",
        route_from_activities,
        path_map={
            "activities": "activities",
            "logistics": "logistics",
            "master_response": "master_response",
        },
    )

    # After logistics: maybe go to activities (rare) or finish
    builder.add_conditional_edges(
        "logistics",
        route_from_logistics,
        path_map={
            "activities": "activities",
            "logistics": "logistics",
            "master_response": "master_response",
        },
    )

    # After master_response: update history, then end
    builder.add_edge("master_response", "update_history")
    builder.add_edge("update_history", END)

    app = builder.compile(checkpointer=checkpointer)
    return app
