"""LangGraph wiring for the AI Architect multi-agent system.

    START -> extractor -> clarifier --(questions)--> END   (wait for the user)
                              |
                              +--(nothing to ask)--> planner -> retriever -> final -> END

Each user turn re-enters at the extractor; state is kept per thread by the
checkpointer, so answers to clarifying questions accumulate.
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Dict, List, Mapping, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .agents import (
    make_clarifier_node,
    make_extractor_node,
    make_final_node,
    make_planner_node,
    make_retriever_node,
)
from .routing import route_after_clarifier, route_after_planner
from .state import ArchitectState


class GraphState(ArchitectState, total=False):
    messages: Annotated[List[Any], add_messages]


def build_graph(
    llm: Optional[Any] = None,
    tools: Optional[Mapping[str, Callable[..., Dict[str, Any]]]] = None,
    checkpointer: Optional[Any] = None,
):
    """Compile the graph. `llm` and `tools` can be injected for testing."""
    if llm is None:
        from .config import get_llm

        llm = get_llm()
    if tools is None:
        from .tools import DEFAULT_TOOLS

        tools = DEFAULT_TOOLS

    workflow = StateGraph(GraphState)
    workflow.add_node("extractor", make_extractor_node(llm))
    workflow.add_node("clarifier", make_clarifier_node(llm))
    workflow.add_node("planner", make_planner_node(llm))
    workflow.add_node("retriever", make_retriever_node(tools))
    workflow.add_node("final", make_final_node(llm))

    workflow.add_edge(START, "extractor")
    workflow.add_edge("extractor", "clarifier")
    workflow.add_conditional_edges("clarifier", route_after_clarifier, {"ask_user": END, "planner": "planner"})
    workflow.add_conditional_edges("planner", route_after_planner, {"retriever": "retriever", "final": "final"})
    workflow.add_edge("retriever", "final")
    workflow.add_edge("final", END)

    return workflow.compile(checkpointer=checkpointer or MemorySaver())
