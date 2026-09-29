"""
agent/app.py

Builds the LangGraph agent (retrieve -> agent -> tools loop) used both by
the data-generation trace collectors (`src/datagen/collect_traces.py`) and
by any future serving layer. This module only assembles the graph; it has
no CLI/menu of its own (see the project root `main.py` for the interactive
data-generation menu).
"""

from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agent import constraints
from src.agent.config import LLM
from src.agent.graph_nodes import call_model, call_tool, should_call_tools, retrieve_context
from src import settings


def build_app(chroma_db, k: int = None, mode: int = 1):
    """Assemble and compile the retrieve -> agent -> tools LangGraph app."""
    from src.agent.tools import TOOLS, TOOLS_BY_NAME

    k = settings.DEFAULT_RETRIEVAL_K if k is None else k

    llm_with_tools = LLM.bind_tools(TOOLS)

    graph = StateGraph(MessagesState)

    # ---------- Nodes ----------
    graph.add_node(
        constraints.NODE_RETRIEVE,
        lambda s: retrieve_context(s, chroma_db, k, mode)
    )

    graph.add_node(
        constraints.NODE_AGENT,
        lambda s: call_model(s, llm_with_tools)
    )

    graph.add_node(
        constraints.NODE_TOOLS,
        lambda s: call_tool(s, TOOLS_BY_NAME)
    )

    # ---------- Edges ----------
    graph.add_edge(START, constraints.NODE_RETRIEVE)
    graph.add_edge(constraints.NODE_RETRIEVE, constraints.NODE_AGENT)

    graph.add_conditional_edges(
        constraints.NODE_AGENT,
        should_call_tools,
        {
            "tool_calls": constraints.NODE_TOOLS,
            "no_tools": END,
        },
    )

    graph.add_edge(constraints.NODE_TOOLS, constraints.NODE_AGENT)

    # ---------- Checkpoint ----------
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
