"""
graph.py — LangGraph workflow for the AI Job Application Agent.

Graph topology
──────────────

  START
    │
    ▼
  parse_documents          (Step 1) parse resume + compute before-score
    │
    ▼
  analyze_gaps             (Step 2) identify matching skills & gaps
    │
    ▼
  rewrite_cv               (Step 3) tailor CV + compute after-score
    │
    ▼
  generate_cover_letter    (Step 4) write personalised cover letter
    │
    ▼
  recommend_resources      (Step 5) suggest learning paths
    │
    ▼
  END

Each node is a plain Python function that receives and returns AgentState.
Conditional edges route to END early if a node writes a non-None "error" key.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from analyze_gaps        import analyze_gaps
from generate_cover_letter import generate_cover_letter
from parse_documents     import parse_documents
from recommend_resources import recommend_resources
from rewrite_cv          import rewrite_cv
from state import AgentState


# ── Guard: stop the graph if a node sets state["error"] ──────────────────────

def _should_continue(state: AgentState) -> str:
    return "end" if state.get("error") else "continue"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Add nodes
    g.add_node("parse_documents",      parse_documents)
    g.add_node("analyze_gaps",         analyze_gaps)
    g.add_node("rewrite_cv",           rewrite_cv)
    g.add_node("generate_cover_letter", generate_cover_letter)
    g.add_node("recommend_resources",  recommend_resources)

    # Entry edge
    g.add_edge(START, "parse_documents")

    # Conditional edges — abort on error, otherwise continue
    for src, dst in [
        ("parse_documents",       "analyze_gaps"),
        ("analyze_gaps",          "rewrite_cv"),
        ("rewrite_cv",            "generate_cover_letter"),
        ("generate_cover_letter", "recommend_resources"),
    ]:
        g.add_conditional_edges(
            src,
            _should_continue,
            {"continue": dst, "end": END},
        )

    # Final edge
    g.add_edge("recommend_resources", END)

    return g


# ── Compiled graph (singleton) ────────────────────────────────────────────────

_compiled: StateGraph | None = None


def get_compiled_graph() -> StateGraph:
    global _compiled
    if _compiled is None:
        _compiled = build_graph().compile()
    return _compiled


# ── Convenience runner ────────────────────────────────────────────────────────

def run_agent(resume_text: str, job_description: str) -> AgentState:
    """
    Run the full agent pipeline synchronously.

    Parameters
    ----------
    resume_text     : raw text of the candidate's resume
    job_description : raw text of the target job description

    Returns
    -------
    Final AgentState after all nodes have run (or after an error node).
    """
    initial_state: AgentState = {
        "resume_text":      resume_text,
        "job_description":  job_description,
        "current_step":     "🚀 Starting agent…",
        "error":            None,
    }

    graph = get_compiled_graph()
    final = graph.invoke(initial_state)
    return final


# ── Streaming runner (yields intermediate states) ─────────────────────────────

def stream_agent(resume_text: str, job_description: str):
    """
    Generator that yields (node_name, AgentState) after each node completes.
    Use this in Streamlit to update the UI in real time.
    """
    initial_state: AgentState = {
        "resume_text":      resume_text,
        "job_description":  job_description,
        "current_step":     "🚀 Starting agent…",
        "error":            None,
    }

    graph = get_compiled_graph()
    for event in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_state in event.items():
            yield node_name, node_state
