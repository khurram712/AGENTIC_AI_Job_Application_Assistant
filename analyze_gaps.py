"""
nodes/analyze_gaps.py

LangGraph node — Step 2: Identify matching skills and skill gaps.

Reads  : state["resume_text"], state["job_description"]
Writes : state["matching_skills"], state["skill_gaps"], state["current_step"]
"""

from __future__ import annotations

import asyncio

from mcp_client import MCPClient
from state import AgentState


async def _run(state: AgentState) -> AgentState:
    async with MCPClient() as client:
        result = await client.call(
            "analyze_skill_gaps",
            resume_text=state["resume_text"],
            job_description=state["job_description"],
        )

    matching = result.get("matching_skills", [])
    gaps     = result.get("skill_gaps", [])

    return {
        **state,
        "matching_skills": matching,
        "skill_gaps":      gaps,
        "current_step":    f"✅ Gap analysis done — {len(gaps)} gap(s) found",
        "error":           None,
    }


def analyze_gaps(state: AgentState) -> AgentState:
    try:
        return asyncio.run(_run(state))
    except Exception as exc:
        return {**state, "error": f"[analyze_gaps] {exc}", "current_step": "❌ Gap analysis failed"}
