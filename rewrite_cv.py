"""
nodes/rewrite_cv.py

LangGraph node — Step 3: Rewrite the CV to match the job description.

Reads  : state["resume_text"], state["job_description"],
         state["matching_skills"], state["skill_gaps"]
Writes : state["rewritten_cv"], state["keyword_match_after"], state["current_step"]
"""

from __future__ import annotations

import asyncio

from mcp_client import MCPClient
from state import AgentState


async def _run(state: AgentState) -> AgentState:
    async with MCPClient() as client:
        rewritten_cv = await client.call(
            "rewrite_cv",
            resume_text=state["resume_text"],
            job_description=state["job_description"],
            matching_skills=state.get("matching_skills", []),
            skill_gaps=state.get("skill_gaps", []),
        )

        score_after = await client.call(
            "calculate_keyword_match",
            text_a=rewritten_cv,
            text_b=state["job_description"],
        )

    return {
        **state,
        "rewritten_cv":       rewritten_cv,
        "keyword_match_after": float(score_after),
        "current_step":        "✅ CV rewritten",
        "error":               None,
    }


def rewrite_cv(state: AgentState) -> AgentState:
    try:
        return asyncio.run(_run(state))
    except Exception as exc:
        return {**state, "error": f"[rewrite_cv] {exc}", "current_step": "❌ CV rewrite failed"}
