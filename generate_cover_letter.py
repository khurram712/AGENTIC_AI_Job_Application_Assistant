"""
nodes/generate_cover_letter.py

LangGraph node — Step 4: Generate a personalised cover letter.

Reads  : state["resume_text"], state["job_description"], state["rewritten_cv"]
Writes : state["cover_letter"], state["current_step"]
"""

from __future__ import annotations

import asyncio

from mcp_client import MCPClient
from state import AgentState


async def _run(state: AgentState) -> AgentState:
    async with MCPClient() as client:
        cover_letter = await client.call(
            "generate_cover_letter",
            resume_text=state["resume_text"],
            job_description=state["job_description"],
            rewritten_cv=state["rewritten_cv"],
        )

    return {
        **state,
        "cover_letter":  cover_letter,
        "current_step":  "✅ Cover letter generated",
        "error":         None,
    }


def generate_cover_letter(state: AgentState) -> AgentState:
    try:
        return asyncio.run(_run(state))
    except Exception as exc:
        return {**state, "error": f"[generate_cover_letter] {exc}", "current_step": "❌ Cover letter failed"}
