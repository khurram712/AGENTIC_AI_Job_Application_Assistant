"""
nodes/parse_documents.py

LangGraph node — Step 1: Parse the resume and compute the BEFORE keyword-match score.

Reads  : state["resume_text"], state["job_description"]
Writes : state["parsed_resume"], state["keyword_match_before"], state["current_step"]
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from mcp_client import MCPClient
from state import AgentState


async def _run(state: AgentState) -> AgentState:
    async with MCPClient() as client:
        # 1. Structural parse
        parsed = await client.call(
            "parse_resume",
            resume_text=state["resume_text"],
            job_description=state["job_description"],
        )

        # 2. Keyword match BEFORE rewrite
        score_before = await client.call(
            "calculate_keyword_match",
            text_a=state["resume_text"],
            text_b=state["job_description"],
        )

    return {
        **state,
        "parsed_resume":       parsed,
        "keyword_match_before": float(score_before),
        "current_step":        "✅ Resume parsed",
        "error":               None,
    }


def parse_documents(state: AgentState) -> AgentState:
    """Synchronous wrapper called by LangGraph."""
    try:
        return asyncio.run(_run(state))
    except Exception as exc:
        return {**state, "error": f"[parse_documents] {exc}", "current_step": "❌ Parsing failed"}
