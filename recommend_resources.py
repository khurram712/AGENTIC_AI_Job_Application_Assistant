"""
nodes/recommend_resources.py

LangGraph node — Step 5: Recommend learning resources for each skill gap.

Reads  : state["skill_gaps"]
Writes : state["learning_resources"], state["current_step"]
"""

from __future__ import annotations

import asyncio

from mcp_client import MCPClient
from state import AgentState


async def _run(state: AgentState) -> AgentState:
    gaps = state.get("skill_gaps", [])

    if not gaps:
        return {
            **state,
            "learning_resources": [],
            "current_step": "✅ No gaps — no resources needed",
            "error": None,
        }

    async with MCPClient() as client:
        resources = await client.call(
            "recommend_resources",
            skill_gaps=gaps,
        )

    # Ensure it's a list (defensive parse)
    if isinstance(resources, str):
        import json
        resources = json.loads(resources)

    return {
        **state,
        "learning_resources": resources,
        "current_step":       f"✅ {len(resources)} resource(s) recommended",
        "error":              None,
    }


def recommend_resources(state: AgentState) -> AgentState:
    try:
        return asyncio.run(_run(state))
    except Exception as exc:
        return {**state, "error": f"[recommend_resources] {exc}", "current_step": "❌ Recommendations failed"}
