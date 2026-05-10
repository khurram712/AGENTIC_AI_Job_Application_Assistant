"""
mcp_client.py — Async MCP client used by the LangGraph nodes.

Spawns mcp_server.py as a subprocess (stdio transport) and keeps the
session alive for the duration of one agent run.

Usage
─────
    async with MCPClient() as client:
        result = await client.call("parse_resume", resume_text=..., job_description=...)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")


class MCPClient:
    """Thin async wrapper around an MCP ClientSession."""

    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._ctx = None

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "MCPClient":
        server_params = StdioServerParameters(
            command=sys.executable,   # same Python interpreter
            args=[SERVER_SCRIPT],
            env={**os.environ},       # forward ANTHROPIC_API_KEY etc.
        )
        self._ctx = stdio_client(server_params)
        read, write = await self._ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.__aexit__(*args)
        if self._ctx:
            await self._ctx.__aexit__(*args)

    # ── Public API ────────────────────────────────────────────────────────────

    async def call(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Call a tool on the MCP server and return the parsed result.

        • If the server returns a JSON string → parse and return the object.
        • Otherwise return the raw string.
        """
        if self._session is None:
            raise RuntimeError("MCPClient must be used as an async context manager")

        result = await self._session.call_tool(tool_name, arguments=kwargs)

        # result.content is a list of TextContent / ImageContent blocks
        raw = "\n".join(
            block.text for block in result.content if hasattr(block, "text")
        ).strip()

        # Attempt JSON decode
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    async def list_tools(self) -> list[str]:
        """Return the names of all tools registered on the server."""
        if self._session is None:
            raise RuntimeError("MCPClient must be used as an async context manager")
        resp = await self._session.list_tools()
        return [t.name for t in resp.tools]


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run a coroutine synchronously (used by Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def run_sync(coro: Any) -> Any:
    """Execute an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # inside an existing event loop (e.g. Jupyter) — use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
