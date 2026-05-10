"""
mcp_server.py — MCP server that exposes the job-application tools.

Run this as a subprocess; the LangGraph nodes connect to it via the
MCP stdio transport (stdin / stdout).

Tools exposed
─────────────
  parse_resume            – structure the raw resume text
  analyze_skill_gaps      – identify matching & missing skills
  rewrite_cv              – rewrite CV bullets to align with the JD
  generate_cover_letter   – write a personalised cover letter
  recommend_resources     – suggest learning resources for each gap
  calculate_keyword_match – TF-IDF cosine similarity between two texts
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import anthropic
import numpy as np
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Bootstrap ────────────────────────────────────────────────────────────────
load_dotenv()

_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL   = "claude-sonnet-4-20250514"
MAX_TOK = 4096

app = Server("job-application-agent")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _ask_claude(system: str, user: str) -> str:
    """Single-turn Claude call; returns the assistant text."""
    response = _claude.messages.create(
        model=MODEL,
        max_tokens=MAX_TOK,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _cosine_sim(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity between two strings (0 – 1)."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform([text_a, text_b])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


def _parse_json_block(raw: str) -> Any:
    """Strip markdown fences and parse JSON."""
    clean = raw.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    return json.loads(clean.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="parse_resume",
            description=(
                "Parses raw resume text and returns a structured JSON object "
                "with keys: name, contact, summary, skills (list), "
                "experience (list of {title, company, dates, bullets}), "
                "education (list of {degree, institution, year})."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "resume_text":     {"type": "string"},
                    "job_description": {"type": "string"},
                },
                "required": ["resume_text", "job_description"],
            },
        ),
        types.Tool(
            name="analyze_skill_gaps",
            description=(
                "Compares the resume against the job description and returns "
                "{'matching_skills': [...], 'skill_gaps': [...]}"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "resume_text":     {"type": "string"},
                    "job_description": {"type": "string"},
                },
                "required": ["resume_text", "job_description"],
            },
        ),
        types.Tool(
            name="rewrite_cv",
            description=(
                "Rewrites the resume (summary + experience bullets) to align "
                "with the job description, emphasising matching skills and "
                "addressing skill gaps where possible. Returns the full "
                "rewritten CV as a string."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "resume_text":      {"type": "string"},
                    "job_description":  {"type": "string"},
                    "matching_skills":  {"type": "array", "items": {"type": "string"}},
                    "skill_gaps":       {"type": "array", "items": {"type": "string"}},
                },
                "required": ["resume_text", "job_description", "matching_skills", "skill_gaps"],
            },
        ),
        types.Tool(
            name="generate_cover_letter",
            description=(
                "Generates a professional, personalised cover letter based on "
                "the rewritten CV and job description. Returns the cover letter "
                "as a string."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "resume_text":     {"type": "string"},
                    "job_description": {"type": "string"},
                    "rewritten_cv":    {"type": "string"},
                },
                "required": ["resume_text", "job_description", "rewritten_cv"],
            },
        ),
        types.Tool(
            name="recommend_resources",
            description=(
                "For each skill gap, recommends learning resources "
                "(Coursera, YouTube, official docs). Returns a JSON list of "
                "{skill, platform, title, url, reason}."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "skill_gaps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["skill_gaps"],
            },
        ),
        types.Tool(
            name="calculate_keyword_match",
            description=(
                "Returns the TF-IDF cosine similarity score (0–1) between "
                "two text strings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text_a": {"type": "string"},
                    "text_b": {"type": "string"},
                },
                "required": ["text_a", "text_b"],
            },
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(
    name: str,
    arguments: dict,
) -> list[types.TextContent]:

    if name == "parse_resume":
        system = (
            "You are a professional resume parser. "
            "Return ONLY valid JSON — no markdown, no explanation. "
            "JSON schema: {name, contact:{email,phone,linkedin,location}, "
            "summary, skills:[str], "
            "experience:[{title,company,dates,bullets:[str]}], "
            "education:[{degree,institution,year}]}"
        )
        user = (
            f"Resume:\n{arguments['resume_text']}\n\n"
            f"Job Description (for context):\n{arguments['job_description']}"
        )
        raw  = _ask_claude(system, user)
        data = _parse_json_block(raw)
        return [types.TextContent(type="text", text=json.dumps(data))]

    elif name == "analyze_skill_gaps":
        system = (
            "You are a recruitment expert. Analyse the resume against the job "
            "description and return ONLY valid JSON:\n"
            '{"matching_skills": [str, ...], "skill_gaps": [str, ...]}\n'
            "Be specific — include tools, frameworks, soft-skills, and domain knowledge."
        )
        user = (
            f"Resume:\n{arguments['resume_text']}\n\n"
            f"Job Description:\n{arguments['job_description']}"
        )
        raw  = _ask_claude(system, user)
        data = _parse_json_block(raw)
        return [types.TextContent(type="text", text=json.dumps(data))]

    elif name == "rewrite_cv":
        system = (
            "You are an expert CV writer. Rewrite the candidate's resume so it "
            "is optimised for the target job description. "
            "Rules:\n"
            "• Keep factual accuracy — do NOT invent experience.\n"
            "• Strengthen the summary to mirror the JD's language.\n"
            "• Rewrite bullet points to highlight impact and relevance.\n"
            "• Incorporate matching skills naturally.\n"
            "• For skill gaps the candidate might bridge, add a brief note in the summary.\n"
            "Return the full CV as clean plain text, preserving the original sections."
        )
        user = (
            f"Original Resume:\n{arguments['resume_text']}\n\n"
            f"Job Description:\n{arguments['job_description']}\n\n"
            f"Matching Skills: {', '.join(arguments['matching_skills'])}\n"
            f"Skill Gaps:      {', '.join(arguments['skill_gaps'])}"
        )
        text = _ask_claude(system, user)
        return [types.TextContent(type="text", text=text)]

    elif name == "generate_cover_letter":
        system = (
            "You are an expert career coach. Write a compelling, personalised "
            "cover letter (≈ 350 words, 4 paragraphs) for the candidate. "
            "Structure: opening hook → relevant experience → why this company → CTA. "
            "Tone: professional yet warm. Mirror the JD's language naturally."
        )
        user = (
            f"Rewritten CV:\n{arguments['rewritten_cv']}\n\n"
            f"Job Description:\n{arguments['job_description']}\n\n"
            f"Original Resume (for personal details):\n{arguments['resume_text']}"
        )
        text = _ask_claude(system, user)
        return [types.TextContent(type="text", text=text)]

    elif name == "recommend_resources":
        gaps = arguments["skill_gaps"]
        if not gaps:
            return [types.TextContent(type="text", text="[]")]
        system = (
            "You are a learning advisor. For each skill gap, recommend ONE high-quality "
            "learning resource. Return ONLY a JSON array:\n"
            '[{"skill":"...","platform":"Coursera|YouTube|Official Docs|Udemy",'
            '"title":"...","url":"https://...","reason":"1-sentence why"}]'
        )
        user = f"Skill gaps to address:\n{json.dumps(gaps)}"
        raw  = _ask_claude(system, user)
        data = _parse_json_block(raw)
        return [types.TextContent(type="text", text=json.dumps(data))]

    elif name == "calculate_keyword_match":
        score = _cosine_sim(arguments["text_a"], arguments["text_b"])
        return [types.TextContent(type="text", text=str(round(score, 4)))]

    else:
        raise ValueError(f"Unknown tool: {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
