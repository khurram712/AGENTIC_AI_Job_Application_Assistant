"""
mcp_server.py  –  MCP server exposing job-application tools (OpenAI backend).

Launched as a subprocess by mcp_client.py; communicates via stdin / stdout
(MCP stdio transport).

Tools
─────
  parse_resume            – structure the raw resume text
  analyze_skill_gaps      – matching skills vs missing skills
  rewrite_cv              – tailor CV bullets/summary to the JD
  generate_cover_letter   – write a personalised cover letter
  recommend_resources     – Coursera / YouTube resources per skill gap
  calculate_keyword_match – TF-IDF cosine similarity between two texts
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")

app = Server("job-application-agent-openai")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chat(system: str, user: str, temperature: float = 0.3) -> str:
    """Single-turn OpenAI chat completion; returns assistant text."""
    resp = _openai.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _cosine_sim(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity (0–1)."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vec   = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform([text_a, text_b])
    return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


def _parse_json(raw: str) -> Any:
    """Strip markdown code fences then parse JSON."""
    clean = raw.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    return json.loads(clean.strip())


# ─────────────────────────────────────────────────────────────────────────────
# MCP tool registry
# ─────────────────────────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="parse_resume",
            description=(
                "Parses raw resume text and returns a structured JSON object: "
                "{name, contact:{email,phone,linkedin,location}, summary, "
                "skills:[str], experience:[{title,company,dates,bullets:[str]}], "
                "education:[{degree,institution,year}]}"
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
                "Compares resume against the job description and returns "
                '{"matching_skills":[...], "skill_gaps":[...]}.'
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
                "Rewrites the resume summary and experience bullets to align "
                "with the JD, emphasising matching skills without fabricating "
                "experience. Returns the full rewritten CV as plain text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "resume_text":     {"type": "string"},
                    "job_description": {"type": "string"},
                    "matching_skills": {"type": "array", "items": {"type": "string"}},
                    "skill_gaps":      {"type": "array", "items": {"type": "string"}},
                },
                "required": ["resume_text", "job_description", "matching_skills", "skill_gaps"],
            },
        ),
        types.Tool(
            name="generate_cover_letter",
            description=(
                "Generates a professional, personalised cover letter (~350 words) "
                "based on the rewritten CV and JD. Returns cover letter text."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "resume_text":    {"type": "string"},
                    "job_description":{"type": "string"},
                    "rewritten_cv":   {"type": "string"},
                },
                "required": ["resume_text", "job_description", "rewritten_cv"],
            },
        ),
        types.Tool(
            name="recommend_resources",
            description=(
                "For each skill gap, recommends one high-quality learning resource. "
                'Returns a JSON list: [{skill, platform, title, url, reason}].'
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
            description="TF-IDF cosine similarity (0–1) between two text strings.",
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
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── parse_resume ──────────────────────────────────────────────────────────
    if name == "parse_resume":
        system = (
            "You are a professional resume parser. "
            "Return ONLY valid JSON, no markdown fences, no explanation. "
            "Schema: {name:str, contact:{email,phone,linkedin,location}, "
            "summary:str, skills:[str], "
            "experience:[{title,company,dates,bullets:[str]}], "
            "education:[{degree,institution,year}]}"
        )
        user = (
            f"RESUME:\n{arguments['resume_text']}\n\n"
            f"JOB DESCRIPTION (context only):\n{arguments['job_description']}"
        )
        raw  = _chat(system, user)
        data = _parse_json(raw)
        return [types.TextContent(type="text", text=json.dumps(data))]

    # ── analyze_skill_gaps ────────────────────────────────────────────────────
    elif name == "analyze_skill_gaps":
        system = (
            "You are a senior technical recruiter. "
            "Compare the resume against the job description. "
            'Return ONLY valid JSON: {"matching_skills":[str,...], "skill_gaps":[str,...]}. '
            "Be specific — include tools, frameworks, soft-skills, certifications, and domain knowledge."
        )
        user = (
            f"RESUME:\n{arguments['resume_text']}\n\n"
            f"JOB DESCRIPTION:\n{arguments['job_description']}"
        )
        raw  = _chat(system, user)
        data = _parse_json(raw)
        return [types.TextContent(type="text", text=json.dumps(data))]

    # ── rewrite_cv ────────────────────────────────────────────────────────────
    elif name == "rewrite_cv":
        system = (
            "You are an expert CV writer. Rewrite the candidate's resume to be "
            "optimised for the target job description.\n"
            "Rules:\n"
            "• Never invent or exaggerate experience — stay factually accurate.\n"
            "• Rewrite the professional summary to mirror the JD's language.\n"
            "• Rephrase experience bullets to highlight impact and relevance.\n"
            "• Naturally incorporate matching skills throughout.\n"
            "• For skill gaps the candidate could credibly bridge, add a brief note in the summary.\n"
            "Return the full CV as clean plain text, preserving all original sections."
        )
        user = (
            f"ORIGINAL RESUME:\n{arguments['resume_text']}\n\n"
            f"JOB DESCRIPTION:\n{arguments['job_description']}\n\n"
            f"MATCHING SKILLS: {', '.join(arguments['matching_skills'])}\n"
            f"SKILL GAPS:      {', '.join(arguments['skill_gaps'])}"
        )
        text = _chat(system, user, temperature=0.4)
        return [types.TextContent(type="text", text=text)]

    # ── generate_cover_letter ─────────────────────────────────────────────────
    elif name == "generate_cover_letter":
        system = (
            "You are an expert career coach. Write a compelling, personalised cover letter "
            "(≈350 words, 4 paragraphs) for the candidate.\n"
            "Structure: (1) strong opening hook, (2) relevant experience & achievements, "
            "(3) why this company / role, (4) confident call-to-action.\n"
            "Tone: professional yet warm. Mirror the JD language naturally. "
            "Do NOT use generic filler phrases."
        )
        user = (
            f"REWRITTEN CV:\n{arguments['rewritten_cv']}\n\n"
            f"JOB DESCRIPTION:\n{arguments['job_description']}\n\n"
            f"ORIGINAL RESUME (for personal details):\n{arguments['resume_text']}"
        )
        text = _chat(system, user, temperature=0.5)
        return [types.TextContent(type="text", text=text)]

    # ── recommend_resources ───────────────────────────────────────────────────
    elif name == "recommend_resources":
        gaps = arguments["skill_gaps"]
        if not gaps:
            return [types.TextContent(type="text", text="[]")]
        system = (
            "You are a learning advisor. For each skill gap, recommend ONE high-quality "
            "learning resource from Coursera, YouTube, or official documentation.\n"
            "Return ONLY a JSON array (no fences):\n"
            '[{"skill":"...","platform":"Coursera|YouTube|Official Docs|Udemy",'
            '"title":"...","url":"https://...","reason":"1-sentence why this resource"}]'
        )
        user = f"Skill gaps:\n{json.dumps(gaps)}"
        raw  = _chat(system, user)
        data = _parse_json(raw)
        return [types.TextContent(type="text", text=json.dumps(data))]

    # ── calculate_keyword_match ───────────────────────────────────────────────
    elif name == "calculate_keyword_match":
        score = _cosine_sim(arguments["text_a"], arguments["text_b"])
        return [types.TextContent(type="text", text=str(round(score, 4)))]

    else:
        raise ValueError(f"Unknown tool: {name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point (stdio transport)
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())