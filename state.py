"""
state.py — LangGraph AgentState definition.

This TypedDict is the single source of truth that flows through every
node in the job-application graph.  Each node reads what it needs and
writes its results back into the same dict.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Inputs ─────────────────────────────────────────────────────────────
    resume_text: str          # raw text extracted from the uploaded resume
    job_description: str      # raw job-description pasted by the user

    # ── Parsed structure (produced by parse_documents node) ────────────────
    parsed_resume: Dict[str, Any]   # {name, contact, summary, skills, experience, education}

    # ── Gap analysis (produced by analyze_gaps node) ────────────────────────
    matching_skills: List[str]      # skills present in both resume and JD
    skill_gaps: List[str]           # skills required by JD but missing from resume

    # ── Keyword-match scores ─────────────────────────────────────────────────
    keyword_match_before: float     # cosine-similarity score BEFORE rewrite
    keyword_match_after: float      # cosine-similarity score AFTER rewrite

    # ── Generated artefacts ─────────────────────────────────────────────────
    rewritten_cv: str               # full CV text tailored to the JD
    cover_letter: str               # personalised cover letter text

    # ── Learning recommendations (produced by recommend_resources node) ───
    learning_resources: List[Dict[str, str]]
    # each item: {skill, platform, title, url, reason}

    # ── Workflow control ─────────────────────────────────────────────────────
    current_step: str               # label shown in the Streamlit progress bar
    error: Optional[str]            # non-None if a node failed
