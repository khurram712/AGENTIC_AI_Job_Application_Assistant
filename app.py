"""
app.py — Streamlit front-end for the AI-Powered Job Application Agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Add project root to path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from graph import stream_agent
from pdf_parser import extract_text
from doc_generator import generate_cv_docx, generate_cover_letter_docx

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Job Application Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Overall background */
.stApp { background-color: #f8f9fa; }

/* Card style */
.result-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #1A73E8;
}

/* Score pill */
.score-pill {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 0.2rem;
}
.score-low  { background: #fce8e6; color: #c5221f; }
.score-mid  { background: #fef7e0; color: #f29900; }
.score-high { background: #e6f4ea; color: #137333; }

/* Skill badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.82rem;
    margin: 0.15rem;
    font-weight: 500;
}
.badge-green { background: #e6f4ea; color: #137333; }
.badge-red   { background: #fce8e6; color: #c5221f; }

/* Agent step log */
.step-log {
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    color: #3c4043;
    background: #f1f3f4;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
}

/* Resource card */
.resource-card {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    border: 1px solid #e8eaed;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score < 0.3:
        return "score-low"
    if score < 0.6:
        return "score-mid"
    return "score-high"


def score_label(score: float) -> str:
    pct = int(score * 100)
    if pct < 30:
        return f"{pct}% 🔴"
    if pct < 60:
        return f"{pct}% 🟡"
    return f"{pct}% 🟢"


def badges(items: list[str], css_class: str) -> str:
    return " ".join(f'<span class="badge {css_class}">{s}</span>' for s in items)


PLATFORM_ICONS = {
    "coursera":      "🎓",
    "youtube":       "▶️",
    "udemy":         "📚",
    "official docs": "📄",
    "docs":          "📄",
}


def platform_icon(platform: str) -> str:
    for key, icon in PLATFORM_ICONS.items():
        if key in platform.lower():
            return icon
    return "🔗"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/resume.png",
        width=60,
    )
    st.title("Job Application\nAgent")
    st.caption("Powered by LangGraph + MCP + Open AI")

    st.divider()
    st.subheader("⚙️ Configuration")

    api_key = st.text_input(
        "OPENAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your OPENAI API key (stored only in this session)",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.subheader("🏗️ Architecture")
    st.markdown("""
**LangGraph Graph Nodes:**
1. `parse_documents` — Extract & structure resume
2. `analyze_gaps` — Identify skill gaps
3. `rewrite_cv` — Tailor CV to JD
4. `generate_cover_letter` — Write cover letter
5. `recommend_resources` — Suggest learning paths

**MCP Tools (via stdio transport):**
- `parse_resume`
- `analyze_skill_gaps`
- `rewrite_cv`
- `generate_cover_letter`
- `recommend_resources`
- `calculate_keyword_match`
    """)

    st.divider()
    st.caption("Built with LangGraph · MCP · Claude API")


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("💼 AI-Powered Job Application Agent")
st.markdown(
    "Upload your resume and paste a job description. "
    "The agent will tailor your CV, write a cover letter, identify skill gaps, "
    "and recommend learning resources — all autonomously."
)

st.divider()

col_l, col_r = st.columns([1, 1], gap="large")

with col_l:
    st.subheader("📄 Your Resume")
    resume_file = st.file_uploader(
        "Upload resume (PDF or TXT)",
        type=["pdf", "txt", "md"],
        help="PDF or plain-text resume",
    )

    resume_text_raw = ""
    if resume_file:
        try:
            resume_text_raw = extract_text(resume_file.read(), resume_file.name)
            st.success(f"✅ Loaded **{resume_file.name}** ({len(resume_text_raw):,} chars)")
            with st.expander("Preview extracted text"):
                st.text(resume_text_raw[:1200] + ("…" if len(resume_text_raw) > 1200 else ""))
        except Exception as e:
            st.error(f"Could not parse file: {e}")

with col_r:
    st.subheader("📋 Job Description")
    job_desc = st.text_area(
        "Paste the job description here",
        height=280,
        placeholder="Copy and paste the full job description…",
    )

st.divider()

# ── Action button ─────────────────────────────────────────────────────────────

run_disabled = not (resume_text_raw and job_desc.strip() and os.getenv("OPENAI_API_KEY"))
if run_disabled and not os.getenv("OPENAI_API_KEY"):
    st.warning("🔑 Enter your OPEN AI API key in the sidebar to enable the agent.")

run_btn = st.button(
    "🚀 Run Agent",
    type="primary",
    disabled=run_disabled,
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Agent execution
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    st.divider()
    st.subheader("🤖 Agent Execution Log")

    step_container = st.empty()
    progress_bar   = st.progress(0, text="Initialising…")

    NODE_LABELS = {
        "parse_documents":       ("Parsing resume…",             1, 5),
        "analyze_gaps":          ("Analysing skill gaps…",       2, 5),
        "rewrite_cv":            ("Rewriting CV…",               3, 5),
        "generate_cover_letter": ("Generating cover letter…",    4, 5),
        "recommend_resources":   ("Finding learning resources…", 5, 5),
    }

    logs: list[str]  = []
    final_state: dict = {}
    had_error = False

    t_start = time.time()

    for node_name, node_state in stream_agent(resume_text_raw, job_desc):
        label, step_num, total_steps = NODE_LABELS.get(
            node_name, (node_name, 0, 5)
        )
        progress = step_num / total_steps
        progress_bar.progress(progress, text=f"Step {step_num}/{total_steps}: {label}")

        current = node_state.get("current_step", node_name)
        logs.append(current)
        step_container.markdown(
            "\n".join(f'<div class="step-log">{lg}</div>' for lg in logs),
            unsafe_allow_html=True,
        )

        final_state.update(node_state)

        if node_state.get("error"):
            had_error = True
            st.error(f"❌ Agent stopped: {node_state['error']}")
            break

    elapsed = time.time() - t_start
    if not had_error:
        progress_bar.progress(1.0, text=f"✅ Completed in {elapsed:.1f}s")

    # ── Results ───────────────────────────────────────────────────────────────
    if not had_error and final_state:
        st.divider()
        st.subheader("📊 Results")

        # ── Keyword match scores ──────────────────────────────────────────────
        before = final_state.get("keyword_match_before", 0.0)
        after  = final_state.get("keyword_match_after",  0.0)
        delta  = after - before

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Match Before", f"{int(before*100)}%")
        col_b.metric("Match After",  f"{int(after*100)}%",  delta=f"+{int(delta*100)}%")
        col_c.metric("Improvement",  f"+{int(delta*100)} pts")

        st.divider()

        # ── Tab layout ────────────────────────────────────────────────────────
        tab_cv, tab_cl, tab_gaps, tab_res = st.tabs([
            "📄 Rewritten CV",
            "✉️ Cover Letter",
            "🔍 Skill Analysis",
            "📚 Learning Resources",
        ])

        # Tab 1: Rewritten CV
        with tab_cv:
            cv_text = final_state.get("rewritten_cv", "")
            if cv_text:
                st.text_area("Tailored CV", cv_text, height=450)

                candidate_name = ""
                parsed = final_state.get("parsed_resume", {})
                if isinstance(parsed, dict):
                    candidate_name = parsed.get("name", "")

                cv_bytes = generate_cv_docx(cv_text, candidate_name)
                st.download_button(
                    "⬇️ Download CV (.docx)",
                    data=cv_bytes,
                    file_name="tailored_cv.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.info("CV not generated yet.")

        # Tab 2: Cover Letter
        with tab_cl:
            cl_text = final_state.get("cover_letter", "")
            if cl_text:
                st.text_area("Cover Letter", cl_text, height=400)

                cl_bytes = generate_cover_letter_docx(cl_text, candidate_name)
                st.download_button(
                    "⬇️ Download Cover Letter (.docx)",
                    data=cl_bytes,
                    file_name="cover_letter.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            else:
                st.info("Cover letter not generated yet.")

        # Tab 3: Skill Analysis
        with tab_gaps:
            matching = final_state.get("matching_skills", [])
            gaps     = final_state.get("skill_gaps", [])

            col_m, col_g = st.columns(2)
            with col_m:
                st.markdown(f"### ✅ Matching Skills ({len(matching)})")
                if matching:
                    st.markdown(
                        badges(matching, "badge-green"),
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("None identified.")

            with col_g:
                st.markdown(f"### ⚠️ Skill Gaps ({len(gaps)})")
                if gaps:
                    st.markdown(
                        badges(gaps, "badge-red"),
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("No gaps found! 🎉")

        # Tab 4: Learning Resources
        with tab_res:
            resources = final_state.get("learning_resources", [])
            if resources:
                for r in resources:
                    icon = platform_icon(r.get("platform", ""))
                    skill    = r.get("skill", "")
                    platform = r.get("platform", "")
                    title    = r.get("title", "")
                    url      = r.get("url", "#")
                    reason   = r.get("reason", "")

                    st.markdown(f"""
<div class="resource-card">
  <b>{icon} {skill}</b> &nbsp;·&nbsp; <span style="color:#5f6368">{platform}</span><br/>
  <a href="{url}" target="_blank">{title}</a><br/>
  <small style="color:#5f6368">{reason}</small>
</div>
""", unsafe_allow_html=True)
            elif not gaps:
                st.success("No skill gaps — no resources needed! 🎉")
            else:
                st.info("Resources not generated yet.")

        # ── Parsed resume (expander) ──────────────────────────────────────────
        with st.expander("🔬 Raw parsed resume data (JSON)"):
            import json
            st.json(final_state.get("parsed_resume", {}))

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AI-Powered Job Application Agent · "
    "Built with LangGraph · MCP · Anthropic Claude · Streamlit"
)
