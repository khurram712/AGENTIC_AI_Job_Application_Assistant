# 💼 AI-Powered Job Application Agent

An agentic AI assistant that automates and personalises the job application process using **LangGraph**, **MCP (Model Context Protocol)**, and **Claude**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI (app.py)                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ invokes
┌────────────────────────────────▼────────────────────────────────────┐
│                    LangGraph StateGraph (graph.py)                  │
│                                                                     │
│  START → parse_documents → analyze_gaps → rewrite_cv               │
│       → generate_cover_letter → recommend_resources → END           │
└────────────┬────────────────────────────────────────────────────────┘
             │ each node calls via MCP stdio client
┌────────────▼────────────────────────────────────────────────────────┐
│                     MCP Server (mcp_server.py)                      │
│                                                                     │
│  Tools:  parse_resume · analyze_skill_gaps · rewrite_cv            │
│          generate_cover_letter · recommend_resources                │
│          calculate_keyword_match                                     │
│                                                                     │
│  LLM Backend: Anthropic Claude (claude-sonnet-4-20250514)          │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Choice | Why |
|--------|-----|
| **LangGraph** | Deterministic multi-step orchestration with typed state, conditional edges for error handling, and streaming support |
| **MCP (stdio transport)** | Each LangGraph node is decoupled from the tool implementation — tools live on the MCP server and can be swapped or extended independently |
| **Typed `AgentState`** | Single TypedDict flows through all nodes; no hidden state, easy to inspect and test |
| **Claude API** | Superior instruction-following for structured JSON outputs and document rewriting |

---

## Project Structure

```
job_application_agent/
├── app.py                        # Streamlit web UI
├── graph.py                      # LangGraph StateGraph definition
├── state.py                      # AgentState TypedDict
├── mcp_server.py                 # MCP server (all tools)
├── mcp_client.py                 # Async MCP client wrapper
├── nodes/
│   ├── parse_documents.py        # Node 1: parse resume + before-score
│   ├── analyze_gaps.py           # Node 2: skill gap analysis
│   ├── rewrite_cv.py             # Node 3: CV rewrite + after-score
│   ├── generate_cover_letter.py  # Node 4: cover letter generation
│   └── recommend_resources.py   # Node 5: learning resources
├── utils/
│   ├── pdf_parser.py             # PDF / TXT text extraction
│   └── doc_generator.py         # .docx output generation
├── requirements.txt
└── .env.example
```

---

## Setup

### 1. Clone / copy the project

```bash
cd job_application_agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The MCP server is spawned automatically as a subprocess when the agent runs — no separate terminal needed.

---

## Usage

1. Open the browser at `http://localhost:8501`
2. Enter your **Anthropic API key** in the sidebar (or set it in `.env`)
3. **Upload your resume** (PDF or TXT)
4. **Paste the job description**
5. Click **🚀 Run Agent**
6. Watch the agent progress through 5 nodes in real time
7. Download the tailored **CV** and **Cover Letter** as `.docx` files

---

## Agent Pipeline

| Step | Node | MCP Tool | Output |
|------|------|----------|--------|
| 1 | `parse_documents` | `parse_resume` + `calculate_keyword_match` | Structured resume JSON + before-score |
| 2 | `analyze_gaps` | `analyze_skill_gaps` | Matching skills & gap list |
| 3 | `rewrite_cv` | `rewrite_cv` + `calculate_keyword_match` | Tailored CV text + after-score |
| 4 | `generate_cover_letter` | `generate_cover_letter` | Personalised cover letter |
| 5 | `recommend_resources` | `recommend_resources` | Learning resources per gap |

---

## Extending the Agent

### Add a new MCP tool

1. Register it in `mcp_server.py` under `list_tools()` and `call_tool()`
2. Create a new node in `nodes/my_node.py`
3. Add it to the graph in `graph.py`
4. Update `AgentState` in `state.py` if new state fields are needed

### Swap the LLM

Change `MODEL` in `mcp_server.py` to any Anthropic model string, or replace the `_ask_claude()` helper with an OpenAI call.

---

## Deliverables

- ✅ Working Streamlit web app with full 5-node agent pipeline
- ✅ Tailored CV output matching the target JD
- ✅ Personalised cover letter generated automatically
- ✅ Skill gap report with matching / missing skills
- ✅ Curated learning recommendations per gap skill
- ✅ Keyword match score — before and after rewrite (TF-IDF cosine similarity)
- ✅ Downloadable `.docx` files for CV and cover letter
