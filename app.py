"""Lumen — Intelligent Multi-Document Reasoning Assistant (Streamlit Frontend)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.config import (
    AVAILABLE_DOMAINS,
    DOMAINS_DIR,
    MODEL_OPTIONS,
    SUPPORTED_EXTENSIONS,
)
from core.ingestion import ingest_file
from core.vector_store import VectorStore
from core.memory import MemoryManager
from models.llm_factory import create_llm, get_available_models
from agents.graph import run_pipeline
from agents.formatter import format_sources_display, format_eval_scorecard
from tracking.mlflow_logger import log_query, get_experiment_summary

# ── Setup ──────────────────────────────────────────────────

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Custom CSS ─────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

/* ── Force dark background everywhere ── */
.stApp {
    background: #0E1117 !important;
    font-family: 'Outfit', sans-serif !important;
    color: #E6EDF3 !important;
}

[data-testid="stSidebar"] {
    background: #161B22 !important;
    border-right: 1px solid rgba(108, 99, 255, 0.15) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    padding: 1rem 0 1.2rem 0;
    text-align: center;
}
.sidebar-brand .logo-text {
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #6C63FF 0%, #00D2FF 50%, #A855F7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.sidebar-brand .tagline {
    font-size: 0.7rem;
    color: #8B949E;
    margin-top: 0.2rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Section Labels ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8B949E;
    margin-bottom: 0.5rem;
    margin-top: 0.5rem;
}

/* ── Document Cards ── */
.doc-card {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.75rem;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.08) 0%, rgba(0, 210, 255, 0.05) 100%);
    border: 1px solid rgba(108, 99, 255, 0.15);
    margin-bottom: 0.4rem;
    transition: all 0.2s ease;
}
.doc-card:hover {
    border-color: rgba(108, 99, 255, 0.35);
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.12) 0%, rgba(0, 210, 255, 0.08) 100%);
}
.doc-card .icon { font-size: 1.1rem; }
.doc-card .name {
    flex: 1;
    font-size: 0.8rem;
    font-weight: 500;
    color: #E6EDF3;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.doc-card .chunks {
    font-size: 0.7rem;
    font-family: 'Fira Code', monospace;
    color: #6C63FF;
    background: rgba(108, 99, 255, 0.12);
    padding: 0.15rem 0.45rem;
    border-radius: 6px;
}

/* ── Metrics Grid ── */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.6rem;
}
.metric-card {
    padding: 0.7rem 0.5rem;
    border-radius: 12px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    border-radius: 12px;
    padding: 1px;
    background: linear-gradient(135deg, rgba(108,99,255,0.3), rgba(0,210,255,0.2));
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
}
.metric-card .value {
    font-size: 1.3rem;
    font-weight: 700;
    font-family: 'Fira Code', monospace;
}
.metric-card .label {
    font-size: 0.6rem;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.15rem;
    font-weight: 500;
}

/* ── Main Header ── */
.main-header {
    padding: 1.5rem 0 0.5rem 0;
    margin-bottom: 0.5rem;
}
.main-header h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    background: linear-gradient(135deg, #6C63FF 0%, #00D2FF 40%, #A855F7 80%, #6C63FF 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 4s ease infinite;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.02em;
}
@keyframes gradient-shift {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}
.main-header .subtitle {
    font-size: 0.9rem;
    color: #8B949E;
    margin-top: 0.3rem;
    font-weight: 400;
}

/* ── Welcome Card ── */
.welcome-card {
    padding: 3rem 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(108, 99, 255, 0.06) 0%, rgba(0, 210, 255, 0.04) 50%, rgba(168, 85, 247, 0.06) 100%);
    border: 1px solid rgba(108, 99, 255, 0.12);
    text-align: center;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}
.welcome-card::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(108, 99, 255, 0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 60%, rgba(0, 210, 255, 0.04) 0%, transparent 50%);
    animation: float 8s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    33% { transform: translate(10px, -10px) rotate(1deg); }
    66% { transform: translate(-5px, 5px) rotate(-1deg); }
}
.welcome-card h2 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.4rem !important;
    color: #E6EDF3 !important;
    margin-bottom: 0.5rem !important;
    position: relative;
    z-index: 1;
}
.welcome-card p {
    color: #8B949E;
    font-size: 0.9rem;
    position: relative;
    z-index: 1;
}

/* ── Chat Avatars ── */
[data-testid="stChatMessage"] {
    border-radius: 16px !important;
    border: 1px solid rgba(108, 99, 255, 0.08) !important;
    margin-bottom: 1rem !important;
}

/* ── Streamlit overrides ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6C63FF 0%, #A855F7 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.4) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:not([kind="primary"]) {
    border-radius: 10px !important;
    border: 1px solid rgba(108, 99, 255, 0.2) !important;
    background: rgba(108, 99, 255, 0.06) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: rgba(108, 99, 255, 0.4) !important;
    background: rgba(108, 99, 255, 0.12) !important;
}

/* ── Selectbox styling ── */
[data-testid="stSelectbox"] {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6C63FF, #00D2FF, #A855F7) !important;
    background-size: 200% auto;
    animation: gradient-shift 2s ease infinite;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    border: 1px solid rgba(108, 99, 255, 0.2) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.15) !important;
}

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
    border: 1px solid rgba(108, 99, 255, 0.15) !important;
}

/* ── Dividers ── */
hr {
    border-color: rgba(108, 99, 255, 0.1) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] section {
    border-radius: 12px !important;
    border: 1px dashed rgba(108, 99, 255, 0.25) !important;
}

/* ── Hide branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""


# ── Session State Initialization ───────────────────────────

def _init_session_state() -> None:
    """Initialize all Streamlit session state variables."""
    defaults = {
        "vector_store": VectorStore(),
        "memory": MemoryManager(),
        "messages": [],
        "documents_processed": False,
        "document_stats": {},
        "selected_domain": "None",
        "selected_model": list(MODEL_OPTIONS.keys())[0],
        "processing": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Domain Loading ─────────────────────────────────────────

def _load_domain_config(domain: str) -> tuple[str, list[str]]:
    """Load system prompt and example questions for a domain."""
    domain_dir = DOMAINS_DIR / domain
    system_prompt = ""
    examples: list[str] = []

    prompt_file = domain_dir / "system_prompt.txt"
    if prompt_file.exists():
        system_prompt = prompt_file.read_text(encoding="utf-8")

    questions_file = domain_dir / "example_questions.json"
    if questions_file.exists():
        data = json.loads(questions_file.read_text(encoding="utf-8"))
        examples = data.get("example_questions", [])

    return system_prompt, examples


# ── Sidebar ────────────────────────────────────────────────

def _render_sidebar() -> None:
    """Render the sidebar with all controls."""
    with st.sidebar:
        # Brand
        st.markdown("""
        <div class="sidebar-brand">
            <div class="logo-text">⚡ Lumen</div>
            <div class="tagline">Document Intelligence Engine</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Domain Selector ──
        st.markdown('<div class="section-label">🧭 Domain</div>', unsafe_allow_html=True)
        domain_options = ["None"] + AVAILABLE_DOMAINS
        domain_labels = {"None": "🌐 General"}
        for d in AVAILABLE_DOMAINS:
            q_file = DOMAINS_DIR / d / "example_questions.json"
            if q_file.exists():
                data = json.loads(q_file.read_text())
                domain_labels[d] = f"{data.get('icon', '📁')} {data.get('display_name', d)}"
            else:
                domain_labels[d] = d.replace("_", " ").title()

        st.selectbox(
            "Domain",
            domain_options,
            format_func=lambda x: domain_labels.get(x, x),
            key="selected_domain",
            label_visibility="collapsed",
        )

        # ── Model Selector ──
        st.markdown('<div class="section-label">🤖 Model</div>', unsafe_allow_html=True)
        available = get_available_models()
        model_options = list(MODEL_OPTIONS.keys())

        model_labels = {}
        for m in model_options:
            if available.get(m):
                model_labels[m] = f"🟢 {m}"
            else:
                model_labels[m] = f"🔴 {m} (no key)"

        st.selectbox(
            "Model",
            model_options,
            format_func=lambda x: model_labels.get(x, x),
            key="selected_model",
            label_visibility="collapsed",
        )

        if not available.get(st.session_state.selected_model):
            st.error(f"API key missing for {st.session_state.selected_model}", icon="🔑")

        st.divider()

        # ── File Upload ──
        st.markdown('<div class="section-label">📄 Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "PDF, TXT, DOCX",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files and st.button("⚡ Process Documents", use_container_width=True, type="primary"):
            _process_documents(uploaded_files)

        # ── Document Inventory ──
        if st.session_state.document_stats:
            st.markdown("")
            for fname, count in st.session_state.document_stats.items():
                ext = Path(fname).suffix.lower()
                icon = "📕" if ext == ".pdf" else "📝" if ext == ".txt" else "📘"
                st.markdown(f"""
                <div class="doc-card">
                    <span class="icon">{icon}</span>
                    <span class="name">{fname}</span>
                    <span class="chunks">{count} chunks</span>
                </div>
                """, unsafe_allow_html=True)
            total = st.session_state.vector_store.num_chunks
            st.markdown(f"""
            <div style="text-align:center; margin-top:0.4rem;">
                <span style="font-size:0.7rem; color:#8B949E; font-family:'Fira Code',monospace;">
                    {total} total vectors indexed
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── MLflow Metrics ──
        st.markdown('<div class="section-label">📊 Experiment Metrics</div>', unsafe_allow_html=True)
        summary = get_experiment_summary()
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value" style="color: #6C63FF;">{summary['total_queries']}</div>
                <div class="label">Queries</div>
            </div>
            <div class="metric-card">
                <div class="value" style="color: #00D2FF;">{summary['avg_faithfulness']}</div>
                <div class="label">Avg Faith.</div>
            </div>
            <div class="metric-card">
                <div class="value" style="color: #A855F7;">{summary['avg_latency_ms']:.0f}<span style="font-size:0.6rem;">ms</span></div>
                <div class="label">Avg Latency</div>
            </div>
            <div class="metric-card">
                <div class="value" style="color: #10B981; font-size: 0.8rem;">{summary['best_model']}</div>
                <div class="label">Best Model</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.session_state.documents_processed = False
            st.session_state.document_stats = {}
            st.rerun()


# ── Document Processing ────────────────────────────────────

def _process_documents(uploaded_files: list) -> None:
    """Ingest and embed uploaded documents."""
    progress = st.sidebar.progress(0, text="Processing...")
    total = len(uploaded_files)

    all_chunks = []
    stats: dict[str, int] = {}

    for i, uf in enumerate(uploaded_files):
        progress.progress((i) / total, text=f"Reading {uf.name}...")
        try:
            result = ingest_file(uf, uf.name)
            if result.error:
                st.sidebar.error(f"❌ {uf.name}: {result.error}")
            else:
                all_chunks.extend(result.chunks)
                stats[result.filename] = result.num_chunks
        except Exception as e:
            st.sidebar.error(f"❌ {uf.name}: {e}")

    if all_chunks:
        progress.progress(0.85, text="Building index...")
        st.session_state.vector_store.add_chunks(all_chunks)
        st.session_state.document_stats.update(stats)
        st.session_state.documents_processed = True

    progress.progress(1.0, text="✅ Ready!")
    time.sleep(0.5)
    st.rerun()


# ── Metadata Rendering ─────────────────────────────────────

def _render_metadata(meta: dict) -> None:
    """Render response metadata with styled indicators."""
    if not meta:
        return

    # Source citations
    if meta.get("sources_display") and meta["sources_display"] != "No sources cited":
        st.info(f"📎 **Sources:** {meta['sources_display']}")

    # Metadata as columns
    pill_data = []
    if meta.get("confidence") is not None:
        pill_data.append(f"🎯 **{meta['confidence']}**/100")
    if meta.get("query_type"):
        qt_display = meta["query_type"].replace("_", " ").title()
        pill_data.append(f"🔍 {qt_display}")
    if meta.get("model_used"):
        pill_data.append(f"⚡ {meta['model_used']}")
    if meta.get("latency_ms"):
        pill_data.append(f"⏱️ {meta['latency_ms']:.0f}ms")

    if pill_data:
        pill_cols = st.columns(len(pill_data))
        for i, text in enumerate(pill_data):
            pill_cols[i].caption(text)

    # Eval scorecard
    if meta.get("eval_scores") and isinstance(meta["eval_scores"], dict) and meta["eval_scores"]:
        st.caption("**Quality Scores**")
        eval_cols = st.columns(len(meta["eval_scores"]))
        for i, (dim, score) in enumerate(meta["eval_scores"].items()):
            short_label = dim.replace("Hallucination Risk", "Halluc.").replace("Completeness", "Complete.")
            color = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
            eval_cols[i].markdown(f"{color} **{score}/5**\n\n`{short_label}`")


# ── Chat Display ───────────────────────────────────────────

def _render_chat() -> None:
    """Render the conversation history."""
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="user" if role == "user" else "🔆"):
            st.markdown(msg["content"])
            if role == "assistant" and "metadata" in msg:
                _render_metadata(msg["metadata"])


# ── Welcome Screen ─────────────────────────────────────────

def _render_welcome() -> None:
    """Show welcome screen with example questions."""
    st.markdown("""
    <div class="welcome-card">
        <h2>Upload documents to get started</h2>
        <p>Drop PDFs, DOCX, or TXT files in the sidebar — then ask anything about them.</p>
    </div>
    """, unsafe_allow_html=True)

    domain = st.session_state.selected_domain
    examples: list[str] = []

    if domain and domain != "None":
        _, examples = _load_domain_config(domain)

    if not examples:
        examples = [
            "Summarize the key points from all uploaded documents.",
            "Compare the main themes across these documents.",
            "Are there any contradictions between the documents?",
            "What are the most important findings?",
        ]

    st.markdown("#### 💡 Try these after uploading:")
    cols = st.columns(2)
    for i, q in enumerate(examples[:4]):
        with cols[i % 2]:
            if st.button(q, key=f"example_{i}", use_container_width=True):
                _handle_query(q)


# ── Query Handling ─────────────────────────────────────────

def _handle_query(query: str) -> None:
    """Process a user query through the full Lumen pipeline."""
    st.session_state.messages.append({"role": "user", "content": query})

    if not st.session_state.vector_store.is_populated:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please upload and process documents first before asking questions.",
            "metadata": {},
        })
        st.rerun()
        return

    model_name = st.session_state.selected_model
    available = get_available_models()
    if not available.get(model_name):
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ {model_name} is not available — API key is missing. Set it in `.env` or pick another model.",
            "metadata": {},
        })
        st.rerun()
        return

    st.rerun()


def _process_pending_query() -> None:
    """Process a pending user query with streaming status."""
    messages = st.session_state.messages
    if not messages or messages[-1]["role"] != "user":
        return

    query = messages[-1]["content"]
    model_name = st.session_state.selected_model
    available = get_available_models()

    if not available.get(model_name) or not st.session_state.vector_store.is_populated:
        return

    domain = st.session_state.selected_domain
    domain_prompt = ""
    if domain and domain != "None":
        domain_prompt, _ = _load_domain_config(domain)

    # Render existing chat history
    for msg in messages:
        role = msg["role"]
        with st.chat_message(role, avatar="user" if role == "user" else "🔆"):
            st.markdown(msg["content"])
            if role == "assistant" and "metadata" in msg:
                _render_metadata(msg["metadata"])

    # Stream the new response
    with st.chat_message("assistant", avatar="🔆"):
        status = st.status("🔍 Analyzing your query...", expanded=True)

        try:
            llm = create_llm(model_name)

            status.update(label="📡 Searching documents...", state="running")
            time.sleep(0.1)

            status.update(label="🧠 Reasoning...", state="running")

            response = run_pipeline(
                query=query,
                llm=llm,
                vector_store=st.session_state.vector_store,
                memory=st.session_state.memory,
                model_name=model_name,
                domain_prompt=domain_prompt,
            )

            status.update(label="✅ Done!", state="complete")

            log_query(query, response)

            st.markdown(response.answer)

            metadata = {
                "sources_display": format_sources_display(response.sources),
                "confidence": response.confidence,
                "query_type": response.query_type,
                "model_used": response.model_used,
                "eval_display": format_eval_scorecard(response.eval_scores),
                "eval_scores": response.eval_scores,
                "latency_ms": response.latency_ms,
            }

            _render_metadata(metadata)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.answer,
                "metadata": metadata,
            })

        except Exception as e:
            logger.error("Query handling failed: %s", e)
            status.update(label="❌ Error", state="error")
            error_msg = f"An error occurred: {e}"
            st.markdown(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "metadata": {},
            })

    return True


# ── Main ───────────────────────────────────────────────────

def main() -> None:
    """Main entry point for the Lumen Streamlit app."""
    st.set_page_config(
        page_title="Lumen — Document Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    _init_session_state()
    _render_sidebar()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ Lumen</h1>
        <div class="subtitle">Multi-Document Reasoning Assistant — powered by LangGraph</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.documents_processed:
        _render_welcome()
    else:
        messages = st.session_state.messages
        has_pending = messages and messages[-1]["role"] == "user" and (
            len(messages) < 2 or messages[-2]["role"] != "user"
        )

        if has_pending:
            _process_pending_query()
        else:
            _render_chat()

        if query := st.chat_input("Ask Lumen anything about your documents..."):
            _handle_query(query)


if __name__ == "__main__":
    main()
