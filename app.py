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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Reset cursor globally ── */
*, *::before, *::after {
    cursor: default !important;
}
a, button, [role="button"], label, select, option,
input[type="file"], input[type="submit"], input[type="button"],
.stButton > button, .stSelectbox, .stFileUploader,
[data-testid="stFileUploader"] *, [data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    cursor: pointer !important;
}
input, textarea, [contenteditable="true"],
[data-testid="stChatInput"] textarea {
    cursor: text !important;
}

/* ── Global ── */
.stApp {
    background: linear-gradient(165deg, #F8F9FC 0%, #EEF0F7 40%, #E8EAF2 100%) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #1a1a2e !important;
}

/* ── Sidebar — dark contrast panel ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16162a 50%, #111128 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.15) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #c8cad0 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.06) !important;
    margin: 0.8rem 0 !important;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    padding: 1.2rem 0.5rem 1rem 0.5rem;
    text-align: center;
}
.sidebar-brand .logo {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: white !important;
}
.sidebar-brand .logo span {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-brand .tagline {
    font-size: 0.62rem;
    color: #6c6c8a !important;
    margin-top: 0.25rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── Section Labels ── */
.s-label {
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #6c6c8a !important;
    margin: 0.8rem 0 0.4rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.s-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

/* ── Document Cards ── */
.doc-card {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.65rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0.3rem;
    transition: all 0.15s ease;
}
.doc-card:hover {
    background: rgba(102, 126, 234, 0.08);
    border-color: rgba(102, 126, 234, 0.2);
}
.doc-card .icon { font-size: 0.95rem; }
.doc-card .name {
    flex: 1;
    font-size: 0.75rem;
    font-weight: 500;
    color: #e0e0e8 !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.doc-card .badge {
    font-size: 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #667eea !important;
    background: rgba(102, 126, 234, 0.1);
    padding: 0.12rem 0.4rem;
    border-radius: 4px;
    font-weight: 500;
}

/* ── Metrics Grid ── */
.m-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
}
.m-card {
    padding: 0.55rem 0.4rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    text-align: center;
}
.m-card .val {
    font-size: 1.15rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
}
.m-card .lbl {
    font-size: 0.55rem;
    color: #6c6c8a !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.1rem;
    font-weight: 600;
}

/* ── Main Area Header ── */
.main-hdr {
    padding: 1.5rem 0 0.75rem 0;
}
.main-hdr h1 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    color: #1a1a2e !important;
    margin: 0 !important;
    line-height: 1.15 !important;
    letter-spacing: -0.03em;
}
.main-hdr h1 .accent {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-hdr .sub {
    font-size: 0.85rem;
    color: #7a7a96;
    margin-top: 0.25rem;
    font-weight: 400;
}

/* ── Welcome Card ── */
.w-card {
    padding: 2.5rem 2rem;
    border-radius: 16px;
    background: white;
    border: 1px solid #e2e4ec;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(102, 126, 234, 0.06);
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.w-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
    background-size: 200% auto;
    animation: shimmer 3s ease infinite;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.w-card h2 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.25rem !important;
    color: #1a1a2e !important;
    margin: 0 0 0.4rem 0 !important;
}
.w-card p {
    color: #7a7a96;
    font-size: 0.88rem;
    margin: 0;
}

/* ── Chat Messages ── */
[data-testid="stChatMessage"] {
    background: white !important;
    border: 1px solid #e8eaf0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    margin-bottom: 0.75rem !important;
    padding: 1rem !important;
}
[data-testid="stChatMessage"] * {
    color: #1a1a2e !important;
}
[data-testid="stChatMessage"] .stCaption p {
    color: #7a7a96 !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.45) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:not([kind="primary"]) {
    background: white !important;
    border: 1px solid #d8dae4 !important;
    border-radius: 8px !important;
    color: #1a1a2e !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #667eea !important;
    color: #667eea !important;
    box-shadow: 0 2px 8px rgba(102, 126, 234, 0.12) !important;
}

/* ── Sidebar buttons override ── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    color: white !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #c8cad0 !important;
}
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background: rgba(102, 126, 234, 0.1) !important;
    border-color: rgba(102, 126, 234, 0.3) !important;
    color: #667eea !important;
}

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: white !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    border: 1px solid #e2e4ec !important;
    border-radius: 12px !important;
    color: #1a1a2e !important;
    background: white !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.12) !important;
}

/* ── Selectbox in sidebar ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}

/* ── File uploader in sidebar ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    border: 1px dashed rgba(102, 126, 234, 0.3) !important;
    border-radius: 8px !important;
    background: rgba(102, 126, 234, 0.04) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: rgba(102, 126, 234, 0.1) !important;
    border: 1px solid rgba(102, 126, 234, 0.25) !important;
    border-radius: 6px !important;
    color: #a0a0ff !important;
    font-size: 0.75rem !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2) !important;
    border-radius: 4px !important;
}

/* ── Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    background: white !important;
    border: 1px solid #e2e4ec !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Hide branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""


# ── Session State ──────────────────────────────────────────

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
    """Render the sidebar."""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <div class="logo">⚡ <span>Lumen</span></div>
            <div class="tagline">Document Intelligence</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Domain
        st.markdown('<div class="s-label">Domain</div>', unsafe_allow_html=True)
        domain_options = ["None"] + AVAILABLE_DOMAINS
        domain_labels = {"None": "🌐 General"}
        for d in AVAILABLE_DOMAINS:
            q_file = DOMAINS_DIR / d / "example_questions.json"
            if q_file.exists():
                data = json.loads(q_file.read_text())
                domain_labels[d] = f"{data.get('icon', '📁')} {data.get('display_name', d)}"
            else:
                domain_labels[d] = d.replace("_", " ").title()

        st.selectbox("Domain", domain_options,
                      format_func=lambda x: domain_labels.get(x, x),
                      key="selected_domain", label_visibility="collapsed")

        # Model
        st.markdown('<div class="s-label">Model</div>', unsafe_allow_html=True)
        available = get_available_models()
        model_options = list(MODEL_OPTIONS.keys())
        model_labels = {}
        for m in model_options:
            model_labels[m] = f"🟢 {m}" if available.get(m) else f"⚪ {m} (no key)"

        st.selectbox("Model", model_options,
                      format_func=lambda x: model_labels.get(x, x),
                      key="selected_model", label_visibility="collapsed")

        if not available.get(st.session_state.selected_model):
            st.warning(f"Add API key for {st.session_state.selected_model} in secrets")

        st.divider()

        # Upload
        st.markdown('<div class="s-label">Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload", type=["pdf", "txt", "docx"],
                                           accept_multiple_files=True,
                                           key="file_uploader", label_visibility="collapsed")

        if uploaded_files and st.button("⚡ Process Documents", use_container_width=True, type="primary"):
            _process_documents(uploaded_files)

        # Inventory
        if st.session_state.document_stats:
            for fname, count in st.session_state.document_stats.items():
                ext = Path(fname).suffix.lower()
                icon = "📕" if ext == ".pdf" else "📝" if ext == ".txt" else "📘"
                st.markdown(f"""<div class="doc-card">
                    <span class="icon">{icon}</span>
                    <span class="name">{fname}</span>
                    <span class="badge">{count}</span>
                </div>""", unsafe_allow_html=True)
            st.caption(f"{st.session_state.vector_store.num_chunks} vectors indexed")

        st.divider()

        # Metrics
        st.markdown('<div class="s-label">Metrics</div>', unsafe_allow_html=True)
        summary = get_experiment_summary()
        st.markdown(f"""<div class="m-grid">
            <div class="m-card"><div class="val" style="color:#667eea;">{summary['total_queries']}</div><div class="lbl">Queries</div></div>
            <div class="m-card"><div class="val" style="color:#10b981;">{summary['avg_faithfulness']}</div><div class="lbl">Faithfulness</div></div>
            <div class="m-card"><div class="val" style="color:#764ba2;">{summary['avg_latency_ms']:.0f}<span style="font-size:0.55rem">ms</span></div><div class="lbl">Latency</div></div>
            <div class="m-card"><div class="val" style="color:#f59e0b;font-size:0.7rem;">{summary['best_model']}</div><div class="lbl">Top Model</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Clear Session", use_container_width=True):
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
        progress.progress(i / total, text=f"Reading {uf.name}...")
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
    time.sleep(0.4)
    st.rerun()


# ── Metadata ───────────────────────────────────────────────

def _render_metadata(meta: dict) -> None:
    """Render response metadata."""
    if not meta:
        return

    if meta.get("sources_display") and meta["sources_display"] != "No sources cited":
        st.info(f"📎 **Sources:** {meta['sources_display']}")

    pill_data = []
    if meta.get("confidence") is not None:
        pill_data.append(f"🎯 **{meta['confidence']}**/100")
    if meta.get("query_type"):
        pill_data.append(f"🔍 {meta['query_type'].replace('_', ' ').title()}")
    if meta.get("model_used"):
        pill_data.append(f"⚡ {meta['model_used']}")
    if meta.get("latency_ms"):
        pill_data.append(f"⏱️ {meta['latency_ms']:.0f}ms")

    if pill_data:
        cols = st.columns(len(pill_data))
        for i, text in enumerate(pill_data):
            cols[i].caption(text)

    if meta.get("eval_scores") and isinstance(meta["eval_scores"], dict) and meta["eval_scores"]:
        st.caption("**Quality Scores**")
        eval_cols = st.columns(len(meta["eval_scores"]))
        for i, (dim, score) in enumerate(meta["eval_scores"].items()):
            short = dim.replace("Hallucination Risk", "Halluc.").replace("Completeness", "Complete.")
            icon = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
            eval_cols[i].markdown(f"{icon} **{score}/5**\n\n`{short}`")


# ── Chat ───────────────────────────────────────────────────

def _render_chat() -> None:
    """Render conversation history."""
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role, avatar="user" if role == "user" else "🔆"):
            st.markdown(msg["content"])
            if role == "assistant" and "metadata" in msg:
                _render_metadata(msg["metadata"])


# ── Welcome ────────────────────────────────────────────────

def _render_welcome() -> None:
    """Show welcome screen."""
    st.markdown("""<div class="w-card">
        <h2>Upload documents to get started</h2>
        <p>Drop PDFs, DOCX, or TXT files in the sidebar — then ask anything about them.</p>
    </div>""", unsafe_allow_html=True)

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
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                _handle_query(q)


# ── Query Handling ─────────────────────────────────────────

def _handle_query(query: str) -> None:
    """Handle user query."""
    st.session_state.messages.append({"role": "user", "content": query})

    if not st.session_state.vector_store.is_populated:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please upload and process documents first.",
            "metadata": {},
        })
        st.rerun()
        return

    model_name = st.session_state.selected_model
    if not get_available_models().get(model_name):
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ {model_name} API key missing.",
            "metadata": {},
        })
        st.rerun()
        return

    st.rerun()


def _process_pending_query() -> None:
    """Process pending query with streaming status."""
    messages = st.session_state.messages
    if not messages or messages[-1]["role"] != "user":
        return

    query = messages[-1]["content"]
    model_name = st.session_state.selected_model

    if not get_available_models().get(model_name) or not st.session_state.vector_store.is_populated:
        return

    domain = st.session_state.selected_domain
    domain_prompt = ""
    if domain and domain != "None":
        domain_prompt, _ = _load_domain_config(domain)

    for msg in messages:
        role = msg["role"]
        with st.chat_message(role, avatar="user" if role == "user" else "🔆"):
            st.markdown(msg["content"])
            if role == "assistant" and "metadata" in msg:
                _render_metadata(msg["metadata"])

    with st.chat_message("assistant", avatar="🔆"):
        status = st.status("Analyzing your query...", expanded=True)

        try:
            llm = create_llm(model_name)
            status.update(label="Searching documents...", state="running")
            time.sleep(0.1)
            status.update(label="Reasoning...", state="running")

            response = run_pipeline(
                query=query, llm=llm,
                vector_store=st.session_state.vector_store,
                memory=st.session_state.memory,
                model_name=model_name, domain_prompt=domain_prompt,
            )

            status.update(label="Done!", state="complete")
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
                "role": "assistant", "content": response.answer, "metadata": metadata,
            })

        except Exception as e:
            logger.error("Query failed: %s", e)
            status.update(label="Error", state="error")
            st.markdown(f"An error occurred: {e}")
            st.session_state.messages.append({
                "role": "assistant", "content": f"Error: {e}", "metadata": {},
            })


# ── Main ───────────────────────────────────────────────────

def main() -> None:
    """Main entry point."""
    st.set_page_config(
        page_title="Lumen — Document Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _init_session_state()
    _render_sidebar()

    st.markdown("""<div class="main-hdr">
        <h1>⚡ <span class="accent">Lumen</span></h1>
        <div class="sub">Multi-Document Reasoning Assistant — powered by LangGraph</div>
    </div>""", unsafe_allow_html=True)

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
