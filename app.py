"""
Lumen — Intelligent Multi-Document Reasoning Assistant

Glassmorphism UI with frosted mint theme, green/teal accents,
animated gradient branding, and mobile-responsive layout.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from core.config import AVAILABLE_DOMAINS, DOMAINS_DIR, MODEL_OPTIONS
from core.ingestion import ingest_file
from core.vector_store import VectorStore
from core.memory import MemoryManager
from models.llm_factory import create_llm, get_available_models
from agents.graph import run_pipeline
from agents.formatter import format_sources_display, format_eval_scorecard
from tracking.mlflow_logger import log_query, get_experiment_summary

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# Theme
# ════════════════════════════════════════════════════════════

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700;800;900&family=Fira+Code:wght@400;500&display=swap');

/* ── Cursor ── */
*, *::before, *::after { cursor: default !important }
a, button, [role="button"], label, select, option,
input[type="file"], .stButton > button,
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] { cursor: pointer !important }
input, textarea, [contenteditable="true"],
[data-testid="stChatInput"] textarea { cursor: text !important }

/* ── Background — frosted mint with subtle gradient mesh ── */
.stApp {
    background: #f0fdf4 !important;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(16,185,129,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 20%, rgba(6,182,212,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 80%, rgba(20,184,166,0.06) 0%, transparent 50%) !important;
    background-attachment: fixed !important;
    font-family: 'Nunito', sans-serif !important;
    color: #134e4a !important;
}

/* ── Glassmorphism mixin (reused everywhere) ── */
/* glass: bg white/60, blur 16px, border white/30, subtle shadow */

/* ── Sidebar — frosted glass panel ── */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255,255,255,0.5) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.04) !important;
    min-width: 290px !important;
    width: 290px !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: 290px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Nunito', sans-serif !important;
    color: #134e4a !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(16,185,129,0.12) !important;
    margin: 0.75rem 0 !important;
}

/* ── Hide collapse button (renders as "keyboard_double_arrow" text) ── */
button[kind="header"],
[data-testid="collapsedControl"],
[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] > div > div > div > button:first-child,
[data-testid="stSidebarContent"] > div:first-child > button {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
}

/* ── Animated gradient brand ── */
.brand {
    padding: 0.8rem 0 0.5rem 0;
    text-align: center;
}
.brand .logo {
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 25%, #14b8a6 50%, #059669 75%, #10b981 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: brand-flow 4s ease infinite;
}
@keyframes brand-flow {
    0% { background-position: 0% 50% }
    50% { background-position: 100% 50% }
    100% { background-position: 0% 50% }
}
.brand .tag {
    font-size: 0.6rem;
    color: #6b9f97 !important;
    margin-top: 0.15rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 700;
}

/* ── Section headings ── */
.sh {
    font-size: 0.6rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #6b9f97 !important;
    margin: 0.7rem 0 0.35rem 0;
}

/* ── Document cards — glass ── */
.dc {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0.65rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.5);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.6);
    margin-bottom: 0.3rem;
    transition: all 0.2s ease;
}
.dc:hover {
    background: rgba(16,185,129,0.08);
    border-color: rgba(16,185,129,0.25);
    transform: translateX(2px);
}
.dc .i { font-size: 0.95rem }
.dc .n {
    flex: 1;
    font-size: 0.73rem;
    font-weight: 600;
    color: #134e4a !important;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.dc .b {
    font-size: 0.6rem;
    font-family: 'Fira Code', monospace;
    color: #059669 !important;
    background: rgba(16,185,129,0.1);
    padding: 0.12rem 0.4rem;
    border-radius: 6px;
    font-weight: 500;
}

/* ── Metrics grid — glass cards ── */
.mg { display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem }
.mc {
    padding: 0.55rem 0.3rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.45);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.5);
    text-align: center;
}
.mc .v {
    font-size: 1.1rem;
    font-weight: 800;
    font-family: 'Fira Code', monospace;
}
.mc .l {
    font-size: 0.5rem;
    color: #6b9f97 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.08rem;
    font-weight: 700;
}

/* ── Main header ── */
.mh { padding: 1.2rem 0 0.5rem 0 }
.mh h1 {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 900 !important;
    font-size: 2.2rem !important;
    margin: 0 !important;
    line-height: 1.15 !important;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 40%, #14b8a6 70%, #059669 100%);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: brand-flow 4s ease infinite;
}
.mh .s {
    font-size: 0.88rem;
    color: #6b9f97;
    margin-top: 0.2rem;
    font-weight: 500;
}

/* ── Welcome card — prominent glass ── */
.wc {
    padding: 2.8rem 2rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 4px 30px rgba(0,0,0,0.04), 0 1px 3px rgba(16,185,129,0.08);
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.wc::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b981, #06b6d4, #14b8a6, #10b981);
    background-size: 300% auto;
    animation: shimmer 4s ease infinite;
}
@keyframes shimmer { 0%{background-position:0% center} 100%{background-position:300% center} }
.wc h2 {
    font-weight: 800 !important;
    font-size: 1.3rem !important;
    color: #134e4a !important;
    margin: 0 0 0.35rem 0 !important;
}
.wc p { color: #6b9f97; font-size: 0.9rem; margin: 0 }

/* ── Chat messages — glass bubbles ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.6) !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03) !important;
    margin-bottom: 0.75rem !important;
    padding: 1rem !important;
}
[data-testid="stChatMessage"] * { color: #134e4a !important }
[data-testid="stChatMessage"] .stCaption p { color: #6b9f97 !important }

/* ── Primary button — teal gradient ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 2px 12px rgba(16,185,129,0.25) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(16,185,129,0.35) !important;
    transform: translateY(-2px) !important;
}

/* ── Secondary buttons — glass ── */
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border: 1px solid rgba(16,185,129,0.15) !important;
    border-radius: 10px !important;
    color: #134e4a !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #10b981 !important;
    color: #059669 !important;
    box-shadow: 0 2px 12px rgba(16,185,129,0.12) !important;
    transform: translateY(-1px) !important;
}

/* ── Chat input — single clean border ── */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    border-radius: 14px !important;
    box-shadow: none !important;
    padding: 0 !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Nunito', sans-serif !important;
    border: 1.5px solid rgba(16,185,129,0.2) !important;
    border-radius: 14px !important;
    color: #134e4a !important;
    background: rgba(255,255,255,0.55) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.03) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.1) !important;
    outline: none !important;
}

/* ── Sidebar selectbox — glass ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.4) !important;
    border: 1px solid rgba(16,185,129,0.12) !important;
    border-radius: 10px !important;
}

/* ── File uploader — glass ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    border: 1.5px dashed rgba(16,185,129,0.3) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.35) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    padding: 0.7rem !important;
}

/* ── Progress bar — teal ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #10b981, #06b6d4) !important;
    border-radius: 6px !important;
}

/* ── Status widget — glass ── */
[data-testid="stStatusWidget"] {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.6) !important;
    border-radius: 12px !important;
}

/* ── Alert/info boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.82rem !important;
    background: rgba(255,255,255,0.4) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
}

/* ── Warning box in sidebar ── */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(251,191,36,0.08) !important;
    border: 1px solid rgba(251,191,36,0.2) !important;
}

/* ── Mobile responsive ── */
@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        width: 260px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 260px !important;
        padding: 0.75rem 0.9rem !important;
    }
    .mh h1 { font-size: 1.6rem !important }
    .mh .s { font-size: 0.78rem }
    .wc { padding: 1.8rem 1.2rem }
    .wc h2 { font-size: 1.1rem !important }
    [data-testid="stChatMessage"] { padding: 0.75rem !important }
}

@media (max-width: 480px) {
    .mh h1 { font-size: 1.3rem !important }
    .brand .logo { font-size: 1.3rem }
    .mg { grid-template-columns: 1fr 1fr; gap: 0.3rem }
    .mc .v { font-size: 0.9rem }
}

/* ── Hide branding ── */
#MainMenu { visibility: hidden }
footer { visibility: hidden }
header { visibility: hidden }
</style>
"""


# ════════════════════════════════════════════════════════════
# Session State
# ════════════════════════════════════════════════════════════

def _init_session_state() -> None:
    """Initialize all session state variables on first load."""
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


def _load_domain(domain: str) -> tuple[str, list[str]]:
    """Load system prompt and example questions for a domain module."""
    domain_path = DOMAINS_DIR / domain
    system_prompt = ""
    examples: list[str] = []

    prompt_file = domain_path / "system_prompt.txt"
    if prompt_file.exists():
        system_prompt = prompt_file.read_text(encoding="utf-8")

    questions_file = domain_path / "example_questions.json"
    if questions_file.exists():
        data = json.loads(questions_file.read_text(encoding="utf-8"))
        examples = data.get("example_questions", [])

    return system_prompt, examples


# ════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    """Render the frosted glass sidebar."""
    with st.sidebar:
        st.markdown(
            '<div class="brand">'
            '<div class="logo">⚡ Lumen</div>'
            '<div class="tag">Document Intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Domain
        st.markdown('<div class="sh">Domain</div>', unsafe_allow_html=True)
        domain_options = ["None"] + AVAILABLE_DOMAINS
        domain_labels = {"None": "🌐 General"}
        for d in AVAILABLE_DOMAINS:
            config_file = DOMAINS_DIR / d / "example_questions.json"
            if config_file.exists():
                data = json.loads(config_file.read_text())
                domain_labels[d] = f"{data.get('icon', '📁')} {data.get('display_name', d)}"
        st.selectbox(
            "domain_select", domain_options,
            format_func=lambda x: domain_labels.get(x, x),
            key="selected_domain", label_visibility="collapsed",
        )

        # Model
        st.markdown('<div class="sh">Model</div>', unsafe_allow_html=True)
        available_models = get_available_models()
        model_labels = {
            m: f"🟢 {m}" if available_models.get(m) else f"⚪ {m} (no key)"
            for m in MODEL_OPTIONS
        }
        st.selectbox(
            "model_select", list(MODEL_OPTIONS.keys()),
            format_func=lambda x: model_labels.get(x, x),
            key="selected_model", label_visibility="collapsed",
        )
        if not available_models.get(st.session_state.selected_model):
            st.warning(f"API key missing for {st.session_state.selected_model}")

        st.divider()

        # File uploader
        st.markdown('<div class="sh">Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or DOCX",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="file_uploader",
        )

        if uploaded_files:
            if st.button("⚡ Process Documents", use_container_width=True, type="primary"):
                _process_documents(uploaded_files)

        # Document inventory
        if st.session_state.document_stats:
            st.markdown('<div class="sh">Indexed</div>', unsafe_allow_html=True)
            for filename, chunk_count in st.session_state.document_stats.items():
                ext = Path(filename).suffix.lower()
                icon = "📕" if ext == ".pdf" else "📝" if ext == ".txt" else "📘"
                st.markdown(
                    f'<div class="dc">'
                    f'<span class="i">{icon}</span>'
                    f'<span class="n">{filename}</span>'
                    f'<span class="b">{chunk_count}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"{st.session_state.vector_store.num_chunks} vectors indexed")

        st.divider()

        # Metrics
        st.markdown('<div class="sh">Metrics</div>', unsafe_allow_html=True)
        summary = get_experiment_summary()
        st.markdown(
            f'<div class="mg">'
            f'<div class="mc"><div class="v" style="color:#059669">{summary["total_queries"]}</div><div class="l">Queries</div></div>'
            f'<div class="mc"><div class="v" style="color:#0891b2">{summary["avg_faithfulness"]}</div><div class="l">Faith.</div></div>'
            f'<div class="mc"><div class="v" style="color:#0d9488">{summary["avg_latency_ms"]:.0f}<span style="font-size:0.5rem">ms</span></div><div class="l">Latency</div></div>'
            f'<div class="mc"><div class="v" style="color:#059669;font-size:0.65rem">{summary["best_model"]}</div><div class="l">Top Model</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Session", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.session_state.documents_processed = False
            st.session_state.document_stats = {}
            st.rerun()


# ════════════════════════════════════════════════════════════
# Document Processing
# ════════════════════════════════════════════════════════════

def _process_documents(uploaded_files: list) -> None:
    """Ingest uploaded files, chunk them, and build the FAISS vector index."""
    progress = st.sidebar.progress(0, text="Processing...")
    all_chunks = []
    stats: dict[str, int] = {}

    for i, uf in enumerate(uploaded_files):
        progress.progress(i / len(uploaded_files), text=f"Reading {uf.name}...")
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


# ════════════════════════════════════════════════════════════
# Response Metadata
# ════════════════════════════════════════════════════════════

def _render_metadata(meta: dict) -> None:
    """Render source citations, confidence, query type, model, latency, and eval scores."""
    if not meta:
        return

    if meta.get("sources_display") and meta["sources_display"] != "No sources cited":
        st.info(f"📎 **Sources:** {meta['sources_display']}")

    pills = []
    if meta.get("confidence") is not None:
        pills.append(f"🎯 **{meta['confidence']}**/100")
    if meta.get("query_type"):
        pills.append(f"🔍 {meta['query_type'].replace('_', ' ').title()}")
    if meta.get("model_used"):
        pills.append(f"⚡ {meta['model_used']}")
    if meta.get("latency_ms"):
        pills.append(f"⏱️ {meta['latency_ms']:.0f}ms")

    if pills:
        cols = st.columns(len(pills))
        for i, text in enumerate(pills):
            cols[i].caption(text)

    if meta.get("eval_scores") and isinstance(meta["eval_scores"], dict):
        st.caption("**Quality Scores**")
        eval_cols = st.columns(len(meta["eval_scores"]))
        for i, (dimension, score) in enumerate(meta["eval_scores"].items()):
            short_name = dimension.replace("Hallucination Risk", "Halluc.").replace("Completeness", "Complete.")
            icon = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
            eval_cols[i].markdown(f"{icon} **{score}/5**\n\n`{short_name}`")


# ════════════════════════════════════════════════════════════
# Chat
# ════════════════════════════════════════════════════════════

def _render_chat() -> None:
    """Render conversation history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="user" if msg["role"] == "user" else "🔆"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                _render_metadata(msg["metadata"])


# ════════════════════════════════════════════════════════════
# Welcome
# ════════════════════════════════════════════════════════════

def _render_welcome() -> None:
    """Show welcome card and example questions."""
    st.markdown(
        '<div class="wc">'
        '<h2>Upload documents to get started</h2>'
        '<p>Drop PDFs, DOCX, or TXT files in the sidebar — then ask anything.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    domain = st.session_state.selected_domain
    examples: list[str] = []
    if domain and domain != "None":
        _, examples = _load_domain(domain)
    if not examples:
        examples = [
            "Summarize the key points from all uploaded documents.",
            "Compare the main themes across these documents.",
            "Are there any contradictions between the documents?",
            "What are the most important findings?",
        ]

    st.markdown("#### 💡 Try these after uploading:")
    cols = st.columns(2)
    for i, question in enumerate(examples[:4]):
        with cols[i % 2]:
            if st.button(question, key=f"ex_{i}", use_container_width=True):
                _handle_query(question)


# ════════════════════════════════════════════════════════════
# Query Handling
# ════════════════════════════════════════════════════════════

def _handle_query(query: str) -> None:
    """Add user message and trigger pipeline."""
    st.session_state.messages.append({"role": "user", "content": query})

    if not st.session_state.vector_store.is_populated:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please upload and process documents first.",
            "metadata": {},
        })
        st.rerun()
        return

    if not get_available_models().get(st.session_state.selected_model):
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ {st.session_state.selected_model} API key missing.",
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

    if not get_available_models().get(model_name) or not st.session_state.vector_store.is_populated:
        return

    domain = st.session_state.selected_domain
    domain_prompt = _load_domain(domain)[0] if domain and domain != "None" else ""

    # Render history
    for msg in messages:
        with st.chat_message(msg["role"], avatar="user" if msg["role"] == "user" else "🔆"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                _render_metadata(msg["metadata"])

    # Stream new response
    with st.chat_message("assistant", avatar="🔆"):
        status = st.status("Analyzing your query...", expanded=True)

        try:
            llm = create_llm(model_name)
            status.update(label="Searching documents...", state="running")
            time.sleep(0.1)
            status.update(label="Reasoning...", state="running")

            response = run_pipeline(
                query=query,
                llm=llm,
                vector_store=st.session_state.vector_store,
                memory=st.session_state.memory,
                model_name=model_name,
                domain_prompt=domain_prompt,
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
                "role": "assistant",
                "content": response.answer,
                "metadata": metadata,
            })

        except Exception as e:
            logger.error("Query failed: %s", e)
            status.update(label="Error", state="error")
            st.markdown(f"An error occurred: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {e}",
                "metadata": {},
            })


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="Lumen — Document Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(THEME_CSS, unsafe_allow_html=True)
    _init_session_state()
    _render_sidebar()

    st.markdown(
        '<div class="mh">'
        '<h1>⚡ Lumen</h1>'
        '<div class="s">Multi-Document Reasoning Assistant — powered by LangGraph</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.documents_processed:
        _render_welcome()
    else:
        messages = st.session_state.messages
        has_pending = (
            messages
            and messages[-1]["role"] == "user"
            and (len(messages) < 2 or messages[-2]["role"] != "user")
        )

        if has_pending:
            _process_pending_query()
        else:
            _render_chat()

        if query := st.chat_input("Ask Lumen anything about your documents..."):
            _handle_query(query)


if __name__ == "__main__":
    main()
