"""
Lumen — Intelligent Multi-Document Reasoning Assistant

Clean Streamlit frontend. CSS is cosmetic-only — no structural
overrides that break sidebar, file uploader, or collapse buttons.
Theme is set via .streamlit/config.toml (teal/mint).
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
# CSS — cosmetic only, nothing structural
# ════════════════════════════════════════════════════════════

SAFE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&display=swap');

/* ── Font for main area only (NOT sidebar — that breaks icon fonts) ── */
.main, .main * {
    font-family: 'Nunito', sans-serif !important;
}

/* ── Main area background ── */
.main .block-container {
    padding-top: 1.5rem !important;
}

/* ── Chat messages — white cards ── */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(13,148,136,0.1) !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
    margin-bottom: 0.6rem !important;
}

/* ── Primary buttons — teal gradient ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0d9488 0%, #06b6d4 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(13,148,136,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(13,148,136,0.35) !important;
}

/* ── Secondary buttons ── */
.stButton > button:not([kind="primary"]) {
    border-radius: 8px !important;
    border: 1px solid rgba(13,148,136,0.2) !important;
    font-weight: 600 !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #0d9488, #06b6d4) !important;
}

/* ── Brand header (custom HTML in main area) ── */
.lumen-brand {
    padding: 0.5rem 0 1rem 0;
}
.lumen-brand h1 {
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    margin: 0 !important;
    color: #134e4a !important;
}
.lumen-brand .grad {
    background: linear-gradient(135deg, #0d9488, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.lumen-brand .sub {
    font-size: 0.88rem;
    color: #5eaba5;
    margin-top: 0.15rem;
}

/* ── Welcome card ── */
.welcome {
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(13,148,136,0.12);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1rem 0 1.5rem 0;
    box-shadow: 0 2px 12px rgba(13,148,136,0.06);
    position: relative;
    overflow: hidden;
}
.welcome::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0d9488, #06b6d4, #0d9488);
    background-size: 200% auto;
    animation: flow 3s ease infinite;
}
@keyframes flow {
    0% { background-position: 0% center }
    100% { background-position: 200% center }
}
.welcome h2 {
    font-family: 'Nunito', sans-serif !important;
    color: #134e4a !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    margin: 0 0 0.3rem 0 !important;
}
.welcome p {
    color: #5eaba5;
    font-size: 0.9rem;
    margin: 0;
}

/* ── Sidebar brand ── */
.sb-brand {
    text-align: center;
    padding: 0.3rem 0 0.5rem 0;
}
.sb-brand .name {
    font-size: 1.4rem;
    font-weight: 800;
    color: #0d9488;
}
.sb-brand .tag {
    font-size: 0.6rem;
    color: #5eaba5;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 700;
}

/* ── Metric cards in sidebar ── */
.met-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.35rem;
}
.met-card {
    background: rgba(255,255,255,0.5);
    border: 1px solid rgba(13,148,136,0.1);
    border-radius: 8px;
    padding: 0.4rem 0.2rem;
    text-align: center;
}
.met-card .v {
    font-size: 1rem;
    font-weight: 800;
    color: #0d9488;
}
.met-card .l {
    font-size: 0.5rem;
    color: #5eaba5;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
}

/* ── Doc cards ── */
.doc-c {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.5rem;
    border-radius: 6px;
    background: rgba(255,255,255,0.4);
    border: 1px solid rgba(13,148,136,0.08);
    margin-bottom: 0.25rem;
    font-size: 0.75rem;
}
.doc-c .fn {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-weight: 600;
    color: #134e4a;
}
.doc-c .cc {
    color: #0d9488;
    font-weight: 700;
    font-size: 0.7rem;
}

/* ── Hide Streamlit branding only ── */
#MainMenu { visibility: hidden }
footer { visibility: hidden }
</style>
"""


# ════════════════════════════════════════════════════════════
# Session State
# ════════════════════════════════════════════════════════════

def _init() -> None:
    """Initialize session state."""
    for k, v in {
        "vector_store": VectorStore(),
        "memory": MemoryManager(),
        "messages": [],
        "documents_processed": False,
        "document_stats": {},
        "selected_domain": "None",
        "selected_model": list(MODEL_OPTIONS.keys())[0],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_domain(domain: str) -> tuple[str, list[str]]:
    """Load domain system prompt and example questions."""
    dp = DOMAINS_DIR / domain
    sp = (dp / "system_prompt.txt").read_text("utf-8") if (dp / "system_prompt.txt").exists() else ""
    eq = []
    qf = dp / "example_questions.json"
    if qf.exists():
        eq = json.loads(qf.read_text("utf-8")).get("example_questions", [])
    return sp, eq


# ════════════════════════════════════════════════════════════
# Sidebar — uses ONLY native Streamlit components
# ════════════════════════════════════════════════════════════

def _sidebar() -> None:
    """Render sidebar with native components."""
    with st.sidebar:
        # Brand
        st.markdown(
            '<div class="sb-brand">'
            '<div class="name">⚡ Lumen</div>'
            '<div class="tag">Document Intelligence</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.divider()

        # Domain selector — native, no CSS override
        st.caption("DOMAIN")
        domain_options = ["None"] + AVAILABLE_DOMAINS
        domain_labels = {"None": "🌐 General"}
        for d in AVAILABLE_DOMAINS:
            cf = DOMAINS_DIR / d / "example_questions.json"
            if cf.exists():
                data = json.loads(cf.read_text())
                domain_labels[d] = f"{data.get('icon', '📁')} {data.get('display_name', d)}"
        st.selectbox(
            "Select domain",
            domain_options,
            format_func=lambda x: domain_labels.get(x, x),
            key="selected_domain",
            label_visibility="collapsed",
        )

        # Model selector — native
        st.caption("MODEL")
        avail = get_available_models()
        mlabels = {m: f"🟢 {m}" if avail.get(m) else f"⚪ {m} (no key)" for m in MODEL_OPTIONS}
        st.selectbox(
            "Select model",
            list(MODEL_OPTIONS.keys()),
            format_func=lambda x: mlabels.get(x, x),
            key="selected_model",
            label_visibility="collapsed",
        )
        if not avail.get(st.session_state.selected_model):
            st.warning(f"API key missing for {st.session_state.selected_model}")

        st.divider()

        # File uploader — NATIVE, visible label, no CSS tricks
        uploaded = st.file_uploader(
            "Upload PDF, TXT, or DOCX",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            key="file_uploader",
        )
        if uploaded:
            st.button(
                "⚡ Process Documents",
                use_container_width=True,
                type="primary",
                on_click=_process,
                args=(uploaded,),
            )

        # Document inventory
        if st.session_state.document_stats:
            st.caption("INDEXED DOCUMENTS")
            for fn, cc in st.session_state.document_stats.items():
                ext = Path(fn).suffix.lower()
                icon = "📕" if ext == ".pdf" else "📝" if ext == ".txt" else "📘"
                st.markdown(
                    f'<div class="doc-c">{icon} <span class="fn">{fn}</span>'
                    f'<span class="cc">{cc} chunks</span></div>',
                    unsafe_allow_html=True,
                )
            st.caption(f"🔢 {st.session_state.vector_store.num_chunks} total vectors")

        st.divider()

        # Metrics
        st.caption("EXPERIMENT METRICS")
        s = get_experiment_summary()
        st.markdown(
            f'<div class="met-grid">'
            f'<div class="met-card"><div class="v">{s["total_queries"]}</div><div class="l">Queries</div></div>'
            f'<div class="met-card"><div class="v">{s["avg_faithfulness"]}</div><div class="l">Faithfulness</div></div>'
            f'<div class="met-card"><div class="v">{s["avg_latency_ms"]:.0f}ms</div><div class="l">Latency</div></div>'
            f'<div class="met-card"><div class="v" style="font-size:0.65rem">{s["best_model"]}</div><div class="l">Top Model</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.memory.clear()
            st.session_state.messages = []
            st.session_state.documents_processed = False
            st.session_state.document_stats = {}
            st.rerun()


# ════════════════════════════════════════════════════════════
# Document Processing
# ════════════════════════════════════════════════════════════

def _process(uploaded: list) -> None:
    """Ingest and index uploaded documents."""
    all_chunks = []
    stats: dict[str, int] = {}

    for uf in uploaded:
        try:
            r = ingest_file(uf, uf.name)
            if r.error:
                st.sidebar.error(f"❌ {uf.name}: {r.error}")
            else:
                all_chunks.extend(r.chunks)
                stats[r.filename] = r.num_chunks
        except Exception as e:
            st.sidebar.error(f"❌ {uf.name}: {e}")

    if all_chunks:
        st.session_state.vector_store.add_chunks(all_chunks)
        st.session_state.document_stats.update(stats)
        st.session_state.documents_processed = True


# ════════════════════════════════════════════════════════════
# Response Metadata
# ════════════════════════════════════════════════════════════

def _meta(m: dict) -> None:
    """Render metadata below assistant response."""
    if not m:
        return

    # Sources
    src = m.get("sources_display", "")
    if src and src != "No sources cited":
        st.info(f"📎 **Sources:** {src}")

    # Pills row
    pills = []
    if m.get("confidence") is not None:
        pills.append(f"🎯 **{m['confidence']}**/100")
    if m.get("query_type"):
        pills.append(f"🔍 {m['query_type'].replace('_', ' ').title()}")
    if m.get("model_used"):
        pills.append(f"⚡ {m['model_used']}")
    if m.get("latency_ms"):
        pills.append(f"⏱️ {m['latency_ms']:.0f}ms")
    if pills:
        cols = st.columns(len(pills))
        for i, t in enumerate(pills):
            cols[i].caption(t)

    # Eval scores
    scores = m.get("eval_scores")
    if scores and isinstance(scores, dict):
        st.caption("**Quality Scores**")
        ec = st.columns(len(scores))
        for i, (dim, sc) in enumerate(scores.items()):
            short = dim.replace("Hallucination Risk", "Halluc.").replace("Completeness", "Complete.")
            icon = "🟢" if sc >= 4 else "🟡" if sc >= 3 else "🔴"
            ec[i].markdown(f"{icon} **{sc}/5**\n\n`{short}`")


# ════════════════════════════════════════════════════════════
# Chat
# ════════════════════════════════════════════════════════════

def _chat() -> None:
    """Render conversation history."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                _meta(msg["metadata"])


# ════════════════════════════════════════════════════════════
# Welcome
# ════════════════════════════════════════════════════════════

def _welcome() -> None:
    """Show welcome card and examples."""
    st.markdown(
        '<div class="welcome">'
        '<h2>Upload documents to get started</h2>'
        '<p>Drop PDFs, DOCX, or TXT files in the sidebar — then ask anything.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    dom = st.session_state.selected_domain
    examples = _load_domain(dom)[1] if dom and dom != "None" else []
    if not examples:
        examples = [
            "Summarize the key points from all uploaded documents.",
            "Compare the main themes across these documents.",
            "Are there any contradictions between the documents?",
            "What are the most important findings?",
        ]

    st.markdown("#### 💡 Try these after uploading:")
    c1, c2 = st.columns(2)
    for i, q in enumerate(examples[:4]):
        with (c1 if i % 2 == 0 else c2):
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                _handle_query(q)


# ════════════════════════════════════════════════════════════
# Query Handling
# ════════════════════════════════════════════════════════════

def _handle_query(query: str) -> None:
    """Add user message and trigger rerun."""
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
            "content": f"⚠️ API key missing for {st.session_state.selected_model}.",
            "metadata": {},
        })
        st.rerun()
        return

    st.rerun()


def _pending() -> None:
    """Process pending query with status updates."""
    msgs = st.session_state.messages
    if not msgs or msgs[-1]["role"] != "user":
        return

    query = msgs[-1]["content"]
    model = st.session_state.selected_model

    if not get_available_models().get(model) or not st.session_state.vector_store.is_populated:
        return

    dom = st.session_state.selected_domain
    dp = _load_domain(dom)[0] if dom and dom != "None" else ""

    # Render history
    for msg in msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg:
                _meta(msg["metadata"])

    # Stream new response
    with st.chat_message("assistant"):
        status = st.status("Analyzing your query...", expanded=True)
        try:
            llm = create_llm(model)
            status.update(label="Searching documents...", state="running")
            time.sleep(0.1)
            status.update(label="Reasoning...", state="running")

            resp = run_pipeline(
                query=query, llm=llm,
                vector_store=st.session_state.vector_store,
                memory=st.session_state.memory,
                model_name=model, domain_prompt=dp,
            )

            status.update(label="✅ Done!", state="complete")
            log_query(query, resp)
            st.markdown(resp.answer)

            md = {
                "sources_display": format_sources_display(resp.sources),
                "confidence": resp.confidence,
                "query_type": resp.query_type,
                "model_used": resp.model_used,
                "eval_display": format_eval_scorecard(resp.eval_scores),
                "eval_scores": resp.eval_scores,
                "latency_ms": resp.latency_ms,
            }
            _meta(md)

            st.session_state.messages.append({
                "role": "assistant",
                "content": resp.answer,
                "metadata": md,
            })

        except Exception as e:
            logger.error("Query failed: %s", e)
            status.update(label="❌ Error", state="error")
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
    """Entry point."""
    st.set_page_config(
        page_title="Lumen — Document Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(SAFE_CSS, unsafe_allow_html=True)
    _init()
    _sidebar()

    # Header
    st.markdown(
        '<div class="lumen-brand">'
        '<h1>⚡ <span class="grad">Lumen</span></h1>'
        '<div class="sub">Multi-Document Reasoning Assistant — powered by LangGraph</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.documents_processed:
        _welcome()
    else:
        msgs = st.session_state.messages
        has_pending = (
            msgs
            and msgs[-1]["role"] == "user"
            and (len(msgs) < 2 or msgs[-2]["role"] != "user")
        )

        if has_pending:
            _pending()
        else:
            _chat()

        if q := st.chat_input("Ask Lumen anything about your documents..."):
            _handle_query(q)


if __name__ == "__main__":
    main()
