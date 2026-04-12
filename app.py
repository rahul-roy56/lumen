"""Lumen — Intelligent Multi-Document Reasoning Assistant (Streamlit Frontend)."""

from __future__ import annotations

import json
import logging
import os
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
# CSS
# ════════════════════════════════════════════════════════════

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Cursor fix ── */
*,*::before,*::after{cursor:default!important}
a,button,[role="button"],label,select,option,
input[type="file"],input[type="submit"],input[type="button"],
.stButton>button,[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"]{cursor:pointer!important}
input,textarea,[contenteditable="true"],
[data-testid="stChatInput"] textarea{cursor:text!important}

/* ── Main area — light ── */
.stApp{
    background:linear-gradient(165deg,#F8F9FC 0%,#EEF0F7 40%,#E8EAF2 100%)!important;
    font-family:'Plus Jakarta Sans',sans-serif!important;
    color:#1a1a2e!important;
}

/* ── Sidebar — dark ── */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#1a1a2e 0%,#16162a 50%,#111128 100%)!important;
    border-right:none!important;
    box-shadow:4px 0 24px rgba(0,0,0,0.15)!important;
    min-width:290px!important;
    width:290px!important;
}
[data-testid="stSidebar"]>div:first-child{
    width:290px!important;
    padding:1rem 1.2rem!important;
}
[data-testid="stSidebar"] *{
    font-family:'Plus Jakarta Sans',sans-serif!important;
    color:#c8cad0!important;
}
[data-testid="stSidebar"] hr{
    border-color:rgba(255,255,255,0.06)!important;
    margin:0.8rem 0!important;
}

/* ── Kill the collapse arrow text ── */
button[kind="header"],
[data-testid="collapsedControl"],
[data-testid="stSidebar"] button[kind="header"]{
    display:none!important;visibility:hidden!important;
}

/* ── Brand ── */
.brand{padding:0.6rem 0 0.6rem 0;text-align:center}
.brand .logo{font-size:1.5rem;font-weight:800;letter-spacing:-0.03em;color:#fff!important}
.brand .logo b{
    background:linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.brand .tag{font-size:0.6rem;color:#6c6c8a!important;margin-top:0.2rem;
    letter-spacing:0.12em;text-transform:uppercase;font-weight:600}

/* ── Section heading ── */
.sh{font-size:0.58rem;font-weight:700;text-transform:uppercase;
    letter-spacing:0.14em;color:#6c6c8a!important;margin:0.8rem 0 0.4rem 0;
    display:flex;align-items:center;gap:0.4rem}
.sh::after{content:'';flex:1;height:1px;background:rgba(255,255,255,0.06)}

/* ── Doc cards ── */
.dc{display:flex;align-items:center;gap:0.5rem;padding:0.45rem 0.6rem;
    border-radius:8px;background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.06);margin-bottom:0.3rem}
.dc:hover{background:rgba(102,126,234,0.08);border-color:rgba(102,126,234,0.2)}
.dc .i{font-size:0.9rem}
.dc .n{flex:1;font-size:0.72rem;font-weight:500;color:#e0e0e8!important;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.dc .b{font-size:0.58rem;font-family:'IBM Plex Mono',monospace;color:#667eea!important;
    background:rgba(102,126,234,0.1);padding:0.1rem 0.35rem;border-radius:4px}

/* ── Metrics ── */
.mg{display:grid;grid-template-columns:1fr 1fr;gap:0.4rem}
.mc{padding:0.5rem 0.3rem;border-radius:8px;background:rgba(255,255,255,0.03);
    border:1px solid rgba(255,255,255,0.06);text-align:center}
.mc .v{font-size:1.1rem;font-weight:700;font-family:'IBM Plex Mono',monospace}
.mc .l{font-size:0.5rem;color:#6c6c8a!important;text-transform:uppercase;
    letter-spacing:0.08em;margin-top:0.08rem;font-weight:600}

/* ── Main header ── */
.mh{padding:1.2rem 0 0.5rem 0}
.mh h1{font-family:'Plus Jakarta Sans',sans-serif!important;font-weight:800!important;
    font-size:2rem!important;color:#1a1a2e!important;margin:0!important;
    line-height:1.15!important;letter-spacing:-0.03em}
.mh h1 em{font-style:normal;
    background:linear-gradient(135deg,#667eea,#764ba2);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.mh .s{font-size:0.85rem;color:#7a7a96;margin-top:0.2rem}

/* ── Welcome card ── */
.wc{padding:2.5rem 2rem;border-radius:16px;background:#fff;
    border:1px solid #e2e4ec;
    box-shadow:0 1px 3px rgba(0,0,0,0.04),0 8px 24px rgba(102,126,234,0.06);
    text-align:center;margin:1.5rem 0;position:relative;overflow:hidden}
.wc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
    background:linear-gradient(90deg,#667eea,#764ba2,#667eea);
    background-size:200% auto;animation:sh 3s ease infinite}
@keyframes sh{0%{background-position:0% center}100%{background-position:200% center}}
.wc h2{font-weight:700!important;font-size:1.2rem!important;color:#1a1a2e!important;
    margin:0 0 0.3rem 0!important}
.wc p{color:#7a7a96;font-size:0.88rem;margin:0}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"]{
    background:#fff!important;border:1px solid #e8eaf0!important;
    border-radius:12px!important;box-shadow:0 1px 3px rgba(0,0,0,0.03)!important;
    margin-bottom:0.75rem!important;padding:1rem!important}
[data-testid="stChatMessage"] *{color:#1a1a2e!important}
[data-testid="stChatMessage"] .stCaption p{color:#7a7a96!important}

/* ── Primary buttons ── */
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#667eea,#764ba2)!important;
    border:none!important;border-radius:8px!important;color:#fff!important;
    font-weight:600!important;font-size:0.82rem!important;
    padding:0.5rem 1rem!important;
    box-shadow:0 2px 8px rgba(102,126,234,0.3)!important;
    transition:all 0.2s ease!important}
.stButton>button[kind="primary"]:hover{
    box-shadow:0 4px 16px rgba(102,126,234,0.45)!important;
    transform:translateY(-1px)!important}

/* ── Secondary buttons ── */
.stButton>button:not([kind="primary"]){
    background:#fff!important;border:1px solid #d8dae4!important;
    border-radius:8px!important;color:#1a1a2e!important;
    font-weight:500!important;font-size:0.8rem!important;
    box-shadow:0 1px 2px rgba(0,0,0,0.04)!important;
    transition:all 0.15s ease!important}
.stButton>button:not([kind="primary"]):hover{
    border-color:#667eea!important;color:#667eea!important;
    box-shadow:0 2px 8px rgba(102,126,234,0.12)!important}

/* ── Sidebar button overrides ── */
[data-testid="stSidebar"] .stButton>button[kind="primary"]{color:#fff!important}
[data-testid="stSidebar"] .stButton>button:not([kind="primary"]){
    background:rgba(255,255,255,0.05)!important;
    border:1px solid rgba(255,255,255,0.1)!important;color:#c8cad0!important}
[data-testid="stSidebar"] .stButton>button:not([kind="primary"]):hover{
    background:rgba(102,126,234,0.1)!important;
    border-color:rgba(102,126,234,0.3)!important;color:#667eea!important}

/* ── Chat input ── */
[data-testid="stChatInput"]{background:#fff!important;border-radius:12px!important;
    box-shadow:0 2px 12px rgba(0,0,0,0.06)!important}
[data-testid="stChatInput"] textarea{
    font-family:'Plus Jakarta Sans',sans-serif!important;
    border:1px solid #e2e4ec!important;border-radius:12px!important;
    color:#1a1a2e!important;background:#fff!important}
[data-testid="stChatInput"] textarea:focus{
    border-color:#667eea!important;
    box-shadow:0 0 0 3px rgba(102,126,234,0.12)!important}

/* ── Sidebar select ── */
[data-testid="stSidebar"] [data-testid="stSelectbox"]>div>div{
    background:rgba(255,255,255,0.05)!important;
    border:1px solid rgba(255,255,255,0.1)!important;border-radius:8px!important}

/* ── File uploader — FULL FIX ── */
[data-testid="stFileUploader"]>label{display:none!important}
[data-testid="stFileUploader"]>div{padding-top:0!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section{
    border:1px dashed rgba(102,126,234,0.3)!important;border-radius:10px!important;
    background:rgba(102,126,234,0.04)!important;padding:0.7rem!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section>button{
    width:100%!important;background:rgba(102,126,234,0.12)!important;
    border:1px solid rgba(102,126,234,0.25)!important;border-radius:6px!important;
    color:#a0a0ff!important;font-size:0.75rem!important;font-weight:500!important;
    padding:0.35rem 0.8rem!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section>span{
    font-size:0.65rem!important;color:#6c6c8a!important}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section>div[data-testid="stMarkdownContainer"]{
    display:none!important}

/* ── Misc ── */
.stProgress>div>div{background:linear-gradient(90deg,#667eea,#764ba2)!important;border-radius:4px!important}
[data-testid="stAlert"]{border-radius:8px!important;font-size:0.82rem!important}
[data-testid="stStatusWidget"]{background:#fff!important;border:1px solid #e2e4ec!important;
    border-radius:10px!important;box-shadow:0 1px 4px rgba(0,0,0,0.04)!important}
#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}
</style>
"""

# ════════════════════════════════════════════════════════════
# Session
# ════════════════════════════════════════════════════════════

def _init():
    for k, v in {
        "vector_store": VectorStore(), "memory": MemoryManager(),
        "messages": [], "documents_processed": False,
        "document_stats": {}, "selected_domain": "None",
        "selected_model": list(MODEL_OPTIONS.keys())[0],
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _load_domain(d):
    dp = DOMAINS_DIR / d
    sp = (dp / "system_prompt.txt").read_text() if (dp / "system_prompt.txt").exists() else ""
    eq = json.loads((dp / "example_questions.json").read_text()).get("example_questions", []) if (dp / "example_questions.json").exists() else []
    return sp, eq

# ════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════

def _sidebar():
    with st.sidebar:
        st.markdown('<div class="brand"><div class="logo">⚡ <b>Lumen</b></div><div class="tag">Document Intelligence</div></div>', unsafe_allow_html=True)
        st.divider()

        st.markdown('<div class="sh">Domain</div>', unsafe_allow_html=True)
        opts = ["None"] + AVAILABLE_DOMAINS
        lbls = {"None": "🌐 General"}
        for d in AVAILABLE_DOMAINS:
            f = DOMAINS_DIR / d / "example_questions.json"
            if f.exists():
                data = json.loads(f.read_text())
                lbls[d] = f"{data.get('icon','📁')} {data.get('display_name',d)}"
        st.selectbox("Domain", opts, format_func=lambda x: lbls.get(x, x), key="selected_domain", label_visibility="collapsed")

        st.markdown('<div class="sh">Model</div>', unsafe_allow_html=True)
        avail = get_available_models()
        ml = {m: f"🟢 {m}" if avail.get(m) else f"⚪ {m} (no key)" for m in MODEL_OPTIONS}
        st.selectbox("Model", list(MODEL_OPTIONS.keys()), format_func=lambda x: ml.get(x, x), key="selected_model", label_visibility="collapsed")
        if not avail.get(st.session_state.selected_model):
            st.warning(f"API key missing for {st.session_state.selected_model}")

        st.divider()
        st.markdown('<div class="sh">Documents</div>', unsafe_allow_html=True)
        files = st.file_uploader("Upload", type=["pdf","txt","docx"], accept_multiple_files=True, key="file_uploader", label_visibility="collapsed")
        if files and st.button("⚡ Process Documents", use_container_width=True, type="primary"):
            _process(files)

        if st.session_state.document_stats:
            for fn, c in st.session_state.document_stats.items():
                ext = Path(fn).suffix.lower()
                ic = "📕" if ext == ".pdf" else "📝" if ext == ".txt" else "📘"
                st.markdown(f'<div class="dc"><span class="i">{ic}</span><span class="n">{fn}</span><span class="b">{c}</span></div>', unsafe_allow_html=True)
            st.caption(f"{st.session_state.vector_store.num_chunks} vectors indexed")

        st.divider()
        st.markdown('<div class="sh">Metrics</div>', unsafe_allow_html=True)
        s = get_experiment_summary()
        st.markdown(f'<div class="mg"><div class="mc"><div class="v" style="color:#667eea">{s["total_queries"]}</div><div class="l">Queries</div></div><div class="mc"><div class="v" style="color:#10b981">{s["avg_faithfulness"]}</div><div class="l">Faith.</div></div><div class="mc"><div class="v" style="color:#764ba2">{s["avg_latency_ms"]:.0f}<span style="font-size:0.5rem">ms</span></div><div class="l">Latency</div></div><div class="mc"><div class="v" style="color:#f59e0b;font-size:0.65rem">{s["best_model"]}</div><div class="l">Top Model</div></div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Session", use_container_width=True):
            st.session_state.vector_store.clear(); st.session_state.memory.clear()
            st.session_state.messages = []; st.session_state.documents_processed = False
            st.session_state.document_stats = {}; st.rerun()

# ════════════════════════════════════════════════════════════
# Processing
# ════════════════════════════════════════════════════════════

def _process(files):
    prog = st.sidebar.progress(0, text="Processing...")
    chunks, stats = [], {}
    for i, f in enumerate(files):
        prog.progress(i / len(files), text=f"Reading {f.name}...")
        try:
            r = ingest_file(f, f.name)
            if r.error: st.sidebar.error(f"❌ {f.name}: {r.error}")
            else: chunks.extend(r.chunks); stats[r.filename] = r.num_chunks
        except Exception as e: st.sidebar.error(f"❌ {f.name}: {e}")
    if chunks:
        prog.progress(0.85, text="Building index...")
        st.session_state.vector_store.add_chunks(chunks)
        st.session_state.document_stats.update(stats)
        st.session_state.documents_processed = True
    prog.progress(1.0, text="✅ Ready!"); time.sleep(0.4); st.rerun()

# ════════════════════════════════════════════════════════════
# Metadata
# ════════════════════════════════════════════════════════════

def _meta(m):
    if not m: return
    if m.get("sources_display") and m["sources_display"] != "No sources cited":
        st.info(f"📎 **Sources:** {m['sources_display']}")
    pills = []
    if m.get("confidence") is not None: pills.append(f"🎯 **{m['confidence']}**/100")
    if m.get("query_type"): pills.append(f"🔍 {m['query_type'].replace('_',' ').title()}")
    if m.get("model_used"): pills.append(f"⚡ {m['model_used']}")
    if m.get("latency_ms"): pills.append(f"⏱️ {m['latency_ms']:.0f}ms")
    if pills:
        cols = st.columns(len(pills))
        for i, t in enumerate(pills): cols[i].caption(t)
    if m.get("eval_scores") and isinstance(m["eval_scores"], dict):
        st.caption("**Quality Scores**")
        ec = st.columns(len(m["eval_scores"]))
        for i, (d, sc) in enumerate(m["eval_scores"].items()):
            short = d.replace("Hallucination Risk","Halluc.").replace("Completeness","Complete.")
            ic = "🟢" if sc >= 4 else "🟡" if sc >= 3 else "🔴"
            ec[i].markdown(f"{ic} **{sc}/5**\n\n`{short}`")

# ════════════════════════════════════════════════════════════
# Chat + Welcome
# ════════════════════════════════════════════════════════════

def _chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="user" if msg["role"]=="user" else "🔆"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metadata" in msg: _meta(msg["metadata"])

def _welcome():
    st.markdown('<div class="wc"><h2>Upload documents to get started</h2><p>Drop PDFs, DOCX, or TXT files in the sidebar — then ask anything.</p></div>', unsafe_allow_html=True)
    dom = st.session_state.selected_domain
    ex = _load_domain(dom)[1] if dom and dom != "None" else []
    if not ex: ex = ["Summarize the key points from all uploaded documents.","Compare the main themes across these documents.","Are there any contradictions between the documents?","What are the most important findings?"]
    st.markdown("#### 💡 Try these after uploading:")
    cols = st.columns(2)
    for i, q in enumerate(ex[:4]):
        with cols[i%2]:
            if st.button(q, key=f"ex_{i}", use_container_width=True): _query(q)

# ════════════════════════════════════════════════════════════
# Query
# ════════════════════════════════════════════════════════════

def _query(q):
    st.session_state.messages.append({"role":"user","content":q})
    if not st.session_state.vector_store.is_populated:
        st.session_state.messages.append({"role":"assistant","content":"Please upload and process documents first.","metadata":{}}); st.rerun(); return
    if not get_available_models().get(st.session_state.selected_model):
        st.session_state.messages.append({"role":"assistant","content":f"⚠️ {st.session_state.selected_model} API key missing.","metadata":{}}); st.rerun(); return
    st.rerun()

def _pending():
    msgs = st.session_state.messages
    if not msgs or msgs[-1]["role"] != "user": return
    q = msgs[-1]["content"]; mn = st.session_state.selected_model
    if not get_available_models().get(mn) or not st.session_state.vector_store.is_populated: return
    dom = st.session_state.selected_domain
    dp = _load_domain(dom)[0] if dom and dom != "None" else ""
    for msg in msgs:
        with st.chat_message(msg["role"], avatar="user" if msg["role"]=="user" else "🔆"):
            st.markdown(msg["content"])
            if msg["role"]=="assistant" and "metadata" in msg: _meta(msg["metadata"])
    with st.chat_message("assistant", avatar="🔆"):
        status = st.status("Analyzing...", expanded=True)
        try:
            llm = create_llm(mn)
            status.update(label="Searching documents...", state="running"); time.sleep(0.1)
            status.update(label="Reasoning...", state="running")
            resp = run_pipeline(query=q, llm=llm, vector_store=st.session_state.vector_store,
                                memory=st.session_state.memory, model_name=mn, domain_prompt=dp)
            status.update(label="Done!", state="complete")
            log_query(q, resp); st.markdown(resp.answer)
            md = {"sources_display":format_sources_display(resp.sources),"confidence":resp.confidence,
                  "query_type":resp.query_type,"model_used":resp.model_used,
                  "eval_display":format_eval_scorecard(resp.eval_scores),
                  "eval_scores":resp.eval_scores,"latency_ms":resp.latency_ms}
            _meta(md); st.session_state.messages.append({"role":"assistant","content":resp.answer,"metadata":md})
        except Exception as e:
            logger.error("Query failed: %s", e); status.update(label="Error", state="error")
            st.markdown(f"Error: {e}"); st.session_state.messages.append({"role":"assistant","content":f"Error: {e}","metadata":{}})

# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Lumen — Document Intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    _init(); _sidebar()
    st.markdown('<div class="mh"><h1>⚡ <em>Lumen</em></h1><div class="s">Multi-Document Reasoning Assistant — powered by LangGraph</div></div>', unsafe_allow_html=True)
    if not st.session_state.documents_processed: _welcome()
    else:
        msgs = st.session_state.messages
        pending = msgs and msgs[-1]["role"]=="user" and (len(msgs)<2 or msgs[-2]["role"]!="user")
        if pending: _pending()
        else: _chat()
        if q := st.chat_input("Ask Lumen anything about your documents..."): _query(q)

if __name__ == "__main__":
    main()
