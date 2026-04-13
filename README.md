<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d9488,50:06b6d4,100:0d9488&height=180&section=header&text=⚡%20Lumen&fontSize=42&fontColor=fff&animation=fadeIn&fontAlignY=32&desc=Multi-Document%20Reasoning%20Assistant%20%7C%20LangGraph%20%7C%20FAISS%20%7C%20LLM-as-Judge&descSize=14&descAlignY=55" />

### Ask anything. Get answers from your documents — not the internet.

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-lumendocuments.streamlit.app-0d9488?style=for-the-badge)](https://lumendocuments.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-8B5CF6?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0d9488?style=for-the-badge)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-27ae60?style=for-the-badge)](LICENSE)

**Upload any document. Ask any question. Get cited, evaluated, trustworthy answers.**

[🚀 **Try Live Demo →**](https://lumendocuments.streamlit.app) · [📐 Architecture](#-architecture) · [⚡ Quick Start](#-quick-start)

</div>

---

## 🧠 Why Lumen?

RAG chatbots hallucinate. They don't cite sources. They can't compare information across documents or detect contradictions. And they never tell you how confident they are.

**Lumen is different.** It's a multi-agent reasoning system — not a simple retrieval chatbot. Every answer comes with source citations, a confidence score, and an automatic quality evaluation across four dimensions. Two free LLMs. Zero API costs. Full transparency.

---

## 📊 Key Results

After tuning chunk sizes, retrieval depth, reasoning prompts, and evaluator calibration:

| Metric | Before | After | Improvement |
|--------|:------:|:-----:|:-----------:|
| 🎯 Faithfulness | 🔴 1/5 | 🟢 **5/5** | +400% |
| 📋 Completeness | 🟡 3/5 | 🟢 **4/5** | +33% |
| ✍️ Clarity | 🟢 4/5 | 🟢 **5/5** | +25% |
| 🛡️ Hallucination Risk | 🔴 1/5 | 🟢 **5/5** | +400% |
| ⚡ Latency (Gemini) | ~15s | **~5.7s** | 2.6x faster |
| 🚀 Latency (Groq Llama 3) | — | **~2s** | 7.5x faster |

> **What drove the improvement:** Larger chunks (800 tokens, 150 overlap) → paragraph-aware splitting → stricter grounding prompts → context-aware evaluation → calibrated confidence scoring.

---

## 📐 Architecture

![Architecture](architecture.png)

Lumen uses a **4-node LangGraph pipeline** where each agent handles one specialized task:

```
User Query → 🔀 Router Agent → 🔍 Retrieval Agent → 🧠 Reasoning Agent → ⭐ Evaluator Agent → Response
                   ↓                   ↓                    ↓                     ↓
             Classifies           Searches FAISS        Generates            Scores quality
             query type            index              grounded answer       on 4 dimensions
```

> The graph is compiled once at module import using `StateGraph` with `TypedDict` state — nodes return partial dicts, LangGraph handles state merging automatically. This is the production pattern for LangGraph applications.

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-format ingestion** | PDF, DOCX, TXT with OCR fallback for scanned documents |
| 🔍 **Semantic search** | FAISS + sentence-transformers (all-MiniLM-L6-v2) — no API key needed |
| 🔀 **5 query types** | Simple retrieval, comparison, contradiction detection, summarization, confidence check |
| 📎 **Source citations** | Every answer cites exact document names and page numbers |
| 🎯 **Confidence scoring** | 0–100 score calibrated from retrieval similarity metrics |
| ⭐ **LLM-as-judge** | Auto-evaluation on Faithfulness, Completeness, Clarity, Hallucination Risk |
| 🤖 **Multi-model** | Gemini 2.5 Flash + Llama 3 70B (Groq) — both completely free |
| 💬 **Conversation memory** | Follow-up questions reference previous answers |
| 🏢 **4 domain modules** | Pre-configured prompts for Finance, Healthcare, Legal, HR/Talent |
| 📊 **Experiment tracking** | MLflow logs every query with params, metrics, and artifacts |
| 🛡️ **Graceful fallback** | Missing API keys disable models instead of crashing |

---

## 🚀 Live Demo

**[https://lumendocuments.streamlit.app](https://lumendocuments.streamlit.app)**

| Feature | Description |
|---------|-------------|
| 🤖 Model selector | Switch between Llama 3 70B (Groq) and Gemini 2.5 Flash |
| 🏢 Domain selector | Choose Finance, Healthcare, Legal, HR or General mode |
| 📄 Document upload | Drag & drop PDFs, DOCX, TXT — process with one click |
| 💬 Chat interface | Ask questions, get cited answers with eval scores |
| 📊 Live metrics | Sidebar shows query count, avg faithfulness, latency, top model |
| 🗑️ Session control | Clear everything and start fresh |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| 🔗 Orchestration | **LangGraph** — compiled StateGraph, TypedDict state, 4 nodes, directed edges |
| 🧬 Embeddings | **sentence-transformers** (all-MiniLM-L6-v2) — free, local, no API |
| 🗄️ Vector Store | **FAISS** (IndexFlatIP with L2-normalized cosine similarity) |
| 🤖 LLMs | **Groq Llama 3 70B** (~2s) · **Google Gemini 2.5 Flash** (~5.7s) — both free |
| ⭐ Evaluation | **LLM-as-judge** with context-aware rubric scoring (4 dimensions) |
| 📊 Tracking | **MLflow** — parameters, metrics, artifacts per query |
| 🖥️ Frontend | **Streamlit** with custom teal/mint glassmorphism theme |
| 📄 PDF Extraction | **PyMuPDF** + pytesseract OCR fallback |
| ✅ Testing | **pytest** — 25 tests covering ingestion, retrieval, and agents |

---

## ⚡ Quick Start

### 1️⃣ Clone the repo
```bash
git clone https://github.com/rahul-roy56/lumen.git
cd lumen
```

### 2️⃣ Setup environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Add API keys (both are free)
```bash
cp .env.example .env
# Add your Groq key → https://console.groq.com (free)
# Add your Gemini key → https://aistudio.google.com/apikey (free)
```

### 4️⃣ Run tests
```bash
pytest tests/ -v   # 25 tests, all should pass
```

### 5️⃣ Launch Lumen
```bash
streamlit run app.py
```
App opens at `http://localhost:8501` → upload documents → ask questions → get cited answers.

### 6️⃣ (Optional) MLflow dashboard
```bash
mlflow ui --backend-store-uri mlruns
# Open http://localhost:5000
```

---

## 🏢 Domain Modules

| Domain | Focus Area | Example Query |
|--------|-----------|---------------|
| 💰 Finance | SEC filings, earnings reports | *"Compare gross margins across these reports"* |
| 🏥 Healthcare | Clinical notes, medical literature | *"What treatment protocols are recommended?"* |
| ⚖️ Legal | Contracts, policy documents | *"Any contradictions between the NDA and MSA?"* |
| 👥 HR & Talent | Job descriptions, resumes | *"What are the required qualifications?"* |

> Select a domain to load specialized system prompts and example questions. Upload your own documents on top of any domain.

---

## 🔬 How It Works

```
1. 📄 Upload     → Drop PDFs, DOCX, or TXT files
2. ✂️ Chunk      → 800-token chunks, 150-token overlap, paragraph-aware splitting
3. 🧬 Embed      → all-MiniLM-L6-v2 sentence transformer → FAISS index
4. 🔀 Route      → Router Agent classifies query into 1 of 5 types
5. 🔍 Retrieve   → Top-k chunks (8–15) fetched with similarity scores
6. 🧠 Reason     → Grounded answer generation with mandatory source citations
7. ⭐ Evaluate   → LLM-as-judge scores on 4 quality dimensions
8. 📊 Track      → MLflow logs params, metrics, and full prompt/response artifacts
```

---

## 🏗️ Project Structure

```
📂 lumen/
├── 📝 app.py                    # Streamlit frontend
├── 📂 core/
│   ├── 📝 config.py             # All constants (chunk size, top-k, models)
│   ├── 📝 ingestion.py          # Document loading, paragraph-aware chunking
│   ├── 📝 embeddings.py         # Sentence-transformer embeddings
│   ├── 📝 vector_store.py       # FAISS index operations
│   └── 📝 memory.py             # Conversation history management
├── 📂 agents/
│   ├── 📝 router.py             # Query classification (5 types)
│   ├── 📝 retrieval.py          # Adaptive top-k + confidence scoring
│   ├── 📝 reasoning.py          # Grounded response with citations
│   ├── 📝 evaluator.py          # LLM-as-judge (context-aware rubrics)
│   ├── 📝 formatter.py          # Response structuring
│   └── 📝 graph.py              # LangGraph StateGraph definition
├── 📂 models/
│   └── 📝 llm_factory.py        # Factory pattern for Gemini / Groq
├── 📂 tracking/
│   └── 📝 mlflow_logger.py      # MLflow logging (graceful fallback)
├── 📂 domains/                   # 4 domain configs + system prompts
├── 📂 tests/                     # 25 pytest tests
├── 📝 .streamlit/config.toml    # Theme configuration
├── 📝 requirements.txt          # Dependencies
└── 📝 .env.example              # API key template
```

---

## 🗺️ Future Roadmap

| Feature | Status |
|---------|--------|
| 🐳 Docker + docker-compose | 🔜 Next |
| ⚙️ CI/CD with GitHub Actions | 🔜 Next |
| 📊 50-query benchmark report | 🔜 Planned |
| 🧪 A/B testing (Gemini vs Groq) | 🔜 Planned |
| 🔀 Hybrid search (FAISS + BM25) | 🔜 Planned |
| 🌊 Token-by-token streaming | 💡 Future |
| 🖼️ Multi-modal (tables, charts, images) | 💡 Future |

---

## 📝 License

This project is **open-source** under the [MIT License](LICENSE). Feel free to fork, improve, or contribute! 💡

---

<div align="center">

### 🔗 Connect with me

[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rahulroy0499/)
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/rahul-roy56)
[![Email](https://img.shields.io/badge/-Email-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:roy.rah@northeastern.edu)

**Rahul Roy** — MS Applied Machine Intelligence @ Northeastern University, Boston

⚡ Stop reading documents. Start asking questions. 🚀

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d9488,50:06b6d4,100:0d9488&height=100&section=footer" />

</div>
