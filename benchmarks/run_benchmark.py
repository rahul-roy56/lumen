"""
Lumen Benchmark Suite — Systematic evaluation of the multi-agent pipeline.

Runs 50 queries across 5 query types against sample documents, collects
eval scores, latency, and confidence for each, then saves results to JSON.

Usage:
    python benchmarks/run_benchmark.py --model "Llama 3 70B"
    python benchmarks/run_benchmark.py --model "Gemini 2.5 Flash"
    python benchmarks/run_benchmark.py --model all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from core.config import MODEL_OPTIONS
from core.ingestion import ingest_file
from core.vector_store import VectorStore
from core.memory import MemoryManager
from models.llm_factory import create_llm, get_available_models
from agents.graph import run_pipeline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).resolve().parent
SAMPLE_DOCS_DIR = BENCHMARKS_DIR / "sample_docs"
QUERIES_FILE = BENCHMARKS_DIR / "queries.json"
RESULTS_DIR = BENCHMARKS_DIR / "results"


def load_and_index_documents() -> VectorStore:
    """Load sample documents and build FAISS index."""
    vs = VectorStore()
    all_chunks = []

    print("📄 Loading sample documents...")
    for doc_path in sorted(SAMPLE_DOCS_DIR.glob("*.txt")):
        with open(doc_path, "rb") as f:
            result = ingest_file(f, doc_path.name)
        if result.error:
            print(f"  ❌ {doc_path.name}: {result.error}")
        else:
            all_chunks.extend(result.chunks)
            print(f"  ✅ {doc_path.name} → {result.num_chunks} chunks")

    if not all_chunks:
        print("❌ No documents loaded. Exiting.")
        sys.exit(1)

    print(f"\n🔍 Indexing {len(all_chunks)} chunks...")
    vs.add_chunks(all_chunks)
    print(f"✅ FAISS index built with {vs.num_chunks} vectors\n")
    return vs


def run_single_query(
    query_data: dict,
    llm,
    vector_store: VectorStore,
    memory: MemoryManager,
    model_name: str,
) -> dict:
    """Run a single benchmark query and collect metrics."""
    query = query_data["query"]
    query_id = query_data["id"]
    expected_type = query_data["type"]

    start = time.time()
    try:
        response = run_pipeline(
            query=query,
            llm=llm,
            vector_store=vector_store,
            memory=memory,
            model_name=model_name,
        )
        elapsed_ms = (time.time() - start) * 1000

        return {
            "id": query_id,
            "query": query,
            "expected_type": expected_type,
            "detected_type": response.query_type,
            "type_correct": response.query_type == expected_type,
            "confidence": response.confidence,
            "latency_ms": round(elapsed_ms, 1),
            "eval_scores": response.eval_scores,
            "faithfulness": response.eval_scores.get("Faithfulness", 0),
            "completeness": response.eval_scores.get("Completeness", 0),
            "clarity": response.eval_scores.get("Clarity", 0),
            "hallucination_risk": response.eval_scores.get("Hallucination Risk", 0),
            "num_sources": len(response.sources),
            "answer_length": len(response.answer),
            "model": model_name,
            "error": None,
        }
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return {
            "id": query_id,
            "query": query,
            "expected_type": expected_type,
            "detected_type": "error",
            "type_correct": False,
            "confidence": 0,
            "latency_ms": round(elapsed_ms, 1),
            "eval_scores": {},
            "faithfulness": 0,
            "completeness": 0,
            "clarity": 0,
            "hallucination_risk": 0,
            "num_sources": 0,
            "answer_length": 0,
            "model": model_name,
            "error": str(e),
        }


def run_benchmark(model_name: str) -> list[dict]:
    """Run full benchmark for a single model."""
    print(f"{'='*60}")
    print(f"🚀 Running benchmark with: {model_name}")
    print(f"{'='*60}\n")

    # Load queries
    queries = json.loads(QUERIES_FILE.read_text())["queries"]
    print(f"📋 Loaded {len(queries)} queries\n")

    # Build index
    vector_store = load_and_index_documents()

    # Create LLM
    try:
        llm = create_llm(model_name)
    except Exception as e:
        print(f"❌ Failed to create LLM '{model_name}': {e}")
        return []

    # Run queries
    results = []
    memory = MemoryManager()
    total = len(queries)

    for i, q in enumerate(queries, 1):
        print(f"  [{i:2d}/{total}] {q['type']:20s} | {q['query'][:50]}...", end="", flush=True)

        # Fresh memory per query to avoid cross-contamination
        memory = MemoryManager()
        result = run_single_query(q, llm, vector_store, memory, model_name)
        results.append(result)

        status = "✅" if result["error"] is None else "❌"
        faith = result["faithfulness"]
        latency = result["latency_ms"]
        print(f" → {status} faith={faith}/5 latency={latency:.0f}ms")

    return results


def compute_summary(results: list[dict], model_name: str) -> dict:
    """Compute aggregate statistics from benchmark results."""
    successful = [r for r in results if r["error"] is None]
    n = len(successful)

    if n == 0:
        return {"model": model_name, "total": len(results), "successful": 0}

    faithfulness_scores = [r["faithfulness"] for r in successful]
    completeness_scores = [r["completeness"] for r in successful]
    clarity_scores = [r["clarity"] for r in successful]
    hallucination_scores = [r["hallucination_risk"] for r in successful]
    latencies = sorted([r["latency_ms"] for r in successful])
    confidences = [r["confidence"] for r in successful]

    type_correct = sum(1 for r in successful if r["type_correct"])

    # Per query-type breakdown
    type_breakdown = {}
    for qtype in ["simple_retrieval", "comparison", "contradiction_check", "summarization", "confidence_check"]:
        type_results = [r for r in successful if r["expected_type"] == qtype]
        if type_results:
            type_breakdown[qtype] = {
                "count": len(type_results),
                "avg_faithfulness": round(sum(r["faithfulness"] for r in type_results) / len(type_results), 2),
                "avg_latency_ms": round(sum(r["latency_ms"] for r in type_results) / len(type_results), 1),
                "avg_confidence": round(sum(r["confidence"] for r in type_results) / len(type_results), 1),
            }

    return {
        "model": model_name,
        "total_queries": len(results),
        "successful": n,
        "failed": len(results) - n,
        "router_accuracy": round(type_correct / n * 100, 1),
        "avg_faithfulness": round(sum(faithfulness_scores) / n, 2),
        "avg_completeness": round(sum(completeness_scores) / n, 2),
        "avg_clarity": round(sum(clarity_scores) / n, 2),
        "avg_hallucination_risk": round(sum(hallucination_scores) / n, 2),
        "avg_overall": round(sum(sum(r["eval_scores"].values()) / max(len(r["eval_scores"]), 1) for r in successful) / n, 2),
        "avg_confidence": round(sum(confidences) / n, 1),
        "avg_latency_ms": round(sum(latencies) / n, 1),
        "p50_latency_ms": round(latencies[n // 2], 1),
        "p95_latency_ms": round(latencies[int(n * 0.95)], 1),
        "p99_latency_ms": round(latencies[int(n * 0.99)], 1),
        "min_latency_ms": round(latencies[0], 1),
        "max_latency_ms": round(latencies[-1], 1),
        "by_query_type": type_breakdown,
    }


def save_results(results: list[dict], summary: dict, model_name: str) -> Path:
    """Save benchmark results and summary to JSON."""
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    output = {
        "summary": summary,
        "results": results,
    }
    out_path = RESULTS_DIR / f"benchmark_{safe_name}.json"
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


def print_summary(summary: dict) -> None:
    """Print formatted summary to console."""
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK RESULTS — {summary['model']}")
    print(f"{'='*60}")
    print(f"  Queries:            {summary['successful']}/{summary['total_queries']} successful")
    print(f"  Router Accuracy:    {summary['router_accuracy']}%")
    print(f"")
    print(f"  ── Evaluation Scores (1-5) ──")
    print(f"  Faithfulness:       {summary['avg_faithfulness']}")
    print(f"  Completeness:       {summary['avg_completeness']}")
    print(f"  Clarity:            {summary['avg_clarity']}")
    print(f"  Hallucination Risk: {summary['avg_hallucination_risk']}")
    print(f"  Overall Average:    {summary['avg_overall']}")
    print(f"")
    print(f"  ── Latency ──")
    print(f"  Average:            {summary['avg_latency_ms']:.0f}ms")
    print(f"  P50:                {summary['p50_latency_ms']:.0f}ms")
    print(f"  P95:                {summary['p95_latency_ms']:.0f}ms")
    print(f"  P99:                {summary['p99_latency_ms']:.0f}ms")
    print(f"")
    print(f"  ── Confidence ──")
    print(f"  Average:            {summary['avg_confidence']}")
    print(f"")
    print(f"  ── By Query Type ──")
    for qtype, stats in summary.get("by_query_type", {}).items():
        print(f"  {qtype:25s} | faith={stats['avg_faithfulness']} | latency={stats['avg_latency_ms']:.0f}ms | conf={stats['avg_confidence']}")
    print(f"{'='*60}\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Lumen benchmark suite")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to benchmark: 'Llama 3 70B', 'Gemini 2.5 Flash', or 'all'",
    )
    args = parser.parse_args()

    available = get_available_models()
    print("\n⚡ Lumen Benchmark Suite")
    print(f"   Available models: {', '.join(m for m, a in available.items() if a)}\n")

    if args.model == "all":
        models = [m for m, a in available.items() if a]
    else:
        if not available.get(args.model):
            print(f"❌ Model '{args.model}' not available. Check your .env file.")
            sys.exit(1)
        models = [args.model]

    all_summaries = []

    for model_name in models:
        results = run_benchmark(model_name)
        if results:
            summary = compute_summary(results, model_name)
            save_path = save_results(results, summary, model_name)
            print_summary(summary)
            print(f"💾 Results saved to {save_path}\n")
            all_summaries.append(summary)

    # Save combined summary if multiple models
    if len(all_summaries) > 1:
        combined_path = RESULTS_DIR / "benchmark_comparison.json"
        combined_path.write_text(json.dumps(all_summaries, indent=2))
        print(f"📊 Model comparison saved to {combined_path}")
        print(f"\n{'='*60}")
        print("📊 MODEL COMPARISON")
        print(f"{'='*60}")
        for s in all_summaries:
            print(f"  {s['model']:25s} | faith={s['avg_faithfulness']} | latency={s['avg_latency_ms']:.0f}ms | overall={s['avg_overall']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
