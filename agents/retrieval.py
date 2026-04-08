"""FAISS retrieval agent that fetches relevant chunks based on query type."""

from __future__ import annotations

import logging

from core.config import DEFAULT_TOP_K, COMPARISON_TOP_K, SUMMARIZATION_TOP_K
from core.vector_store import VectorStore, RetrievalResult

logger = logging.getLogger(__name__)

# Map query types to optimal k values
_K_MAP: dict[str, int] = {
    "simple_retrieval": DEFAULT_TOP_K,
    "comparison": COMPARISON_TOP_K,
    "contradiction_check": COMPARISON_TOP_K,
    "summarization": SUMMARIZATION_TOP_K,
    "confidence_check": DEFAULT_TOP_K,
}


def retrieve_chunks(
    vector_store: VectorStore,
    query: str,
    query_type: str,
) -> list[RetrievalResult]:
    """Retrieve relevant chunks from the vector store, adjusting k by query type."""
    top_k = _K_MAP.get(query_type, DEFAULT_TOP_K)

    try:
        results = vector_store.search(query, top_k=top_k)
        logger.info(
            "Retrieved %d chunks for query type '%s' (k=%d)",
            len(results), query_type, top_k,
        )
        return results
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        return []


def format_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks into a context string for the reasoning agent."""
    if not results:
        return "No relevant documents found."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        source = r.chunk.metadata.filename
        page = r.chunk.metadata.page_number
        page_str = f" (page {page})" if page else ""
        score = f"{r.score:.3f}"
        parts.append(
            f"[Source {i}: {source}{page_str} | Relevance: {score}]\n{r.chunk.text}"
        )

    return "\n\n---\n\n".join(parts)


def extract_sources(results: list[RetrievalResult]) -> list[dict]:
    """Extract unique source citations from retrieval results."""
    seen: set[tuple[str, int | None]] = set()
    sources: list[dict] = []

    for r in results:
        key = (r.chunk.metadata.filename, r.chunk.metadata.page_number)
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": r.chunk.metadata.filename,
                "page": r.chunk.metadata.page_number,
                "score": round(r.score, 3),
            })

    return sources


def compute_confidence(results: list[RetrievalResult]) -> int:
    """Compute a confidence score (0-100) based on retrieval similarity scores."""
    if not results:
        return 0

    scores = [r.score for r in results]
    top_score = max(scores)
    top_3_avg = sum(sorted(scores, reverse=True)[:3]) / min(3, len(scores))

    # Cosine similarity with normalized vectors typically ranges 0.2-0.85
    # Calibrate: 0.3 = low confidence, 0.5 = moderate, 0.7+ = high
    # Use top score (50%) + top-3 average (30%) + coverage bonus (20%)
    raw = 0.5 * top_score + 0.3 * top_3_avg + 0.2 * min(len(scores) / 5.0, 1.0)

    # Map from ~0.3-0.85 range to 20-95 range for human-readable scores
    calibrated = (raw - 0.25) / (0.80 - 0.25)  # normalize to 0-1
    confidence = int(min(max(calibrated * 80 + 20, 10), 98))  # scale to 20-98

    return confidence
