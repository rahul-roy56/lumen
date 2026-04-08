"""Response formatting agent that structures the final output."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FormattedResponse:
    """Structured response with all metadata for display."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    confidence: int = 0
    query_type: str = "simple_retrieval"
    model_used: str = "Unknown"
    latency_ms: float = 0.0
    eval_scores: dict[str, int] = field(default_factory=dict)
    token_count: int = 0


def format_sources_display(sources: list[dict]) -> str:
    """Format source citations for display in the UI."""
    if not sources:
        return "No sources cited"

    # Group pages by filename
    file_pages: dict[str, list[int | None]] = {}
    for s in sources:
        fname = s["filename"]
        page = s.get("page")
        if fname not in file_pages:
            file_pages[fname] = []
        if page is not None:
            file_pages[fname].append(page)

    parts: list[str] = []
    for fname, pages in file_pages.items():
        if pages:
            page_str = ", ".join(str(p) for p in sorted(set(pages)))
            parts.append(f"📄 {fname} (pages {page_str})")
        else:
            parts.append(f"📄 {fname}")

    return " | ".join(parts)


def format_eval_scorecard(eval_scores: dict[str, int]) -> str:
    """Format evaluation scores as a mini scorecard."""
    if not eval_scores:
        return ""

    parts: list[str] = []
    for dim, score in eval_scores.items():
        stars = "⭐" * score
        parts.append(f"{dim} {stars}{score}/5")

    return " | ".join(parts)


def build_response(
    answer: str,
    sources: list[dict],
    confidence: int,
    query_type: str,
    model_used: str,
    latency_ms: float,
    eval_scores: dict[str, int] | None = None,
    token_count: int = 0,
) -> FormattedResponse:
    """Build a complete formatted response object."""
    return FormattedResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        query_type=query_type,
        model_used=model_used,
        latency_ms=latency_ms,
        eval_scores=eval_scores or {},
        token_count=token_count,
    )
