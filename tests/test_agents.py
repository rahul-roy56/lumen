"""Tests for agent modules (router, retrieval helpers, formatter, evaluator)."""

from __future__ import annotations

import pytest

from agents.retrieval import compute_confidence, extract_sources, format_context
from agents.formatter import (
    build_response,
    format_sources_display,
    format_eval_scorecard,
    FormattedResponse,
)
from agents.evaluator import compute_overall_score
from core.ingestion import DocumentChunk, ChunkMetadata
from core.vector_store import RetrievalResult


def _make_retrieval_result(
    text: str, filename: str = "doc.pdf", page: int = 1, score: float = 0.85
) -> RetrievalResult:
    """Helper to create a RetrievalResult."""
    chunk = DocumentChunk(
        text=text,
        metadata=ChunkMetadata(filename=filename, page_number=page, chunk_index=0, total_chunks=1),
    )
    return RetrievalResult(chunk=chunk, score=score)


class TestRetrievalHelpers:
    """Tests for retrieval utility functions."""

    def test_compute_confidence_empty(self) -> None:
        """Empty results should give zero confidence."""
        assert compute_confidence([]) == 0

    def test_compute_confidence_high_scores(self) -> None:
        """High similarity scores should produce high confidence."""
        results = [
            _make_retrieval_result("a", score=0.95),
            _make_retrieval_result("b", score=0.90),
        ]
        conf = compute_confidence(results)
        assert conf >= 80

    def test_extract_sources_dedup(self) -> None:
        """Sources should be deduplicated by filename+page."""
        results = [
            _make_retrieval_result("a", filename="f.pdf", page=1),
            _make_retrieval_result("b", filename="f.pdf", page=1),
            _make_retrieval_result("c", filename="f.pdf", page=2),
        ]
        sources = extract_sources(results)
        assert len(sources) == 2  # page 1 and page 2

    def test_format_context(self) -> None:
        """Formatted context should include source info."""
        results = [_make_retrieval_result("Test content", filename="report.pdf", page=3)]
        ctx = format_context(results)
        assert "report.pdf" in ctx
        assert "page 3" in ctx
        assert "Test content" in ctx


class TestFormatter:
    """Tests for the response formatter."""

    def test_format_sources_display(self) -> None:
        """Source display should group pages by filename."""
        sources = [
            {"filename": "a.pdf", "page": 1, "score": 0.9},
            {"filename": "a.pdf", "page": 3, "score": 0.8},
            {"filename": "b.txt", "page": None, "score": 0.7},
        ]
        display = format_sources_display(sources)
        assert "a.pdf" in display
        assert "b.txt" in display

    def test_format_eval_scorecard(self) -> None:
        """Scorecard should include all dimensions."""
        scores = {"Faithfulness": 4, "Completeness": 5, "Clarity": 3, "Hallucination Risk": 4}
        card = format_eval_scorecard(scores)
        assert "Faithfulness" in card
        assert "5/5" in card

    def test_build_response(self) -> None:
        """build_response should create a complete FormattedResponse."""
        resp = build_response(
            answer="Test answer",
            sources=[{"filename": "x.pdf", "page": 1, "score": 0.9}],
            confidence=85,
            query_type="simple_retrieval",
            model_used="GPT-4o",
            latency_ms=1200.5,
            eval_scores={"Faithfulness": 4},
            token_count=500,
        )
        assert isinstance(resp, FormattedResponse)
        assert resp.answer == "Test answer"
        assert resp.confidence == 85
        assert resp.model_used == "GPT-4o"


class TestEvaluator:
    """Tests for evaluation utility functions."""

    def test_overall_score_empty(self) -> None:
        """Empty scores should return 0."""
        assert compute_overall_score({}) == 0.0

    def test_overall_score_average(self) -> None:
        """Overall score should be the average of dimensions."""
        scores = {"Faithfulness": 4, "Completeness": 5, "Clarity": 3, "Hallucination Risk": 4}
        avg = compute_overall_score(scores)
        assert avg == 4.0

    def test_overall_score_rounding(self) -> None:
        """Overall score should be rounded to 2 decimal places."""
        scores = {"Faithfulness": 3, "Completeness": 4, "Clarity": 5}
        avg = compute_overall_score(scores)
        assert avg == 4.0
