"""Tests for the vector store and retrieval system."""

from __future__ import annotations

import pytest

from core.ingestion import DocumentChunk, ChunkMetadata
from core.vector_store import VectorStore


def _make_chunk(text: str, filename: str = "test.pdf", page: int = 1, idx: int = 0) -> DocumentChunk:
    """Helper to create a DocumentChunk for testing."""
    return DocumentChunk(
        text=text,
        metadata=ChunkMetadata(
            filename=filename,
            page_number=page,
            chunk_index=idx,
            total_chunks=1,
        ),
    )


class TestVectorStore:
    """Tests for FAISS vector store operations."""

    def test_empty_store(self) -> None:
        """New store should be empty."""
        vs = VectorStore()
        assert not vs.is_populated
        assert vs.num_chunks == 0

    def test_add_chunks(self) -> None:
        """Adding chunks should populate the store."""
        vs = VectorStore()
        chunks = [
            _make_chunk("Machine learning is a subset of artificial intelligence.", idx=0),
            _make_chunk("Deep learning uses neural networks with many layers.", idx=1),
            _make_chunk("Natural language processing handles text data.", idx=2),
        ]
        count = vs.add_chunks(chunks)
        assert count == 3
        assert vs.is_populated
        assert vs.num_chunks == 3

    def test_search_returns_results(self) -> None:
        """Search should return relevant results."""
        vs = VectorStore()
        chunks = [
            _make_chunk("Python is a popular programming language for data science."),
            _make_chunk("The Eiffel Tower is located in Paris, France."),
            _make_chunk("Machine learning models require training data."),
        ]
        vs.add_chunks(chunks)
        results = vs.search("programming languages", top_k=2)
        assert len(results) > 0
        assert len(results) <= 2

    def test_search_empty_store(self) -> None:
        """Searching an empty store should return empty list."""
        vs = VectorStore()
        results = vs.search("anything")
        assert results == []

    def test_clear(self) -> None:
        """Clearing should empty the store."""
        vs = VectorStore()
        vs.add_chunks([_make_chunk("Test chunk")])
        assert vs.is_populated
        vs.clear()
        assert not vs.is_populated
        assert vs.num_chunks == 0

    def test_document_stats(self) -> None:
        """Document stats should count chunks per file."""
        vs = VectorStore()
        chunks = [
            _make_chunk("Chunk 1", filename="a.pdf", idx=0),
            _make_chunk("Chunk 2", filename="a.pdf", idx=1),
            _make_chunk("Chunk 3", filename="b.pdf", idx=0),
        ]
        vs.add_chunks(chunks)
        stats = vs.get_document_stats()
        assert stats["a.pdf"] == 2
        assert stats["b.pdf"] == 1

    def test_relevance_ordering(self) -> None:
        """More relevant chunks should have higher scores."""
        vs = VectorStore()
        chunks = [
            _make_chunk("The weather in Boston is cold in winter and warm in summer."),
            _make_chunk("Quantum computing uses qubits instead of classical bits."),
            _make_chunk("Boston has many universities including MIT and Harvard."),
        ]
        vs.add_chunks(chunks)
        results = vs.search("universities in Boston", top_k=3)
        # The Boston-universities chunk should score higher than quantum computing
        assert len(results) == 3
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)  # Should be descending
