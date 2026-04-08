"""Tests for the document ingestion pipeline."""

from __future__ import annotations

import pytest

from core.ingestion import _chunk_text, ingest_file, DocumentChunk, IngestionResult


class TestChunking:
    """Tests for text chunking logic."""

    def test_short_text_single_chunk(self) -> None:
        """Short text should produce a single chunk."""
        text = "This is a short test sentence."
        chunks = _chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self) -> None:
        """Long text should be split into multiple overlapping chunks."""
        text = "word " * 1000  # ~1000 words ≈ many chunks
        chunks = _chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_chunks_not_empty(self) -> None:
        """No chunk should be empty."""
        text = "Hello world. " * 500
        chunks = _chunk_text(text, chunk_size=50, overlap=10)
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_overlap_exists(self) -> None:
        """Adjacent chunks should share some overlapping content."""
        text = "The quick brown fox jumped over the lazy dog. " * 100
        chunks = _chunk_text(text, chunk_size=50, overlap=15)
        if len(chunks) >= 2:
            # Check that the end of chunk 0 overlaps with start of chunk 1
            tail = chunks[0][-50:]
            head = chunks[1][:50]
            # At least some words should overlap
            tail_words = set(tail.split())
            head_words = set(head.split())
            assert len(tail_words & head_words) > 0


class TestIngestion:
    """Tests for file ingestion."""

    def test_unsupported_format(self) -> None:
        """Unsupported file types should return an error."""
        result = ingest_file(b"data", "file.xyz")
        assert result.error is not None
        assert "Unsupported" in result.error

    def test_txt_ingestion(self) -> None:
        """Plain text files should ingest correctly."""
        content = b"This is a test document with enough content to verify ingestion works properly."
        result = ingest_file(content, "test.txt")
        assert result.error is None
        assert result.num_chunks >= 1
        assert result.chunks[0].metadata.filename == "test.txt"

    def test_empty_txt(self) -> None:
        """Empty text files should produce zero chunks."""
        result = ingest_file(b"   ", "empty.txt")
        assert result.num_chunks == 0

    def test_metadata_populated(self) -> None:
        """Every chunk should have complete metadata."""
        content = b"Test content for metadata verification. " * 50
        result = ingest_file(content, "meta_test.txt")
        for chunk in result.chunks:
            assert chunk.metadata.filename == "meta_test.txt"
            assert chunk.metadata.chunk_index >= 0
            assert chunk.metadata.total_chunks == result.num_chunks
