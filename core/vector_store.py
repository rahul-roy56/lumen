"""FAISS vector store for storing and retrieving document chunk embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import faiss
import numpy as np

from core.config import EMBEDDING_DIMENSION, DEFAULT_TOP_K
from core.embeddings import embed_texts, embed_query
from core.ingestion import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with score and chunk."""

    chunk: DocumentChunk
    score: float


class VectorStore:
    """FAISS-backed vector store for document chunks."""

    def __init__(self) -> None:
        """Initialize an empty FAISS index."""
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.chunks: list[DocumentChunk] = []
        self._is_populated: bool = False

    @property
    def is_populated(self) -> bool:
        """Check if the store has any chunks."""
        return self._is_populated and len(self.chunks) > 0

    @property
    def num_chunks(self) -> int:
        """Return total number of stored chunks."""
        return len(self.chunks)

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Embed and add chunks to the FAISS index, returning count added."""
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        try:
            embeddings = embed_texts(texts)
            # L2-normalize for cosine similarity via inner product
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            self.chunks.extend(chunks)
            self._is_populated = True
            logger.info("Added %d chunks to vector store (total: %d)", len(chunks), len(self.chunks))
            return len(chunks)
        except Exception as e:
            logger.error("Failed to add chunks to vector store: %s", e)
            raise RuntimeError(f"Vector store add failed: {e}") from e

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Search for the top-k most relevant chunks for a query."""
        if not self.is_populated:
            logger.warning("Search called on empty vector store")
            return []

        try:
            query_vec = embed_query(query).reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vec)
            scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

            results: list[RetrievalResult] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                results.append(RetrievalResult(chunk=self.chunks[idx], score=float(score)))

            return results
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            raise RuntimeError(f"Vector search failed: {e}") from e

    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.chunks = []
        self._is_populated = False
        logger.info("Vector store cleared")

    def get_document_names(self) -> list[str]:
        """Return unique document filenames in the store."""
        return list({c.metadata.filename for c in self.chunks})

    def get_document_stats(self) -> dict[str, int]:
        """Return chunk count per document."""
        stats: dict[str, int] = {}
        for chunk in self.chunks:
            name = chunk.metadata.filename
            stats[name] = stats.get(name, 0) + 1
        return stats
