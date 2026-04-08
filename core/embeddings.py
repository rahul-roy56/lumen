"""Sentence-transformer embedding generation for document chunks."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import EMBEDDING_MODEL

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into a 2-D numpy array of shape (n, dim)."""
    if not texts:
        return np.array([], dtype=np.float32).reshape(0, 0)
    try:
        model = _load_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.astype(np.float32)
    except Exception as e:
        logger.error("Embedding failed: %s", e)
        raise RuntimeError(f"Failed to generate embeddings: {e}") from e


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string into a 1-D numpy array."""
    result = embed_texts([query])
    return result[0]
