"""Centralized configuration constants for the Lumen project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DOMAINS_DIR: Path = PROJECT_ROOT / "domains"
ASSETS_DIR: Path = PROJECT_ROOT / "assets"

# ── Chunking ───────────────────────────────────────────────
CHUNK_SIZE: int = 800          # tokens per chunk (larger = more context per chunk)
CHUNK_OVERLAP: int = 150       # token overlap (larger = smoother boundaries)
CHARS_PER_TOKEN: int = 4       # rough char-to-token ratio

# ── Embeddings ─────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384

# ── Retrieval ──────────────────────────────────────────────
DEFAULT_TOP_K: int = 8
COMPARISON_TOP_K: int = 12
SUMMARIZATION_TOP_K: int = 15

# ── LLM Models ─────────────────────────────────────────────
MODEL_OPTIONS: dict[str, str] = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Llama 3 70B": "llama-3.3-70b-versatile",
}
DEFAULT_MODEL: str = "Gemini 2.5 Flash"
LLM_TEMPERATURE: float = 0.1
LLM_MAX_TOKENS: int = 2048

# ── Query Types ────────────────────────────────────────────
QUERY_TYPES: list[str] = [
    "simple_retrieval",
    "comparison",
    "contradiction_check",
    "summarization",
    "confidence_check",
]

# ── Evaluation Dimensions ──────────────────────────────────
EVAL_DIMENSIONS: list[str] = [
    "Faithfulness",
    "Completeness",
    "Clarity",
    "Hallucination Risk",
]

# ── MLflow ─────────────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME: str = "lumen-experiments"
MLFLOW_TRACKING_URI: str = "mlruns"

# ── Supported File Types ───────────────────────────────────
SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".txt", ".docx"]

# ── Domain Modules ─────────────────────────────────────────
AVAILABLE_DOMAINS: list[str] = ["finance", "healthcare", "legal", "hr_tech"]
