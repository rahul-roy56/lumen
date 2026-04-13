"""Factory pattern for initializing Gemini 2.5 Flash and Llama 3 70B (via Groq)."""

from __future__ import annotations

import logging
import os

from langchain_core.language_models import BaseChatModel

from core.config import MODEL_OPTIONS, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)


def _check_api_key(env_var: str) -> bool:
    """Check if an API key environment variable is set and non-empty."""
    val = os.getenv(env_var, "").strip()
    return len(val) > 0 and not val.startswith("your_")


def get_available_models() -> dict[str, bool]:
    """Return a dict of model display names mapped to availability status."""
    availability: dict[str, bool] = {}
    key_map = {
        "Gemini 2.5 Flash": "GOOGLE_API_KEY",
        "Llama 3 70B": "GROQ_API_KEY",
    }
    for display_name in MODEL_OPTIONS:
        env_var = key_map.get(display_name, "")
        availability[display_name] = _check_api_key(env_var)
    return availability


def create_llm(model_display_name: str) -> BaseChatModel:
    """Create and return an LLM instance for the given display name."""
    model_id = MODEL_OPTIONS.get(model_display_name)
    if model_id is None:
        raise ValueError(f"Unknown model: {model_display_name}")

    try:
        if model_display_name == "Gemini 2.5 Flash":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_id,
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
            )
        elif model_display_name == "Llama 3 70B":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model_id,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        else:
            raise ValueError(f"No factory handler for model: {model_display_name}")
    except Exception as e:
        logger.error("Failed to create LLM '%s': %s", model_display_name, e)
        raise RuntimeError(f"Could not initialize {model_display_name}: {e}") from e
