"""Query classification router agent."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import QUERY_TYPES

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a query classification agent. Your ONLY job is to classify the user's query into exactly one of these categories:

- simple_retrieval: A straightforward factual question that can be answered from a single document passage.
- comparison: The user wants to compare information across multiple documents or sections.
- contradiction_check: The user wants to check if documents conflict or contain contradictory information.
- summarization: The user wants a summary of a document, section, or topic across documents.
- confidence_check: The user is asking about how confident or certain the system is about a previous answer.

Respond with ONLY the category name, nothing else. No explanation, no punctuation, just the category."""


def classify_query(llm: BaseChatModel, query: str, chat_history: str = "") -> str:
    """Classify a user query into one of the predefined query types."""
    try:
        context = ""
        if chat_history:
            context = f"\n\nRecent conversation context:\n{chat_history}\n"

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Classify this query:{context}\n\nQuery: {query}"),
        ]

        response = llm.invoke(messages)
        classification = response.content.strip().lower().replace('"', "").replace("'", "")

        # Validate classification
        if classification not in QUERY_TYPES:
            logger.warning("Router returned unknown type '%s', defaulting to simple_retrieval", classification)
            return "simple_retrieval"

        logger.info("Query classified as: %s", classification)
        return classification

    except Exception as e:
        logger.error("Router classification failed: %s", e)
        return "simple_retrieval"
