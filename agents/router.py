"""Query classification router agent."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import QUERY_TYPES

logger = logging.getLogger(__name__)

ROUTER_SYSTEM_PROMPT = """You are a query classification agent. Classify the user's query into EXACTLY ONE category.

DECISION RULES — check in this order:

1. **confidence_check** — The user asks about YOUR confidence, reliability, certainty, or limitations. Keywords: "how confident", "how reliable", "how sure", "how well", "limitations", "can you trust", "how accurate".
   Examples:
   - "How confident are you in that answer?" → confidence_check
   - "How reliable is your analysis?" → confidence_check
   - "What are the limitations of your analysis?" → confidence_check

2. **contradiction_check** — The user asks if documents CONFLICT, CONTRADICT, or are INCONSISTENT. Keywords: "contradict", "conflict", "inconsistent", "disagree", "contradictions", "conflicting".
   Examples:
   - "Do the two reports contradict each other?" → contradiction_check
   - "Are there conflicting statements about revenue?" → contradiction_check
   - "Any inconsistencies between the documents?" → contradiction_check

3. **comparison** — The user wants to COMPARE or CONTRAST information across documents. Keywords: "compare", "comparison", "versus", "vs", "difference between", "how do X and Y differ", "which company".
   Examples:
   - "Compare the revenue of both companies" → comparison
   - "How do the operating margins differ?" → comparison
   - "Which company has more employees?" → comparison

4. **summarization** — The user wants a SUMMARY, OVERVIEW, KEY POINTS, or HIGHLIGHTS. Keywords: "summarize", "summary", "overview", "key points", "main findings", "give me an overview", "tell me about", "what are the main".
   Examples:
   - "Summarize the annual report" → summarization
   - "What are the key points from both documents?" → summarization
   - "Give me an overview of all product launches" → summarization
   - "Provide an executive summary" → summarization

5. **simple_retrieval** — A direct factual question that asks for a SPECIFIC piece of information. This is the DEFAULT only if none of the above match.
   Examples:
   - "What was the total revenue in 2024?" → simple_retrieval
   - "How many employees does the company have?" → simple_retrieval
   - "Where is TechNova headquartered?" → simple_retrieval

IMPORTANT: Do NOT default to simple_retrieval. Check rules 1-4 first. Only use simple_retrieval for specific factual questions.

Respond with ONLY the category name. No explanation, no punctuation, no quotes."""


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
