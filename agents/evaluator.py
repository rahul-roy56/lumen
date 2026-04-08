"""LLM-as-judge evaluation agent for scoring response quality."""

from __future__ import annotations

import json
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import EVAL_DIMENSIONS

logger = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = """You are an expert evaluator for a document-based RAG (Retrieval-Augmented Generation) system called Lumen. Your job is to fairly evaluate a response on exactly 4 dimensions, each scored from 1 to 5.

IMPORTANT CONTEXT: This system retrieves relevant chunks from uploaded documents and generates answers based ONLY on those chunks. The system cannot see the full documents — only the retrieved excerpts shown in "DOCUMENT CONTEXT PROVIDED". Judge the response ONLY against what was available in the provided context, NOT against what an ideal answer would look like with full document access.

Scoring dimensions:
1. **Faithfulness** (1-5): Is every claim in the answer directly traceable to the provided context? 
   - 5 = Every statement maps to specific text in the context
   - 4 = Almost all claims are grounded, minor paraphrasing that preserves meaning
   - 3 = Most claims grounded but some reasonable inferences made
   - 2 = Several claims go beyond what context supports
   - 1 = Mostly fabricated

2. **Completeness** (1-5): Given the AVAILABLE context (not the full documents), does the answer address the question as well as possible?
   - 5 = Extracts all relevant information from the provided context
   - 4 = Covers most relevant information from context
   - 3 = Addresses the core question but misses some available details
   - 2 = Only partially addresses the question despite relevant context being available
   - 1 = Fails to use available context

3. **Clarity** (1-5): Is the response well-structured, easy to read, and properly formatted?
   - 5 = Excellent organization with clear citations
   - 3 = Readable but could be better organized
   - 1 = Confusing or poorly structured

4. **Hallucination Risk** (1-5): Does the answer stay within the bounds of the provided context?
   - 5 = Zero claims beyond the context, includes appropriate hedging
   - 4 = All substantive claims grounded, only minor connecting language added
   - 3 = Mostly grounded but adds some reasonable context
   - 2 = Adds notable information not in the context
   - 1 = Significant fabrication

You MUST respond with ONLY a valid JSON object in this exact format, with no other text:
{"Faithfulness": <int>, "Completeness": <int>, "Clarity": <int>, "Hallucination Risk": <int>}"""


def evaluate_response(
    llm: BaseChatModel,
    query: str,
    response: str,
    context: str,
) -> dict[str, int]:
    """Evaluate a response on 4 quality dimensions using LLM-as-judge."""
    try:
        # Give the evaluator more context — up to 5000 chars
        eval_context = context[:5000]

        eval_prompt = f"""Evaluate this RAG system response. Remember: judge the response ONLY against the provided context, not against what an ideal answer would look like.

USER QUERY: {query}

DOCUMENT CONTEXT PROVIDED TO THE SYSTEM:
{eval_context}

SYSTEM RESPONSE TO EVALUATE:
{response}

Score each dimension 1-5 based on the rubric. Be fair — if the response accurately uses the available context and cites sources, that deserves high scores even if the context itself was limited. Return ONLY the JSON object."""

        messages = [
            SystemMessage(content=EVAL_SYSTEM_PROMPT),
            HumanMessage(content=eval_prompt),
        ]

        eval_response = llm.invoke(messages)
        raw_text = eval_response.content.strip()

        # Clean up potential markdown formatting
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        scores = json.loads(raw_text)

        # Validate scores
        validated: dict[str, int] = {}
        for dim in EVAL_DIMENSIONS:
            val = scores.get(dim, 3)
            validated[dim] = max(1, min(5, int(val)))

        logger.info("Evaluation scores: %s", validated)
        return validated

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse eval JSON: %s — raw: %s", e, raw_text[:200] if 'raw_text' in dir() else "N/A")
        return {dim: 3 for dim in EVAL_DIMENSIONS}
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        return {dim: 3 for dim in EVAL_DIMENSIONS}


def compute_overall_score(scores: dict[str, int]) -> float:
    """Compute the average evaluation score across all dimensions."""
    if not scores:
        return 0.0
    return round(sum(scores.values()) / len(scores), 2)
