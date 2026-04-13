"""LangGraph graph definition connecting all agents into the Lumen pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from agents.router import classify_query
from agents.retrieval import retrieve_chunks, format_context, extract_sources, compute_confidence
from agents.reasoning import generate_response
from agents.formatter import build_response, FormattedResponse
from agents.evaluator import evaluate_response, compute_overall_score
from core.vector_store import VectorStore
from core.memory import MemoryManager

logger = logging.getLogger(__name__)


# ── State Schema ───────────────────────────────────────────
# LangGraph requires TypedDict for state, not dataclass.
# All fields must be defined here; LangGraph merges returned
# dicts from each node into the running state automatically.

class GraphState(TypedDict, total=False):
    """State object passed through every node in the LangGraph pipeline."""

    # ── Inputs (set before graph execution) ──
    query: str
    llm: Any
    vector_store: Any
    memory: Any
    domain_prompt: str
    model_name: str
    start_time: float

    # ── Populated by pipeline nodes ──
    query_type: str
    retrieval_results: list
    context: str
    sources: list
    confidence: int
    answer: str
    eval_scores: dict
    overall_score: float
    token_count: int
    error: Optional[str]


# ── Node Functions ─────────────────────────────────────────
# Each node receives the full state dict and returns a dict
# of ONLY the keys it modifies. LangGraph merges the update
# back into state automatically — do NOT return the full state.

def _node_route(state: GraphState) -> dict:
    """Router node: classify the query type from the user query."""
    try:
        memory = state.get("memory")
        chat_history = memory.get_history_string() if memory else ""
        query_type = classify_query(state["llm"], state["query"], chat_history)
        logger.info("Router classified query as: %s", query_type)
        return {"query_type": query_type}
    except Exception as e:
        logger.error("Router node failed: %s", e)
        return {"query_type": "simple_retrieval", "error": str(e)}


def _node_retrieve(state: GraphState) -> dict:
    """Retrieval node: fetch relevant chunks from the FAISS vector store."""
    try:
        retrieval_results = retrieve_chunks(
            state["vector_store"],
            state["query"],
            state.get("query_type", "simple_retrieval"),
        )
        return {
            "retrieval_results": retrieval_results,
            "context": format_context(retrieval_results),
            "sources": extract_sources(retrieval_results),
            "confidence": compute_confidence(retrieval_results),
        }
    except Exception as e:
        logger.error("Retrieval node failed: %s", e)
        return {
            "retrieval_results": [],
            "context": "No relevant documents found due to a retrieval error.",
            "sources": [],
            "confidence": 0,
            "error": str(e),
        }


def _node_reason(state: GraphState) -> dict:
    """Reasoning node: generate a grounded answer using the selected LLM."""
    try:
        memory = state.get("memory")
        chat_history = memory.get_history_string() if memory else ""
        answer = generate_response(
            llm=state["llm"],
            query=state["query"],
            query_type=state.get("query_type", "simple_retrieval"),
            context=state.get("context", ""),
            chat_history=chat_history,
            domain_prompt=state.get("domain_prompt", ""),
        )
        token_count = (len(state.get("context", "")) + len(answer)) // 4
        return {"answer": answer, "token_count": token_count}
    except Exception as e:
        logger.error("Reasoning node failed: %s", e)
        return {
            "answer": f"I encountered an error while generating a response: {e}",
            "token_count": 0,
            "error": str(e),
        }


def _node_evaluate(state: GraphState) -> dict:
    """Evaluation node: score the response quality using LLM-as-judge."""
    try:
        eval_scores = evaluate_response(
            llm=state["llm"],
            query=state["query"],
            response=state.get("answer", ""),
            context=state.get("context", ""),
        )
        overall_score = compute_overall_score(eval_scores)
        return {"eval_scores": eval_scores, "overall_score": overall_score}
    except Exception as e:
        logger.error("Evaluation node failed: %s", e)
        return {"eval_scores": {}, "overall_score": 0.0, "error": str(e)}


# ── Graph Construction ─────────────────────────────────────

def _build_graph() -> Any:
    """Construct and compile the Lumen LangGraph StateGraph."""
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("router", _node_route)
    workflow.add_node("retrieval", _node_retrieve)
    workflow.add_node("reasoning", _node_reason)
    workflow.add_node("evaluator", _node_evaluate)

    # Define edges — sequential pipeline
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retrieval")
    workflow.add_edge("retrieval", "reasoning")
    workflow.add_edge("reasoning", "evaluator")
    workflow.add_edge("evaluator", END)

    return workflow.compile()


# Compile once at module import — reused across all requests
_graph = _build_graph()


# ── Public API ─────────────────────────────────────────────

def run_pipeline(
    query: str,
    llm: BaseChatModel,
    vector_store: VectorStore,
    memory: MemoryManager,
    model_name: str = "Unknown",
    domain_prompt: str = "",
) -> FormattedResponse:
    """Execute the full Lumen pipeline via LangGraph: route → retrieve → reason → evaluate → format."""
    start_time = time.time()

    # Build initial state
    initial_state: GraphState = {
        "query": query,
        "llm": llm,
        "vector_store": vector_store,
        "memory": memory,
        "domain_prompt": domain_prompt,
        "model_name": model_name,
        "start_time": start_time,
        # Defaults for output fields
        "query_type": "",
        "retrieval_results": [],
        "context": "",
        "sources": [],
        "confidence": 0,
        "answer": "",
        "eval_scores": {},
        "overall_score": 0.0,
        "token_count": 0,
        "error": None,
    }

    # Execute the compiled graph
    try:
        final_state: GraphState = _graph.invoke(initial_state)
    except Exception as e:
        logger.error("LangGraph pipeline execution failed: %s", e)
        final_state = {**initial_state, "answer": f"Pipeline error: {e}", "error": str(e)}

    elapsed_ms = (time.time() - start_time) * 1000

    # Build structured response from final state
    response = build_response(
        answer=final_state.get("answer", "No response generated."),
        sources=final_state.get("sources", []),
        confidence=final_state.get("confidence", 0),
        query_type=final_state.get("query_type", "simple_retrieval"),
        model_used=model_name,
        latency_ms=round(elapsed_ms, 1),
        eval_scores=final_state.get("eval_scores", {}),
        token_count=final_state.get("token_count", 0),
    )

    # Update conversation memory after pipeline completes
    if memory:
        memory.add_user_message(query)
        memory.add_assistant_message(
            final_state.get("answer", ""),
            metadata={
                "query_type": final_state.get("query_type", ""),
                "confidence": final_state.get("confidence", 0),
                "model": model_name,
            },
        )

    return response
