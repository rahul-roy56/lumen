"""MLflow experiment tracking and logging for all Lumen queries."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from agents.formatter import FormattedResponse
from core.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

_initialized: bool = False


def init_mlflow() -> None:
    """Initialize MLflow tracking with the Lumen experiment."""
    global _initialized
    if _initialized:
        return
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        _initialized = True
        logger.info("MLflow initialized — experiment: %s", MLFLOW_EXPERIMENT_NAME)
    except Exception as e:
        logger.warning("MLflow initialization failed (non-fatal): %s", e)


def log_query(query: str, response: FormattedResponse, prompt_text: str = "") -> None:
    """Log a complete query-response cycle as an MLflow run."""
    init_mlflow()
    try:
        with mlflow.start_run(nested=True):
            # Parameters
            mlflow.log_param("query", query[:250])
            mlflow.log_param("query_type", response.query_type)
            mlflow.log_param("model_used", response.model_used)
            mlflow.log_param("chunk_count", len(response.sources))

            # Metrics
            mlflow.log_metric("latency_ms", response.latency_ms)
            mlflow.log_metric("confidence", response.confidence)
            mlflow.log_metric("token_count", response.token_count)

            for dim, score in response.eval_scores.items():
                safe_name = dim.lower().replace(" ", "_")
                mlflow.log_metric(f"eval_{safe_name}", score)

            if response.eval_scores:
                avg = sum(response.eval_scores.values()) / len(response.eval_scores)
                mlflow.log_metric("eval_overall", round(avg, 2))

            # Artifacts — log full prompt and response as text files
            if prompt_text:
                mlflow.log_text(prompt_text, "prompt.txt")
            mlflow.log_text(response.answer, "response.txt")

        logger.info("Logged MLflow run for query: %s...", query[:50])
    except Exception as e:
        logger.warning("MLflow logging failed (non-fatal): %s", e)


def get_experiment_summary() -> dict[str, Any]:
    """Retrieve summary statistics from all MLflow runs in the experiment."""
    init_mlflow()
    summary: dict[str, Any] = {
        "total_queries": 0,
        "avg_faithfulness": 0.0,
        "avg_latency_ms": 0.0,
        "best_model": "N/A",
    }

    try:
        client = MlflowClient(MLFLOW_TRACKING_URI)
        experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return summary

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=500,
        )

        if not runs:
            return summary

        summary["total_queries"] = len(runs)

        # Average faithfulness
        faith_scores = [
            r.data.metrics.get("eval_faithfulness", 0)
            for r in runs
            if "eval_faithfulness" in r.data.metrics
        ]
        if faith_scores:
            summary["avg_faithfulness"] = round(sum(faith_scores) / len(faith_scores), 2)

        # Average latency
        latencies = [
            r.data.metrics.get("latency_ms", 0)
            for r in runs
            if "latency_ms" in r.data.metrics
        ]
        if latencies:
            summary["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)

        # Best model by average overall eval
        model_scores: dict[str, list[float]] = {}
        for r in runs:
            model = r.data.params.get("model_used", "Unknown")
            overall = r.data.metrics.get("eval_overall", 0)
            if model not in model_scores:
                model_scores[model] = []
            model_scores[model].append(overall)

        if model_scores:
            best = max(model_scores, key=lambda m: sum(model_scores[m]) / len(model_scores[m]))
            summary["best_model"] = best

    except Exception as e:
        logger.warning("Failed to retrieve MLflow summary: %s", e)

    return summary
