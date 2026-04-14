"""
Generate visual benchmark report from Lumen benchmark results.

Reads JSON results from benchmarks/results/ and produces:
- Evaluation scores bar chart (by dimension)
- Latency distribution histogram
- Scores by query type heatmap
- Model comparison (if multiple models)
- Summary statistics table

Usage:
    python benchmarks/generate_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "report"


def load_results() -> dict[str, dict]:
    """Load all benchmark result files."""
    results = {}
    for f in RESULTS_DIR.glob("benchmark_*.json"):
        if f.name == "benchmark_comparison.json":
            continue
        data = json.loads(f.read_text())
        model = data["summary"]["model"]
        results[model] = data
    return results


def plot_eval_scores(all_results: dict[str, dict], output_dir: Path) -> None:
    """Bar chart comparing eval dimensions across models."""
    dimensions = ["Faithfulness", "Completeness", "Clarity", "Hallucination Risk"]
    dim_keys = ["avg_faithfulness", "avg_completeness", "avg_clarity", "avg_hallucination_risk"]
    models = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(dimensions))
    width = 0.35 if len(models) > 1 else 0.5
    colors = ["#0d9488", "#06b6d4", "#8b5cf6", "#f59e0b"]

    for i, model in enumerate(models):
        summary = all_results[model]["summary"]
        scores = [summary[k] for k in dim_keys]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width, label=model, color=colors[i % len(colors)], alpha=0.85, edgecolor="white", linewidth=0.5)

        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{score:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (1-5)", fontweight="bold")
    ax.set_title("Evaluation Scores by Dimension", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions)
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "eval_scores.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ eval_scores.png")


def plot_latency_distribution(all_results: dict[str, dict], output_dir: Path) -> None:
    """Histogram of latency distribution per model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#0d9488", "#06b6d4"]

    for i, (model, data) in enumerate(all_results.items()):
        latencies = [r["latency_ms"] for r in data["results"] if r["error"] is None]
        ax.hist(latencies, bins=15, alpha=0.6, label=model, color=colors[i % len(colors)], edgecolor="white")

        # Add p50 and p95 lines
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        ax.axvline(p50, color=colors[i % len(colors)], linestyle="--", alpha=0.8, linewidth=1.5)
        ax.axvline(p95, color=colors[i % len(colors)], linestyle=":", alpha=0.6, linewidth=1.5)

    ax.set_xlabel("Latency (ms)", fontweight="bold")
    ax.set_ylabel("Query Count", fontweight="bold")
    ax.set_title("Latency Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "latency_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ latency_distribution.png")


def plot_scores_by_type(all_results: dict[str, dict], output_dir: Path) -> None:
    """Grouped bar chart of faithfulness scores by query type."""
    query_types = ["simple_retrieval", "comparison", "contradiction_check", "summarization", "confidence_check"]
    type_labels = ["Retrieval", "Comparison", "Contradiction", "Summary", "Confidence"]
    models = list(all_results.keys())
    colors = ["#0d9488", "#06b6d4"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Faithfulness by type
    ax = axes[0]
    x = np.arange(len(query_types))
    width = 0.35 if len(models) > 1 else 0.5

    for i, model in enumerate(models):
        by_type = all_results[model]["summary"].get("by_query_type", {})
        scores = [by_type.get(qt, {}).get("avg_faithfulness", 0) for qt in query_types]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, scores, width, label=model, color=colors[i % len(colors)], alpha=0.85)

    ax.set_ylabel("Faithfulness (1-5)", fontweight="bold")
    ax.set_title("Faithfulness by Query Type", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, rotation=25, ha="right")
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Latency by type
    ax = axes[1]
    for i, model in enumerate(models):
        by_type = all_results[model]["summary"].get("by_query_type", {})
        latencies = [by_type.get(qt, {}).get("avg_latency_ms", 0) for qt in query_types]
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, latencies, width, label=model, color=colors[i % len(colors)], alpha=0.85)

    ax.set_ylabel("Latency (ms)", fontweight="bold")
    ax.set_title("Latency by Query Type", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, rotation=25, ha="right")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "scores_by_type.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ scores_by_type.png")


def plot_confidence_vs_faithfulness(all_results: dict[str, dict], output_dir: Path) -> None:
    """Scatter plot of confidence score vs faithfulness."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#0d9488", "#06b6d4"]

    for i, (model, data) in enumerate(all_results.items()):
        successful = [r for r in data["results"] if r["error"] is None]
        confidences = [r["confidence"] for r in successful]
        faithfulness = [r["faithfulness"] for r in successful]
        ax.scatter(confidences, faithfulness, alpha=0.5, s=40, label=model,
                   color=colors[i % len(colors)], edgecolors="white", linewidth=0.5)

    ax.set_xlabel("Confidence Score (0-100)", fontweight="bold")
    ax.set_ylabel("Faithfulness (1-5)", fontweight="bold")
    ax.set_title("Confidence vs Faithfulness", fontsize=14, fontweight="bold", pad=15)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "confidence_vs_faithfulness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ confidence_vs_faithfulness.png")


def generate_summary_table(all_results: dict[str, dict], output_dir: Path) -> None:
    """Generate a markdown summary table."""
    lines = ["# Lumen Benchmark Report\n"]
    lines.append(f"**Models tested:** {', '.join(all_results.keys())}\n")
    lines.append(f"**Queries per model:** 50\n\n")

    lines.append("## Summary\n")
    lines.append("| Metric | " + " | ".join(all_results.keys()) + " |")
    lines.append("|---|" + "|".join(["---"] * len(all_results)) + "|")

    metrics = [
        ("Queries", "successful", "total_queries"),
        ("Router Accuracy", "router_accuracy", None),
        ("Avg Faithfulness", "avg_faithfulness", None),
        ("Avg Completeness", "avg_completeness", None),
        ("Avg Clarity", "avg_clarity", None),
        ("Avg Hallucination Risk", "avg_hallucination_risk", None),
        ("Overall Score", "avg_overall", None),
        ("Avg Latency (ms)", "avg_latency_ms", None),
        ("P50 Latency (ms)", "p50_latency_ms", None),
        ("P95 Latency (ms)", "p95_latency_ms", None),
        ("Avg Confidence", "avg_confidence", None),
    ]

    for label, key, total_key in metrics:
        row = f"| {label} |"
        for model in all_results:
            s = all_results[model]["summary"]
            if total_key:
                row += f" {s[key]}/{s[total_key]} |"
            elif "latency" in key.lower():
                row += f" {s[key]:.0f} |"
            elif key == "router_accuracy":
                row += f" {s[key]}% |"
            else:
                row += f" {s[key]} |"
        lines.append(row)

    lines.append("\n## Charts\n")
    lines.append("![Evaluation Scores](eval_scores.png)\n")
    lines.append("![Latency Distribution](latency_distribution.png)\n")
    lines.append("![Scores by Query Type](scores_by_type.png)\n")
    lines.append("![Confidence vs Faithfulness](confidence_vs_faithfulness.png)\n")

    (output_dir / "REPORT.md").write_text("\n".join(lines))
    print("  ✅ REPORT.md")


def main() -> None:
    """Generate benchmark report from saved results."""
    print("\n📊 Generating Lumen Benchmark Report\n")

    all_results = load_results()
    if not all_results:
        print("❌ No benchmark results found in benchmarks/results/")
        print("   Run: python benchmarks/run_benchmark.py --model all")
        sys.exit(1)

    print(f"   Found results for: {', '.join(all_results.keys())}\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("📈 Generating charts...")
    plot_eval_scores(all_results, OUTPUT_DIR)
    plot_latency_distribution(all_results, OUTPUT_DIR)
    plot_scores_by_type(all_results, OUTPUT_DIR)
    plot_confidence_vs_faithfulness(all_results, OUTPUT_DIR)
    generate_summary_table(all_results, OUTPUT_DIR)

    print(f"\n✅ Report generated in {OUTPUT_DIR}/")
    print(f"   Open {OUTPUT_DIR}/REPORT.md for the full report\n")


if __name__ == "__main__":
    main()
