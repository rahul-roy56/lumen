"""Generate the Lumen architecture diagram using matplotlib (no emojis)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


def create_architecture_diagram(output_path: str = "architecture.png") -> None:
    """Generate and save the Lumen architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # Color scheme
    C_INGEST = "#4A90D9"
    C_AGENT = "#8B5CF6"
    C_STORAGE = "#10B981"
    C_TRACK = "#F59E0B"
    C_LLM = "#EF4444"
    C_USER = "#6B7280"
    C_TEXT = "white"

    def box(x, y, w, h, label, color, fontsize=9):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.92,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C_TEXT,
                family="sans-serif")

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
        )

    # ── Title ──
    ax.text(8, 9.6, "LUMEN  -  System Architecture", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#1F2937", family="sans-serif")

    # ── User ──
    box(0.5, 7.2, 2.5, 1.0, "USER\n(Streamlit UI)", C_USER, 10)

    # ── Ingestion Layer ──
    ax.text(5.75, 8.8, "INGESTION PIPELINE", ha="center", fontsize=10,
            fontweight="bold", color=C_INGEST, family="sans-serif")
    box(3.8, 7.2, 1.4, 1.0, "PDF\nExtractor", C_INGEST)
    box(5.3, 7.2, 1.4, 1.0, "DOCX\nExtractor", C_INGEST)
    box(6.8, 7.2, 1.4, 1.0, "TXT\nReader", C_INGEST)
    box(4.5, 5.8, 2.8, 0.9, "Chunker\n(500 tok / 50 overlap)", C_INGEST, 8)

    # ── Vector Store ──
    box(8.8, 5.8, 2.8, 1.4, "FAISS\nVector Store\n(all-MiniLM-L6-v2)", C_STORAGE, 9)

    # ── Agent Pipeline ──
    ax.text(6.5, 4.8, "MULTI-AGENT REASONING PIPELINE  (LangGraph)", ha="center",
            fontsize=10, fontweight="bold", color=C_AGENT, family="sans-serif")
    box(1.0, 3.2, 2.2, 1.0, "ROUTER\nAgent", C_AGENT, 9)
    box(3.8, 3.2, 2.2, 1.0, "RETRIEVAL\nAgent", C_AGENT, 9)
    box(6.6, 3.2, 2.4, 1.0, "REASONING\nAgent", C_AGENT, 9)
    box(9.6, 3.2, 2.2, 1.0, "FORMATTER\nAgent", C_AGENT, 9)

    # ── Evaluator ──
    box(6.6, 1.2, 2.4, 1.0, "EVALUATOR\n(LLM-as-Judge)", C_AGENT, 9)

    # ── LLM Factory ──
    box(12.5, 3.2, 2.8, 1.0, "LLM FACTORY\nGPT-4o | Claude | Gemini", C_LLM, 8)

    # ── MLflow ──
    box(10.0, 1.2, 2.8, 1.0, "MLflow\nExperiment Tracker", C_TRACK, 9)

    # ── Memory ──
    box(13.0, 1.2, 2.5, 1.0, "Conversation\nMemory", C_STORAGE, 9)

    # ── Arrows ──
    arrow(3.0, 7.7, 3.8, 7.7)       # User -> Ingestion
    arrow(4.5, 7.2, 5.2, 6.7)       # PDF -> Chunker
    arrow(6.0, 7.2, 5.9, 6.7)       # DOCX -> Chunker
    arrow(7.5, 7.2, 6.6, 6.7)       # TXT -> Chunker
    arrow(7.3, 6.25, 8.8, 6.4)      # Chunker -> FAISS
    arrow(1.75, 7.2, 1.75, 4.2)     # User -> Router (query)
    arrow(3.2, 3.7, 3.8, 3.7)       # Router -> Retrieval
    arrow(9.5, 5.8, 5.5, 4.2)       # FAISS -> Retrieval
    arrow(6.0, 3.7, 6.6, 3.7)       # Retrieval -> Reasoning
    arrow(9.0, 3.7, 9.6, 3.7)       # Reasoning -> Formatter
    arrow(10.7, 4.2, 2.5, 7.2)      # Formatter -> User (response)
    arrow(12.5, 3.7, 9.0, 3.7)      # LLM Factory -> Reasoning
    arrow(13.2, 3.2, 8.5, 2.2)      # LLM Factory -> Evaluator
    arrow(10.2, 3.2, 9.0, 2.2)      # Formatter -> Evaluator
    arrow(9.0, 1.7, 10.0, 1.7)      # Evaluator -> MLflow
    arrow(13.0, 1.7, 2.5, 3.2)      # Memory -> Router

    # ── Legend ──
    legend_items = [
        mpatches.Patch(color=C_INGEST, label="Ingestion"),
        mpatches.Patch(color=C_AGENT, label="Agents"),
        mpatches.Patch(color=C_STORAGE, label="Storage"),
        mpatches.Patch(color=C_TRACK, label="Tracking"),
        mpatches.Patch(color=C_LLM, label="LLM Layer"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9,
              framealpha=0.9, ncol=5, bbox_to_anchor=(0.05, -0.02))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"Architecture diagram saved to {output_path}")


if __name__ == "__main__":
    create_architecture_diagram()
