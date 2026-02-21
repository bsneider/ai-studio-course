#!/usr/bin/env python3
"""Generate publication-quality benchmark charts from results.json.

Usage:
    uv run python scripts/generate_charts.py [--results path] [--output-dir path]

Reads benchmark_results/results.json (produced by pytest conftest.py plugin)
and generates 4 PNG charts with MIT AI Studio branding.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Brand constants ──────────────────────────────────────────────────────────

MIT_CRIMSON = "#A31F34"
MIT_GRAY = "#8A8B8C"
WARM_GOLD = "#C4A265"
DARK_NAVY = "#1B365D"

APPROACH_COLORS = {
    "Hybrid": MIT_CRIMSON,
    "BM25": MIT_GRAY,
    "Semantic": WARM_GOLD,
    "LLM-Rerank": DARK_NAVY,
}

FONT_FAMILY = "Helvetica Neue"
FALLBACK_FONT = "DejaVu Sans"
DPI = 300
FOOTER_TEXT = "Research: Brandon Sneider | MIT AI Studio (MAS.664/665)"


def _setup_font():
    """Configure matplotlib font with fallback."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    if FONT_FAMILY in available:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [FONT_FAMILY, FALLBACK_FONT]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [FALLBACK_FONT]
    plt.rcParams["font.size"] = 10


def add_branding(fig):
    """Add MIT Crimson header line and researcher attribution footer."""
    # Thin crimson line at top
    fig.patches.append(mpatches.FancyBboxPatch(
        (0, 0.985), 1.0, 0.015,
        boxstyle="square,pad=0",
        facecolor=MIT_CRIMSON,
        edgecolor="none",
        transform=fig.transFigure,
        zorder=100,
    ))
    # Footer attribution
    fig.text(
        0.98, 0.008, FOOTER_TEXT,
        ha="right", va="bottom",
        fontsize=8, color=MIT_GRAY,
        transform=fig.transFigure,
    )


# ── Chart 1: Grouped Bar — Recall@K by Difficulty Level ─────────────────────


def chart_recall_by_level(data, output_dir):
    """Grouped bar chart of mean Recall@K by difficulty level."""
    _setup_font()

    per_level = data["per_level"]
    approaches = data["approaches"]
    levels = sorted(per_level.keys())
    n_approaches = len(approaches)
    n_levels = len(levels)

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.8 / n_approaches
    x = np.arange(n_levels)

    for i, aname in enumerate(approaches):
        color = APPROACH_COLORS.get(aname, "#333333")
        vals = []
        for level in levels:
            vals.append(per_level[level]["approaches"].get(aname, {}).get("mean_recall", 0))
        offset = (i - n_approaches / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width * 0.9, label=aname, color=color, zorder=3)
        # Value labels above bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.0%}", ha="center", va="bottom", fontsize=7, fontweight="bold",
            )

    # Pass threshold line
    ax.axhline(y=0.4, color=MIT_CRIMSON, linestyle="--", linewidth=1, alpha=0.7, zorder=2)
    ax.text(n_levels - 0.3, 0.42, "Pass threshold (40%)", fontsize=8, color=MIT_CRIMSON, alpha=0.8)

    # L8 annotation if present
    if "L8" in levels:
        l8_idx = levels.index("L8")
        ax.text(
            l8_idx, -0.09, "(inverse-scored)",
            fontsize=7, ha="center", color=MIT_CRIMSON, fontstyle="italic",
            transform=ax.get_xaxis_transform(), clip_on=False,
        )

    ax.set_xlabel("Difficulty Level", fontsize=11)
    ax.set_ylabel("Mean Recall@K", fontsize=11)
    ax.set_title("Retrieval Recall by Difficulty Level", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart1_recall_by_level.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 1 saved: {path}")
    return path


# ── Chart 2: Radar — Metric Profiles ────────────────────────────────────────


def chart_radar_metrics(data, output_dir):
    """Radar chart comparing metric profiles across approaches."""
    _setup_font()

    approaches = data["approaches"]
    aggregates = data["aggregates"]
    metric_names = ["Recall@K", "MRR", "NDCG@K", "pass@1", "pass@3", "pass^3"]
    n_metrics = len(metric_names)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for aname in approaches:
        color = APPROACH_COLORS.get(aname, "#333333")
        agg = aggregates.get(aname, {})
        values = [agg.get(m, 0) for m in metric_names]
        values += values[:1]  # close
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=aname, markersize=5)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color=MIT_GRAY)
    ax.set_title("Retrieval Metric Profiles", fontsize=14, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), framealpha=0.9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart2_radar_metrics.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 2 saved: {path}")
    return path


# ── Chart 3: Heatmap — Per-Query Scores ──────────────────────────────────────


def chart_heatmap(data, output_dir):
    """Heatmap of per-query recall scores across all approaches."""
    _setup_font()

    per_query = data["per_query"]
    approaches = data["approaches"]
    n_queries = len(per_query)
    n_approaches = len(approaches)

    # Build matrix and labels
    matrix = np.zeros((n_queries, n_approaches))
    query_labels = []
    for i, q in enumerate(per_query):
        label = q["query"][:45]
        if q["scoring"] == "inverse":
            label += " [INV]"
        query_labels.append(f"[{q['level']}] {label}")
        for j, aname in enumerate(approaches):
            matrix[i, j] = q["approaches"].get(aname, {}).get("effective_recall", 0)

    # Identify level boundaries for separator lines
    level_breaks = []
    for i in range(1, n_queries):
        if per_query[i]["level"] != per_query[i - 1]["level"]:
            level_breaks.append(i)

    fig_height = max(10, n_queries * 0.45)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Custom colormap: Red -> Yellow -> Green
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rag", ["#d32f2f", "#fdd835", "#388e3c"])

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Text annotations in each cell
    for i in range(n_queries):
        for j in range(n_approaches):
            val = matrix[i, j]
            text_color = "white" if val < 0.3 or val > 0.8 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=text_color)

    # Level separator lines
    for brk in level_breaks:
        ax.axhline(y=brk - 0.5, color="white", linewidth=2)

    ax.set_xticks(range(n_approaches))
    ax.set_xticklabels(approaches, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_queries))
    ax.set_yticklabels(query_labels, fontsize=7)
    ax.set_title("Per-Query Effective Recall (All Approaches)", fontsize=14, fontweight="bold", pad=15)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Effective Recall", fontsize=10)
    # Mark threshold on colorbar
    cbar.ax.axhline(y=0.4, color=MIT_CRIMSON, linewidth=2, linestyle="--")

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart3_heatmap.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 3 saved: {path}")
    return path


# ── Chart 4: Summary Metrics Table ───────────────────────────────────────────


def chart_summary_table(data, output_dir):
    """Styled matplotlib table figure showing aggregate metrics."""
    _setup_font()

    approaches = data["approaches"]
    aggregates = data["aggregates"]
    metric_names = ["Recall@K", "MRR", "NDCG@K", "pass@1", "pass@3", "pass^3", "pass_rate"]
    metric_display = ["Recall@K", "MRR", "NDCG@K", "pass@1", "pass@3", "pass^3", "Pass Rate"]

    # Build table data
    col_labels = ["Metric"] + approaches
    cell_text = []
    cell_colors = []

    for idx, (mname, mdisp) in enumerate(zip(metric_names, metric_display)):
        row_vals = []
        for aname in approaches:
            row_vals.append(aggregates.get(aname, {}).get(mname, 0))
        best_val = max(row_vals)
        row = [mdisp]
        for val in row_vals:
            row.append(f"{val:.4f}")
        cell_text.append(row)

        # Alternating row shading
        base_color = "#f5f5f5" if idx % 2 == 0 else "white"
        row_colors = [base_color] * (len(approaches) + 1)
        cell_colors.append(row_colors)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.set_title("Aggregate Benchmark Metrics Summary", fontsize=14, fontweight="bold", pad=20)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style header row
    for j, label in enumerate(col_labels):
        cell = table[0, j]
        cell.set_facecolor(MIT_CRIMSON)
        cell.set_text_props(color="white", fontweight="bold")

    # Bold best values in crimson
    for i, (mname, _) in enumerate(zip(metric_names, metric_display)):
        row_vals = []
        for aname in approaches:
            row_vals.append(aggregates.get(aname, {}).get(mname, 0))
        best_val = max(row_vals)
        for j, val in enumerate(row_vals):
            if val == best_val and best_val > 0:
                cell = table[i + 1, j + 1]  # +1 for header row, +1 for metric column
                cell.set_text_props(color=MIT_CRIMSON, fontweight="bold")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_branding(fig)

    path = output_dir / "chart4_summary_table.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 4 saved: {path}")
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark charts from results.json")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmark_results/results.json"),
        help="Path to results.json (default: benchmark_results/results.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for PNG charts (default: benchmark_results)",
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found.")
        print("Run 'uv run pytest tests/test_pageindex.py -v -s' first to generate benchmark data.")
        sys.exit(1)

    data = json.loads(args.results.read_text())
    print(f"Loaded {data['n_queries']} queries x {len(data['approaches'])} approaches")
    print(f"Generated at: {data['generated_at']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating charts...")
    chart_recall_by_level(data, args.output_dir)
    chart_radar_metrics(data, args.output_dir)
    chart_heatmap(data, args.output_dir)
    chart_summary_table(data, args.output_dir)
    print("\nDone! All charts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
