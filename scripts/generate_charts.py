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


# ── Chart 5: Weight Sweep — BM25/Semantic Mix ────────────────────────────────


def chart_weight_sweep(data, output_dir):
    """Line chart showing how recall changes as BM25/Semantic ratio varies."""
    _setup_font()

    sweep = data.get("weight_sweep", [])
    if not sweep:
        print("  Chart 5 skipped: no weight_sweep data in results.json")
        return None

    # Separate hybrid sweep points from LLM-Rerank
    hybrid_pts = [p for p in sweep if p["kw"] is not None]
    llm_pt = next((p for p in sweep if p["label"] == "LLM-Rerank"), None)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # X-axis: semantic weight (0.0 to 1.0)
    x = [p["sw"] for p in hybrid_pts]
    recalls = [p["mean_recall"] for p in hybrid_pts]
    pass_rates = [p["pass_rate"] for p in hybrid_pts]

    # Primary axis: Mean Recall
    line1 = ax1.plot(x, recalls, "o-", color=MIT_CRIMSON, linewidth=2.5,
                     markersize=8, label="Mean Recall@K", zorder=4)
    ax1.set_xlabel("Semantic Weight (BM25 weight = 1 - semantic)", fontsize=11)
    ax1.set_ylabel("Mean Effective Recall@K", fontsize=11, color=MIT_CRIMSON)
    ax1.tick_params(axis="y", labelcolor=MIT_CRIMSON)

    # Secondary axis: Pass Rate
    ax2 = ax1.twinx()
    line2 = ax2.plot(x, pass_rates, "s--", color=DARK_NAVY, linewidth=2,
                     markersize=7, label="Pass Rate (>=40%)", zorder=3)
    ax2.set_ylabel("Pass Rate", fontsize=11, color=DARK_NAVY)
    ax2.tick_params(axis="y", labelcolor=DARK_NAVY)
    ax2.set_ylim(0, 1.05)

    # Mark the best hybrid point
    best_idx = max(range(len(recalls)), key=lambda i: recalls[i])
    ax1.annotate(
        f"Best: sw={x[best_idx]:.1f}\n({recalls[best_idx]:.0%})",
        xy=(x[best_idx], recalls[best_idx]),
        xytext=(x[best_idx] - 0.15, recalls[best_idx] + 0.06),
        fontsize=9, fontweight="bold", color=MIT_CRIMSON,
        arrowprops=dict(arrowstyle="->", color=MIT_CRIMSON, lw=1.5),
    )

    # LLM-Rerank as a horizontal reference line
    if llm_pt:
        ax1.axhline(y=llm_pt["mean_recall"], color=WARM_GOLD, linestyle=":",
                     linewidth=2, alpha=0.8, zorder=2)
        ax1.text(0.02, llm_pt["mean_recall"] + 0.01,
                 f"LLM-Rerank ({llm_pt['mean_recall']:.0%})",
                 fontsize=9, color=WARM_GOLD, fontweight="bold")

    # Labels for endpoints
    ax1.text(0.0, recalls[0] - 0.03, "Pure\nBM25", fontsize=8, ha="center",
             color=MIT_GRAY, fontstyle="italic")
    ax1.text(1.0, recalls[-1] - 0.03, "Pure\nSemantic", fontsize=8, ha="center",
             color=MIT_GRAY, fontstyle="italic")

    # Pass threshold
    ax1.axhline(y=0.4, color=MIT_GRAY, linestyle="--", linewidth=0.8, alpha=0.5)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower center", framealpha=0.9)

    ax1.set_title("Effect of BM25/Semantic Weight Mix on Retrieval Quality",
                   fontsize=14, fontweight="bold", pad=20)
    ax1.set_xlim(-0.05, 1.05)
    ax1.grid(axis="both", alpha=0.2, zorder=0)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart5_weight_sweep.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 5 saved: {path}")
    return path


# ── Chart 6: Approach Progression — Waterfall ────────────────────────────────


def chart_approach_progression(data, output_dir):
    """Waterfall chart showing incremental improvement from BM25 -> Hybrid -> LLM-Rerank."""
    _setup_font()

    aggregates = data.get("aggregates", {})
    sweep = data.get("weight_sweep", [])

    # Build progression stages — ordered to tell an "improvement" story:
    # worst baseline -> better baseline -> hybrid combines both -> LLM tops it
    stages = []

    # Determine which single-method baseline is weaker to put it first
    bm25_recall = aggregates.get("BM25", {}).get("Recall@K", 0)
    sem_recall = aggregates.get("Semantic", {}).get("Recall@K", 0)

    if sem_recall <= bm25_recall:
        # Semantic is the weaker baseline
        if "Semantic" in aggregates:
            stages.append({
                "label": "Semantic\n(embeddings only)",
                "recall": sem_recall,
                "color": WARM_GOLD,
            })
        if "BM25" in aggregates:
            stages.append({
                "label": "BM25\n(keywords only)",
                "recall": bm25_recall,
                "color": MIT_GRAY,
            })
    else:
        # BM25 is the weaker baseline
        if "BM25" in aggregates:
            stages.append({
                "label": "BM25\n(keywords only)",
                "recall": bm25_recall,
                "color": MIT_GRAY,
            })
        if "Semantic" in aggregates:
            stages.append({
                "label": "Semantic\n(embeddings only)",
                "recall": sem_recall,
                "color": WARM_GOLD,
            })

    # Hybrid combines both — use best mix from sweep if available
    hybrid_pts = [p for p in sweep if p["kw"] is not None]
    if hybrid_pts:
        best = max(hybrid_pts, key=lambda p: p["mean_recall"])
        stages.append({
            "label": f"Hybrid\n(BM25={best['kw']:.0%}+Sem={best['sw']:.0%})",
            "recall": best["mean_recall"],
            "color": MIT_CRIMSON,
        })
    elif "Hybrid" in aggregates:
        stages.append({
            "label": "Hybrid\n(BM25=30%+Sem=70%)",
            "recall": aggregates["Hybrid"]["Recall@K"],
            "color": MIT_CRIMSON,
        })

    # LLM-Rerank adds reasoning on top
    if "LLM-Rerank" in aggregates:
        stages.append({
            "label": "LLM-Rerank\n(hybrid + LLM judge)",
            "recall": aggregates["LLM-Rerank"]["Recall@K"],
            "color": DARK_NAVY,
        })

    if len(stages) < 2:
        print("  Chart 6 skipped: not enough approaches for progression")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(stages))
    bar_vals = [s["recall"] for s in stages]
    colors = [s["color"] for s in stages]
    labels = [s["label"] for s in stages]

    bars = ax.bar(x, bar_vals, width=0.6, color=colors, edgecolor="white",
                  linewidth=1.5, zorder=3)

    # Value labels on bars
    for bar, val in zip(bars, bar_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.1%}", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color="#333333")

    # Delta arrows between stages
    for i in range(1, len(stages)):
        prev = bar_vals[i - 1]
        curr = bar_vals[i]
        delta = curr - prev
        sign = "+" if delta >= 0 else ""
        mid_y = (prev + curr) / 2
        color = "#2e7d32" if delta >= 0 else "#c62828"

        ax.annotate(
            "", xy=(i, curr), xytext=(i - 1, prev),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=2,
                connectionstyle="arc3,rad=-0.2",
            ),
        )
        # Delta text along the arrow
        mid_x = (i - 1 + i) / 2
        ax.text(mid_x, max(prev, curr) + 0.04, f"{sign}{delta:.1%}",
                ha="center", fontsize=10, fontweight="bold", color=color)

    # Pass threshold
    ax.axhline(y=0.4, color=MIT_CRIMSON, linestyle="--", linewidth=1, alpha=0.5, zorder=2)
    ax.text(len(stages) - 0.7, 0.415, "40% pass threshold", fontsize=8,
            color=MIT_CRIMSON, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Effective Recall@K", fontsize=11)
    ax.set_title("Retrieval Quality: From Single Methods to Combined Approaches",
                  fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0, max(bar_vals) + 0.15)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart6_approach_progression.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 6 saved: {path}")
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
    chart_weight_sweep(data, args.output_dir)
    chart_approach_progression(data, args.output_dir)
    print("\nDone! All charts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
