#!/usr/bin/env python3
"""Generate publication-quality benchmark charts from results.json.

Usage:
    uv run python scripts/generate_charts.py [--results path] [--output-dir path]

Reads benchmark_results/results.json (produced by pytest conftest.py plugin)
and generates PNG charts with MIT AI Studio branding.
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
from matplotlib.colors import LinearSegmentedColormap

# ── Brand constants ──────────────────────────────────────────────────────────

# High-contrast, colorblind-friendly palette
CLR_HYBRID = "#D62728"    # vivid red
CLR_BM25 = "#1F77B4"      # standard blue
CLR_SEMANTIC = "#FF7F0E"   # orange
CLR_LLM = "#2CA02C"        # green

MIT_CRIMSON = "#A31F34"
MIT_GRAY = "#8A8B8C"

# Base colors for known approaches; LLM variants get auto-assigned
APPROACH_COLORS = {
    "Hybrid": CLR_HYBRID,
    "Hybrid+Expand": "#E45756",   # lighter red
    "Hybrid+RRF": "#B22222",      # darker red (firebrick)
    "BM25": CLR_BM25,
    "Semantic": CLR_SEMANTIC,
    "LLM-Rerank": CLR_LLM,
}

# Extended palette for multiple LLM models
_LLM_COLORS = ["#2CA02C", "#17BECF", "#9467BD", "#8C564B", "#E377C2"]


def _get_color(approach_name):
    """Return color for an approach, auto-assigning for LLM variants."""
    if approach_name in APPROACH_COLORS:
        return APPROACH_COLORS[approach_name]
    # Auto-assign colors for LLM:* variants
    llm_approaches = [a for a in APPROACH_COLORS if a.startswith("LLM:")]
    if approach_name.startswith("LLM:"):
        idx = len(llm_approaches)
        color = _LLM_COLORS[idx % len(_LLM_COLORS)]
        APPROACH_COLORS[approach_name] = color
        return color
    return "#333333"

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
    fig.patches.append(mpatches.FancyBboxPatch(
        (0, 0.985), 1.0, 0.015,
        boxstyle="square,pad=0",
        facecolor=MIT_CRIMSON,
        edgecolor="none",
        transform=fig.transFigure,
        zorder=100,
    ))
    fig.text(
        0.98, 0.008, FOOTER_TEXT,
        ha="right", va="bottom",
        fontsize=8, color=MIT_GRAY,
        transform=fig.transFigure,
    )


# ── Chart 1: Recall by Level — focused grouped bar ──────────────────────────


def chart_recall_by_category(data, output_dir):
    """Grouped bar chart of mean Recall@K by difficulty category."""
    _setup_font()

    per_level = data["per_level"]
    approaches = data["approaches"]
    n_approaches = len(approaches)

    # Aggregate levels into 4 categories with weighted averages
    categories = [
        ("Easy", ["L1", "L2"]),
        ("Hard", ["L3", "L4", "L5"]),
        ("Differentiating", ["L6", "L7", "L8"]),
        ("Complex", ["L9", "L10", "L11", "L12"]),
    ]

    cat_data = {}
    cat_n = {}
    for cat_label, level_list in categories:
        present = [lv for lv in level_list if lv in per_level]
        total_n = sum(per_level[lv]["n"] for lv in present)
        cat_n[cat_label] = total_n
        cat_data[cat_label] = {}
        for aname in approaches:
            weighted_sum = sum(
                per_level[lv]["approaches"].get(aname, {}).get("mean_recall", 0)
                * per_level[lv]["n"]
                for lv in present
            )
            cat_data[cat_label][aname] = weighted_sum / total_n if total_n else 0

    n_cats = len(categories)
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.8 / n_approaches
    x = np.arange(n_cats)

    for i, aname in enumerate(approaches):
        color = _get_color(aname)
        vals = [cat_data[cat_label][aname] for cat_label, _ in categories]
        offset = (i - n_approaches / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width * 0.88, label=aname, color=color,
                      alpha=0.9, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Pass threshold
    ax.axhline(y=0.4, color="#999999", linestyle="--", linewidth=1, alpha=0.6, zorder=2)
    ax.text(n_cats - 0.5, 0.415, "40% pass", fontsize=8, color="#999999", ha="right")

    # X-axis: category names with query counts and level ranges
    cat_labels = [f"{cat_label}\n(n={cat_n[cat_label]})"
                  for cat_label, _ in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    for i, (_, lvs) in enumerate(categories):
        lmin = min(int(lv[1:]) for lv in lvs)
        lmax = max(int(lv[1:]) for lv in lvs)
        ax.text(i, -0.12, f"L{lmin}\u2013L{lmax}", fontsize=8, ha="center", color="#888888",
                fontstyle="italic", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xlabel("Difficulty Category", fontsize=11)
    ax.set_ylabel("Mean Effective Recall@K", fontsize=11)
    ax.set_title("Retrieval Recall by Difficulty Category",
                 fontsize=14, fontweight="bold", pad=20)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart1_recall_by_category.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 1 saved: {path}")
    return path


# ── Chart 2: Failure Analysis — where approaches disagree ─────────────────────


def chart_failure_analysis(data, output_dir):
    """Show only queries where at least one approach fails (recall < 40%).

    This is much more actionable than a full heatmap — it highlights exactly
    where the approaches differ and which queries are hard.
    """
    _setup_font()

    per_query = data["per_query"]
    approaches = data["approaches"]

    # Filter to queries where at least one approach fails
    THRESHOLD = 0.4
    interesting = []
    for q in per_query:
        recalls = {a: q["approaches"].get(a, {}).get("effective_recall", 0) for a in approaches}
        any_fail = any(v < THRESHOLD for v in recalls.values())
        if any_fail:
            interesting.append(q)

    if not interesting:
        # All queries pass for all approaches — show a simple message
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "All queries pass the 40% threshold for all approaches!",
                ha="center", va="center", fontsize=14, fontweight="bold", color="#2e7d32")
        add_branding(fig)
        path = output_dir / "chart2_failure_analysis.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Chart 2 saved: {path}")
        return path

    n_queries = len(interesting)
    n_approaches = len(approaches)

    fig_height = max(4, n_queries * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(max(8, n_approaches * 1.8), fig_height))

    matrix = np.zeros((n_queries, n_approaches))
    query_labels = []
    for i, q in enumerate(interesting):
        label = q["query"][:50]
        if q["scoring"] == "inverse":
            label += " [INV]"
        query_labels.append(f"[{q['level']}] {label}")
        for j, aname in enumerate(approaches):
            matrix[i, j] = q["approaches"].get(aname, {}).get("effective_recall", 0)

    # Simple two-tone: pass (light green) / fail (light red)
    cmap = LinearSegmentedColormap.from_list("passfail", [
        "#ffcdd2",  # fail red
        "#ffcdd2",  # fail red
        "#c8e6c9",  # pass green
        "#c8e6c9",  # pass green
    ])

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for i in range(n_queries):
        for j in range(n_approaches):
            val = matrix[i, j]
            status = "PASS" if val >= THRESHOLD else "FAIL"
            color = "#1b5e20" if val >= THRESHOLD else "#b71c1c"
            ax.text(j, i, f"{val:.0%}\n{status}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(range(n_approaches))
    ax.set_xticklabels(approaches, fontsize=9, fontweight="bold", rotation=30, ha="right")
    ax.set_yticks(range(n_queries))
    ax.set_yticklabels(query_labels, fontsize=7)
    ax.set_title(f"Failure Analysis: {n_queries} Queries Where Approaches Disagree",
                 fontsize=13, fontweight="bold", pad=15)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart2_failure_analysis.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 2 saved: {path}")
    return path


# ── Chart 3: Summary Metrics Table ───────────────────────────────────────────


def chart_summary_table(data, output_dir):
    """Styled table showing aggregate metrics as percentages."""
    _setup_font()

    approaches = data["approaches"]
    aggregates = data["aggregates"]

    metrics = [
        ("Recall@K", "Recall@K",
         "Of all right answers, how many did we find?"),
        ("MRR", "MRR",
         "How high is the first useful result?"),
        ("NDCG@K", "NDCG@K",
         "Are the best results near the top?"),
        ("pass_rate", "Pass Rate",
         "% of queries above 40% threshold"),
        ("pass^3", "pass^3",
         "Probability all 3 random queries pass"),
    ]

    col_labels = ["Metric", "What It Measures"] + approaches
    cell_text = []
    cell_colors = []

    for idx, (mname, mdisp, explanation) in enumerate(metrics):
        row_vals = [aggregates.get(a, {}).get(mname, 0) for a in approaches]
        best_val = max(row_vals)
        row = [mdisp, explanation]
        for val in row_vals:
            row.append(f"{val:.1%}")
        cell_text.append(row)

        base = "#f7f7f7" if idx % 2 == 0 else "white"
        row_colors = [base] * len(col_labels)
        # Highlight best value cell
        for j, val in enumerate(row_vals):
            if val == best_val and best_val > 0:
                row_colors[j + 2] = "#e8f5e9"  # light green highlight
        cell_colors.append(row_colors)

    n_cols = len(col_labels)
    n_app = len(approaches)
    # Scale figure width to fit all columns
    fig_width = max(10, 4 + n_app * 1.4)
    font_size = 9 if n_app <= 5 else 7.5 if n_app <= 8 else 6.5
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.axis("off")
    ax.set_title("Aggregate Benchmark Metrics", fontsize=14, fontweight="bold", pad=15)

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.8)

    # Auto-size columns proportionally
    metric_w = 0.06
    desc_w = 0.18
    remaining = 1.0 - metric_w - desc_w
    app_w = remaining / n_app
    for row_idx in range(len(metrics) + 1):
        table[row_idx, 0].set_width(metric_w)
        table[row_idx, 1].set_width(desc_w)
        for j in range(2, n_cols):
            table[row_idx, j].set_width(app_w)

    for i in range(1, len(metrics) + 1):
        table[i, 1].get_text().set_ha("left")
        table[i, 1].get_text().set_fontsize(max(6, font_size - 1.5))
        table[i, 1].get_text().set_color("#666666")

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor(MIT_CRIMSON)
        cell.set_text_props(color="white", fontweight="bold")

    # Bold best values
    for i, (mname, _, _) in enumerate(metrics):
        row_vals = [aggregates.get(a, {}).get(mname, 0) for a in approaches]
        best_val = max(row_vals)
        for j, val in enumerate(row_vals):
            if val == best_val and best_val > 0:
                cell = table[i + 1, j + 2]
                cell.set_text_props(fontweight="bold", color="#1b5e20")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_branding(fig)

    path = output_dir / "chart3_summary_table.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 3 saved: {path}")
    return path


# ── Chart 4: Weight Sweep — single axis, clean ──────────────────────────────


def chart_weight_sweep(data, output_dir):
    """Line chart showing how recall changes as BM25/Semantic ratio varies."""
    _setup_font()

    sweep = data.get("weight_sweep", [])
    if not sweep:
        print("  Chart 4 skipped: no weight_sweep data")
        return None

    hybrid_pts = [p for p in sweep if p["kw"] is not None]
    llm_pts = [p for p in sweep if p["kw"] is None]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = [p["sw"] for p in hybrid_pts]
    recalls = [p["mean_recall"] for p in hybrid_pts]

    # Fill area under curve
    ax.fill_between(x, recalls, alpha=0.15, color=CLR_HYBRID)
    ax.plot(x, recalls, "o-", color=CLR_HYBRID, linewidth=2.5, markersize=7,
            label="Hybrid Recall@K", zorder=4)

    # Best point
    best_idx = max(range(len(recalls)), key=lambda i: recalls[i])
    ax.annotate(
        f"Best: {x[best_idx]:.0%} semantic\n({recalls[best_idx]:.1%} recall)",
        xy=(x[best_idx], recalls[best_idx]),
        xytext=(x[best_idx] - 0.2, recalls[best_idx] + 0.04),
        fontsize=9, fontweight="bold", color=CLR_HYBRID,
        arrowprops=dict(arrowstyle="->", color=CLR_HYBRID, lw=1.5),
    )

    # LLM reference lines
    linestyles = [":", "--", "-."]
    for i, llm_pt in enumerate(llm_pts):
        color = _get_color(llm_pt["label"])
        ls = linestyles[i % len(linestyles)]
        ax.axhline(y=llm_pt["mean_recall"], color=color, linestyle=ls,
                    linewidth=2, alpha=0.8, zorder=2,
                    label=f"{llm_pt['label']} ({llm_pt['mean_recall']:.1%})")

    ax.text(-0.02, recalls[0] - 0.015, "Pure BM25", fontsize=8, ha="right",
            color=CLR_BM25, fontweight="bold")
    ax.text(1.02, recalls[-1] - 0.015, "Pure Semantic", fontsize=8, ha="left",
            color=CLR_SEMANTIC, fontweight="bold")

    ax.set_xlabel("Semantic Weight  (BM25 = 1 - semantic)", fontsize=11)
    ax.set_ylabel("Mean Recall@K", fontsize=11)
    ax.set_title("How BM25/Semantic Mix Affects Retrieval Quality",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(min(recalls) - 0.05, max(recalls) + 0.08)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart4_weight_sweep.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 4 saved: {path}")
    return path


# ── Chart 5: Approach Comparison — horizontal bars ───────────────────────────


def chart_approach_comparison(data, output_dir):
    """Horizontal bar chart comparing approaches on the 3 most important metrics."""
    _setup_font()

    aggregates = data.get("aggregates", {})
    approaches = data["approaches"]

    # Focus on the 3 metrics that matter most
    key_metrics = [
        ("Recall@K", "Recall@K"),
        ("MRR", "MRR"),
        ("pass_rate", "Pass Rate"),
    ]

    fig, axes = plt.subplots(1, len(key_metrics), figsize=(14, 5), sharey=True)

    y = np.arange(len(approaches))
    colors = [_get_color(a) for a in approaches]

    for ax, (mkey, mlabel) in zip(axes, key_metrics):
        vals = [aggregates.get(a, {}).get(mkey, 0) for a in approaches]
        best = max(vals)

        bars = ax.barh(y, vals, color=colors, alpha=0.85, height=0.6, zorder=3)

        for bar, val in zip(bars, vals):
            weight = "bold" if val == best else "normal"
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=10, fontweight=weight)

        ax.set_xlim(0, 1.08)
        ax.set_title(mlabel, fontsize=12, fontweight="bold")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.grid(axis="x", alpha=0.2, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(approaches, fontsize=10, fontweight="bold")

    fig.suptitle("Approach Comparison: Key Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_branding(fig)

    path = output_dir / "chart5_approach_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 5 saved: {path}")
    return path


# ── Chart 6: Cost Comparison ──────────────────────────────────────────────────


def chart_cost_comparison(data, output_dir):
    """Bar chart comparing cost per 1,000 queries for each approach."""
    _setup_font()

    approaches = data["approaches"]
    aggregates = data.get("aggregates", {})

    # Cost estimates per 1,000 queries (OpenRouter pricing, Feb 2026)
    # BM25/Semantic/Hybrid: all local computation, $0
    # LLM models: ~8,500 input tokens + ~750 output tokens per query
    # Gemini 2.0 Flash: $0.10/M in, $0.40/M out → ~$1.15/1K queries
    # Nemotron 70B:     $1.20/M in, $1.20/M out → ~$11.10/1K queries
    # Llama 3.3 70B:    $0.10/M in, $0.32/M out → ~$1.09/1K queries
    cost_per_1k = {}
    for aname in approaches:
        if aname in ("Hybrid", "BM25", "Semantic") or aname.startswith("Hybrid+"):
            cost_per_1k[aname] = 0.0
        elif "gemini" in aname.lower():
            cost_per_1k[aname] = 1.15
        elif "nemotron" in aname.lower():
            cost_per_1k[aname] = 11.10
        elif "llama-3.3" in aname.lower():
            cost_per_1k[aname] = 1.09
        else:
            cost_per_1k[aname] = 2.00  # default estimate for unknown LLM

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(approaches))
    colors = [_get_color(a) for a in approaches]
    costs = [cost_per_1k.get(a, 0) for a in approaches]
    recalls = [aggregates.get(a, {}).get("Recall@K", 0) for a in approaches]

    bars = ax.barh(y, costs, color=colors, alpha=0.85, height=0.6, zorder=3)

    for bar, cost, recall in zip(bars, costs, recalls):
        label = f"${cost:.2f}" if cost > 0 else "Free"
        recall_str = f" ({recall:.0%} recall)" if recall > 0 else ""
        ax.text(max(cost, 0.05) + 0.08, bar.get_y() + bar.get_height() / 2,
                f"{label}{recall_str}", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(approaches, fontsize=10, fontweight="bold")
    ax.set_xlabel("Cost per 1,000 Queries ($)", fontsize=11)
    ax.set_title("Retrieval Cost vs. Quality",
                 fontsize=14, fontweight="bold", pad=15)
    max_cost = max(costs) if max(costs) > 0 else 1.0
    ax.set_xlim(0, max_cost * 1.5)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_branding(fig)

    path = output_dir / "chart6_cost_comparison.png"
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
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
    )
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found.")
        print("Run 'uv run pytest tests/test_pageindex.py -v -s' first.")
        sys.exit(1)

    data = json.loads(args.results.read_text())
    print(f"Loaded {data['n_queries']} queries x {len(data['approaches'])} approaches")
    print(f"Generated at: {data['generated_at']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating charts...")
    chart_recall_by_category(data, args.output_dir)
    chart_failure_analysis(data, args.output_dir)
    chart_summary_table(data, args.output_dir)
    chart_weight_sweep(data, args.output_dir)
    chart_approach_comparison(data, args.output_dir)
    chart_cost_comparison(data, args.output_dir)
    print("\nDone! All charts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
