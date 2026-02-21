#!/usr/bin/env python3
"""Generate benchmark charts from results.json.

Usage:
    uv run python scripts/generate_charts.py [--results path] [--output-dir path]

Reads benchmark_results/results.json (produced by pytest conftest.py plugin)
and generates PNG charts with MIT AI Studio branding.

Design: minimal, data-focused, inspired by artificialanalysis.ai.
Each chart tells exactly one story.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Brand constants ──────────────────────────────────────────────────────────

MIT_CRIMSON = "#A31F34"
MIT_GRAY = "#8A8B8C"

# Approach family colors
CLR_HYBRID = "#D62728"
CLR_BM25 = "#1F77B4"
CLR_SEMANTIC = "#FF7F0E"

# Named colors for each approach
APPROACH_COLORS = {
    "Hybrid":         "#D62728",
    "Hybrid+Expand":  "#E45756",
    "Hybrid+RRF":     "#B22222",
    "BM25":           "#1F77B4",
    "Semantic":       "#FF7F0E",
}

_LLM_PALETTE = ["#2CA02C", "#17BECF", "#9467BD", "#8C564B"]


def _color(name):
    if name in APPROACH_COLORS:
        return APPROACH_COLORS[name]
    if name.startswith("LLM:"):
        llm_keys = [k for k in APPROACH_COLORS if k.startswith("LLM:")]
        idx = len(llm_keys)
        c = _LLM_PALETTE[idx % len(_LLM_PALETTE)]
        APPROACH_COLORS[name] = c
        return c
    return "#333333"


# Cost per 1,000 queries (OpenRouter pricing, Feb 2026)
# ~8,500 input + ~750 output tokens per query
COST_MAP = {
    "gemini":   1.15,   # $0.10/M in, $0.40/M out
    "nemotron": 11.10,  # $1.20/M in, $1.20/M out
    "llama":    1.09,   # $0.10/M in, $0.32/M out
}

DPI = 300
FOOTER = "Research: Brandon Sneider | MIT AI Studio (MAS.664/665)"


def _setup():
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    font = "Helvetica Neue" if "Helvetica Neue" in available else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
    })


def _brand(fig):
    """Thin crimson header + attribution footer."""
    fig.patches.append(mpatches.FancyBboxPatch(
        (0, 0.988), 1.0, 0.012,
        boxstyle="square,pad=0", facecolor=MIT_CRIMSON,
        edgecolor="none", transform=fig.transFigure, zorder=100,
    ))
    fig.text(0.98, 0.008, FOOTER, ha="right", va="bottom",
             fontsize=7, color=MIT_GRAY, transform=fig.transFigure)


def _cost(name):
    """Return cost per 1K queries for an approach."""
    nl = name.lower()
    if not name.startswith("LLM:"):
        return 0.0
    for key, val in COST_MAP.items():
        if key in nl:
            return val
    return 2.0


# ── Chart 1: Quality vs Cost scatter ─────────────────────────────────────────


def chart_quality_vs_cost(data, output_dir):
    """Horizontal bar: Recall@K sorted by quality, cost annotated on right.

    Story: free hybrid approaches top the chart.
    Paid LLM rerankers are at the bottom — worse AND expensive.
    """
    _setup()
    agg = data["aggregates"]
    approaches = data["approaches"]

    # Sort by recall ascending (best on top when plotted)
    sorted_apps = sorted(approaches, key=lambda a: agg[a]["Recall@K"])

    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(sorted_apps))
    recalls = [agg[a]["Recall@K"] for a in sorted_apps]
    colors = [_color(a) for a in sorted_apps]
    costs = [_cost(a) for a in sorted_apps]

    bars = ax.barh(y, recalls, color=colors, alpha=0.85, height=0.6, zorder=3,
                   edgecolor="white", linewidth=0.5)

    # Recall value on bar + cost on far right
    max_recall = max(recalls)
    for i, (bar, recall, cost, name) in enumerate(zip(bars, recalls, costs, sorted_apps)):
        # Recall label just past bar end
        weight = "bold" if recall == max_recall else "normal"
        ax.text(recall + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{recall:.1%}", va="center", fontsize=10, fontweight=weight,
                color="#333")

        # Cost tag on the far right
        cost_str = f"${cost:.2f}/1K" if cost > 0 else "Free"
        cost_color = "#1b5e20" if cost == 0 else "#b71c1c"
        ax.text(0.99, bar.get_y() + bar.get_height() / 2,
                cost_str, va="center", ha="right", fontsize=9,
                fontweight="bold", color=cost_color,
                transform=ax.get_yaxis_transform())

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_apps, fontsize=10, fontweight="bold")
    ax.set_xlabel("Recall@K", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0.62, max_recall + 0.06)
    ax.set_title("Retrieval Quality vs. Cost", fontsize=15, fontweight="bold",
                 pad=20)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _brand(fig)

    path = output_dir / "chart1_quality_vs_cost.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 1 saved: {path}")
    return path


# ── Chart 2: Hybrid Mix Optimization ─────────────────────────────────────────


def chart_hybrid_mix(data, output_dir):
    """Line: how recall changes as BM25/Semantic ratio varies.

    Story: combining signals (hybrid) beats either alone.
    The optimal mix at ~70-80% semantic outperforms all LLM rerankers — for free.
    """
    _setup()
    sweep = data.get("weight_sweep", [])
    if not sweep:
        print("  Chart 2 skipped: no weight_sweep data")
        return None

    hybrid_pts = [p for p in sweep if p["kw"] is not None]
    llm_pts = [p for p in sweep if p["kw"] is None]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = [p["sw"] for p in hybrid_pts]
    y = [p["mean_recall"] for p in hybrid_pts]

    ax.fill_between(x, y, alpha=0.08, color=CLR_HYBRID)
    ax.plot(x, y, "o-", color=CLR_HYBRID, linewidth=2.5, markersize=6,
            zorder=4, label="Hybrid (free)")

    # Best point
    best_i = max(range(len(y)), key=lambda i: y[i])
    ax.annotate(
        f"Best: {x[best_i]:.0%} semantic\n{y[best_i]:.1%} recall",
        xy=(x[best_i], y[best_i]),
        xytext=(x[best_i] - 0.22, y[best_i] + 0.035),
        fontsize=9, fontweight="bold", color=CLR_HYBRID,
        arrowprops=dict(arrowstyle="->", color=CLR_HYBRID, lw=1.5),
    )

    # LLM reference lines — single band showing range
    if llm_pts:
        llm_recalls = [p["mean_recall"] for p in llm_pts]
        llm_min, llm_max = min(llm_recalls), max(llm_recalls)
        ax.axhspan(llm_min - 0.002, llm_max + 0.002, alpha=0.12,
                   color="#2CA02C", zorder=1)
        ax.axhline(y=np.mean(llm_recalls), color="#2CA02C", linestyle="--",
                   linewidth=1.5, alpha=0.6, zorder=2)
        # Label inside the chart area to avoid right-edge overlap
        ax.text(0.45, llm_max + 0.008,
                f"LLM Rerankers ({llm_min:.1%}–{llm_max:.1%}, $1–$11/1K)",
                fontsize=8, color="#2CA02C", va="bottom", fontweight="bold")

    # Endpoint labels
    ax.text(-0.02, y[0] - 0.012, "Pure\nBM25", fontsize=8, ha="right",
            color=CLR_BM25, fontweight="bold", va="top")
    ax.text(1.02, y[-1] + 0.012, "Pure\nSemantic", fontsize=8, ha="left",
            color=CLR_SEMANTIC, fontweight="bold", va="bottom")

    ax.set_xlabel("Semantic Weight  (BM25 = 1 − semantic)", fontsize=11)
    ax.set_ylabel("Recall@K", fontsize=11)
    ax.set_title("Hybrid Search: Finding the Optimal Mix",
                 fontsize=15, fontweight="bold", pad=20)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(min(y) - 0.04, max(y) + 0.06)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _brand(fig)

    path = output_dir / "chart2_hybrid_mix.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 2 saved: {path}")
    return path


# ── Chart 3: Summary Table ───────────────────────────────────────────────────


def chart_summary_table(data, output_dir):
    """Clean summary table — the reference card."""
    _setup()

    approaches = data["approaches"]
    agg = data["aggregates"]
    n_app = len(approaches)

    metrics = [
        ("Recall@K", "How many right answers did we find?"),
        ("MRR",      "How high is the first useful result?"),
        ("NDCG@K",   "Are the best results near the top?"),
        ("pass_rate", "% of queries above 40% threshold"),
    ]

    # Add cost row
    col_labels = [""] + approaches
    rows = []
    row_colors_list = []

    for idx, (mkey, mdesc) in enumerate(metrics):
        vals = [agg[a][mkey] for a in approaches]
        best = max(vals)
        row = [mdesc]
        colors = ["#f9f9f9" if idx % 2 == 0 else "white"]
        base = colors[0]
        for v in vals:
            row.append(f"{v:.1%}")
            colors.append("#e8f5e9" if v == best else base)
        rows.append(row)
        row_colors_list.append(colors)

    # Cost row
    cost_row = ["Cost / 1K queries"]
    cost_colors = ["#f9f9f9"]
    costs = [_cost(a) for a in approaches]
    min_cost = min(costs)
    for c in costs:
        cost_row.append(f"${c:.2f}" if c > 0 else "Free")
        cost_colors.append("#e8f5e9" if c == min_cost else "#f9f9f9")
    rows.append(cost_row)
    row_colors_list.append(cost_colors)

    fig_w = max(10, 3 + n_app * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))
    ax.axis("off")
    ax.set_title("Benchmark Summary — 35 queries, 8 approaches",
                 fontsize=13, fontweight="bold", pad=15)

    table = ax.table(
        cellText=rows, colLabels=col_labels, cellColours=row_colors_list,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    fs = 8.5 if n_app <= 5 else 7 if n_app <= 8 else 6
    table.set_fontsize(fs)
    table.scale(1.0, 1.8)

    # Column widths
    desc_w = 0.20
    app_w = (1.0 - desc_w) / n_app
    for r in range(len(rows) + 1):
        table[r, 0].set_width(desc_w)
        for j in range(1, n_app + 1):
            table[r, j].set_width(app_w)

    # Header style
    for j in range(n_app + 1):
        cell = table[0, j]
        cell.set_facecolor(MIT_CRIMSON)
        cell.set_text_props(color="white", fontweight="bold")

    # Description column left-aligned, muted
    for i in range(1, len(rows) + 1):
        table[i, 0].get_text().set_ha("left")
        table[i, 0].get_text().set_fontsize(max(5.5, fs - 1))
        table[i, 0].get_text().set_color("#666")

    # Bold best values
    for i, (mkey, _) in enumerate(metrics):
        vals = [agg[a][mkey] for a in approaches]
        best = max(vals)
        for j, v in enumerate(vals):
            if v == best:
                table[i + 1, j + 1].set_text_props(fontweight="bold", color="#1b5e20")

    # Bold free cost
    for j, c in enumerate(costs):
        if c == min_cost:
            table[len(metrics) + 1, j + 1].set_text_props(fontweight="bold", color="#1b5e20")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _brand(fig)

    path = output_dir / "chart3_summary_table.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Chart 3 saved: {path}")
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark charts")
    parser.add_argument("--results", type=Path,
                        default=Path("benchmark_results/results.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("benchmark_results"))
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
    chart_quality_vs_cost(data, args.output_dir)
    chart_hybrid_mix(data, args.output_dir)
    chart_summary_table(data, args.output_dir)
    print("\nDone! All charts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
