#!/usr/bin/env python3
"""Generate benchmark charts from results.json.

Usage:
    uv run python scripts/generate_charts.py [--results path] [--output-dir path]

Design: minimal, data-focused, inspired by artificialanalysis.ai.
Each chart tells exactly one story. Soft colors, generous whitespace.
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

# ── Brand ────────────────────────────────────────────────────────────────────

MIT_CRIMSON = "#A31F34"
MIT_GRAY = "#8A8B8C"
BG_COLOR = "#FAFAFA"

# Softer, desaturated palette — easy on the eyes
CLR_HYBRID = "#C0392B"
CLR_BM25 = "#2980B9"
CLR_SEMANTIC = "#E67E22"
CLR_LLM = "#27AE60"

APPROACH_COLORS = {
    "Hybrid":        CLR_HYBRID,
    "Hybrid+Expand": "#E74C3C",
    "Hybrid+RRF":    "#922B21",
    "BM25":          CLR_BM25,
    "Semantic":      CLR_SEMANTIC,
}
_LLM_PALETTE = ["#27AE60", "#16A085", "#8E44AD", "#7F8C8D"]


def _color(name):
    if name in APPROACH_COLORS:
        return APPROACH_COLORS[name]
    if name.startswith("LLM:"):
        llm_keys = [k for k in APPROACH_COLORS if k.startswith("LLM:")]
        c = _LLM_PALETTE[len(llm_keys) % len(_LLM_PALETTE)]
        APPROACH_COLORS[name] = c
        return c
    return "#555"


COST_MAP = {
    "gemini":   1.15,
    "nemotron": 11.10,
    "llama":    1.09,
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
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#ccc",
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linewidth": 0.4,
        "grid.color": "#bbb",
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "white",
        "xtick.color": "#555",
        "ytick.color": "#555",
        "text.color": "#333",
    })


def _brand(fig):
    fig.patches.append(mpatches.FancyBboxPatch(
        (0, 0.99), 1.0, 0.01,
        boxstyle="square,pad=0", facecolor=MIT_CRIMSON,
        edgecolor="none", transform=fig.transFigure, zorder=100,
    ))
    fig.text(0.98, 0.006, FOOTER, ha="right", va="bottom",
             fontsize=6.5, color="#aaa", transform=fig.transFigure)


def _cost(name):
    if not name.startswith("LLM:"):
        return 0.0
    for key, val in COST_MAP.items():
        if key in name.lower():
            return val
    return 2.0


# ── Chart 1: Quality vs Cost ─────────────────────────────────────────────────


def chart_quality_vs_cost(data, output_dir):
    """Horizontal bars ranked by recall, cost annotated."""
    _setup()
    agg = data["aggregates"]
    approaches = data["approaches"]
    sorted_apps = sorted(approaches, key=lambda a: agg[a]["Recall@K"])

    fig, ax = plt.subplots(figsize=(9, 5))

    y = np.arange(len(sorted_apps))
    recalls = [agg[a]["Recall@K"] for a in sorted_apps]
    colors = [_color(a) for a in sorted_apps]
    costs = [_cost(a) for a in sorted_apps]

    bars = ax.barh(y, recalls, color=colors, alpha=0.78, height=0.55, zorder=3,
                   edgecolor="white", linewidth=0.8)

    # Separator line between free and paid
    free_count = sum(1 for c in costs if c == 0)
    if 0 < free_count < len(sorted_apps):
        ax.axhline(y=free_count - 0.5, color="#ddd", linewidth=1, linestyle="-", zorder=2)
        ax.text(0.625, free_count - 0.5 + 0.15, "FREE", fontsize=7,
                color="#aaa", fontweight="bold", va="bottom")
        ax.text(0.625, free_count - 0.5 - 0.15, "PAID", fontsize=7,
                color="#aaa", fontweight="bold", va="top")

    max_recall = max(recalls)
    for bar, recall, cost in zip(bars, recalls, costs):
        weight = "bold" if recall == max_recall else "normal"
        ax.text(recall + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{recall:.1%}", va="center", fontsize=9.5, fontweight=weight,
                color="#333")
        cost_str = f"${cost:.2f}/1K" if cost > 0 else "Free"
        cost_color = "#27AE60" if cost == 0 else "#C0392B"
        ax.text(0.99, bar.get_y() + bar.get_height() / 2,
                cost_str, va="center", ha="right", fontsize=8.5,
                fontweight="bold", color=cost_color,
                transform=ax.get_yaxis_transform())

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_apps, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Recall@K", fontsize=11, color="#555")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0.62, max_recall + 0.06)
    ax.set_title("Retrieval Quality vs. Cost", fontsize=14, fontweight="bold",
                 pad=18, color="#222")

    fig.tight_layout(rect=[0, 0.025, 1, 0.975])
    _brand(fig)
    path = output_dir / "chart1_quality_vs_cost.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Chart 1 saved: {path}")
    return path


# ── Chart 2: Hybrid Mix ──────────────────────────────────────────────────────


def chart_hybrid_mix(data, output_dir):
    """Line chart: BM25/Semantic weight sweep with LLM baseline band."""
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

    # Smooth the curve slightly for visual appeal
    ax.fill_between(x, y, alpha=0.06, color=CLR_HYBRID)
    ax.plot(x, y, "o-", color=CLR_HYBRID, linewidth=2, markersize=5.5,
            zorder=4, markeredgecolor="white", markeredgewidth=1.2)

    best_i = max(range(len(y)), key=lambda i: y[i])
    ax.annotate(
        f"Best: {x[best_i]:.0%} semantic\n{y[best_i]:.1%} recall",
        xy=(x[best_i], y[best_i]),
        xytext=(x[best_i] - 0.22, y[best_i] + 0.03),
        fontsize=8.5, fontweight="bold", color=CLR_HYBRID,
        arrowprops=dict(arrowstyle="->", color=CLR_HYBRID, lw=1.2),
    )

    if llm_pts:
        llm_recalls = [p["mean_recall"] for p in llm_pts]
        llm_min, llm_max = min(llm_recalls), max(llm_recalls)
        ax.axhspan(llm_min - 0.002, llm_max + 0.002, alpha=0.08,
                   color=CLR_LLM, zorder=1)
        ax.axhline(y=np.mean(llm_recalls), color=CLR_LLM, linestyle="--",
                   linewidth=1.2, alpha=0.5, zorder=2)
        ax.text(0.45, llm_max + 0.006,
                f"LLM Rerankers ({llm_min:.1%}–{llm_max:.1%}, $1–$11/1K)",
                fontsize=7.5, color=CLR_LLM, va="bottom", fontweight="bold")

    ax.text(-0.02, y[0] - 0.01, "Pure\nBM25", fontsize=7.5, ha="right",
            color=CLR_BM25, fontweight="bold", va="top")
    ax.text(1.02, y[-1] + 0.01, "Pure\nSemantic", fontsize=7.5, ha="left",
            color=CLR_SEMANTIC, fontweight="bold", va="bottom")

    ax.set_xlabel("Semantic Weight  (BM25 = 1 − semantic)", fontsize=10, color="#555")
    ax.set_ylabel("Recall@K", fontsize=10, color="#555")
    ax.set_title("Hybrid Search: Finding the Optimal Mix",
                 fontsize=14, fontweight="bold", pad=18, color="#222")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(min(y) - 0.04, max(y) + 0.055)

    fig.tight_layout(rect=[0, 0.025, 1, 0.975])
    _brand(fig)
    path = output_dir / "chart2_hybrid_mix.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Chart 2 saved: {path}")
    return path


# ── Chart 3: Summary Table ───────────────────────────────────────────────────


def chart_summary_table(data, output_dir):
    """Clean summary table with cost row."""
    _setup()
    approaches = data["approaches"]
    agg = data["aggregates"]
    n_app = len(approaches)

    metrics = [
        ("Recall@K", "How many right answers found?"),
        ("MRR",      "First useful result rank?"),
        ("NDCG@K",   "Best results near the top?"),
        ("pass_rate", "Queries above 40% threshold"),
    ]

    col_labels = [""] + approaches
    rows = []
    row_colors_list = []

    for idx, (mkey, mdesc) in enumerate(metrics):
        vals = [agg[a][mkey] for a in approaches]
        best = max(vals)
        row = [mdesc]
        base = "#f5f5f5" if idx % 2 == 0 else "white"
        colors = [base]
        for v in vals:
            row.append(f"{v:.1%}")
            colors.append("#e8f5e9" if v == best else base)
        rows.append(row)
        row_colors_list.append(colors)

    # Cost row
    cost_row = ["Cost / 1K queries"]
    cost_colors = ["#f5f5f5"]
    costs = [_cost(a) for a in approaches]
    for c in costs:
        cost_row.append(f"${c:.2f}" if c > 0 else "Free")
        cost_colors.append("#e8f5e9" if c == 0 else "#f5f5f5")
    rows.append(cost_row)
    row_colors_list.append(cost_colors)

    fig_w = max(10, 3 + n_app * 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 3.8))
    ax.axis("off")
    ax.set_title("Benchmark Summary — 35 queries, 8 approaches",
                 fontsize=12, fontweight="bold", pad=15, color="#222")

    table = ax.table(
        cellText=rows, colLabels=col_labels, cellColours=row_colors_list,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    fs = 8 if n_app <= 5 else 6.8 if n_app <= 8 else 6
    table.set_fontsize(fs)
    table.scale(1.0, 1.9)

    desc_w = 0.17
    app_w = (1.0 - desc_w) / n_app
    for r in range(len(rows) + 1):
        table[r, 0].set_width(desc_w)
        for j in range(1, n_app + 1):
            table[r, j].set_width(app_w)

    for j in range(n_app + 1):
        cell = table[0, j]
        cell.set_facecolor(MIT_CRIMSON)
        cell.set_text_props(color="white", fontweight="bold", fontsize=fs)

    for i in range(1, len(rows) + 1):
        table[i, 0].get_text().set_ha("left")
        table[i, 0].get_text().set_fontsize(max(5.5, fs - 0.5))
        table[i, 0].get_text().set_color("#777")

    for i, (mkey, _) in enumerate(metrics):
        vals = [agg[a][mkey] for a in approaches]
        best = max(vals)
        for j, v in enumerate(vals):
            if v == best:
                table[i + 1, j + 1].set_text_props(fontweight="bold", color="#1b5e20")

    for j, c in enumerate(costs):
        if c == 0:
            table[len(metrics) + 1, j + 1].set_text_props(fontweight="bold", color="#1b5e20")

    fig.tight_layout(rect=[0, 0.025, 1, 0.955])
    _brand(fig)
    path = output_dir / "chart3_summary_table.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Chart 3 saved: {path}")
    return path


# ── Chart 4: Where Each Approach Wins ─────────────────────────────────────────


def chart_approach_strengths(data, output_dir):
    """Dot plot: recall by query type for each approach family."""
    _setup()
    per_level = data["per_level"]
    approaches = data["approaches"]

    query_types = [
        ("Keyword\nLookup",        ["L1", "L3"],  "Exact terms in content"),
        ("Semantic\nUnderstanding", ["L2", "L4"],  "Meaning-based retrieval"),
        ("Cross-Source\nReasoning", ["L5", "L7"],  "Combining multiple pages"),
        ("Paraphrased\nQueries",    ["L6"],        "Different user wording"),
        ("Temporal\nComparison",    ["L9", "L12"], "Cross-semester changes"),
        ("Complex\nReasoning",     ["L10", "L11"], "Inference & aggregation"),
    ]

    families = {
        "Hybrid":     {"key": ["Hybrid", "Hybrid+Expand", "Hybrid+RRF"],
                       "color": CLR_HYBRID, "marker": "o"},
        "BM25":       {"key": ["BM25"],
                       "color": CLR_BM25, "marker": "s"},
        "Semantic":   {"key": ["Semantic"],
                       "color": CLR_SEMANTIC, "marker": "D"},
        "LLM Rerank": {"key": [a for a in approaches if a.startswith("LLM:")],
                       "color": CLR_LLM, "marker": "^"},
    }

    type_scores = {}
    for type_label, levels, _ in query_types:
        present = [lv for lv in levels if lv in per_level]
        if not present:
            continue
        total_n = sum(per_level[lv]["n"] for lv in present)
        type_scores[type_label] = {}
        for fam_name, fam_info in families.items():
            best_recall = 0
            for variant in fam_info["key"]:
                if variant not in approaches:
                    continue
                weighted = sum(
                    per_level[lv]["approaches"].get(variant, {}).get("mean_recall", 0)
                    * per_level[lv]["n"]
                    for lv in present
                )
                best_recall = max(best_recall, weighted / total_n if total_n else 0)
            type_scores[type_label][fam_name] = best_recall

    type_labels = [t[0] for t in query_types if t[0] in type_scores]
    type_descs = {t[0]: t[2] for t in query_types}
    n_types = len(type_labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(n_types)

    # Connecting lines (range)
    for i, tl in enumerate(type_labels):
        scores = list(type_scores[tl].values())
        ax.plot([min(scores), max(scores)], [i, i],
                color="#e0e0e0", linewidth=2.5, zorder=1, solid_capstyle="round")

    # Dots
    for fam_name, fam_info in families.items():
        x_vals = [type_scores[tl][fam_name] for tl in type_labels]
        ax.scatter(x_vals, y_pos, s=140, c=fam_info["color"],
                   marker=fam_info["marker"], label=fam_name,
                   zorder=5, edgecolors="white", linewidth=1.5, alpha=0.9)

        for i, (xv, tl) in enumerate(zip(x_vals, type_labels)):
            if xv == max(type_scores[tl].values()):
                ax.annotate(f"{xv:.0%}", (xv, i), fontsize=7.5,
                            fontweight="bold", color=fam_info["color"],
                            xytext=(0, 11), textcoords="offset points",
                            ha="center", va="bottom")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(type_labels, fontsize=9.5, fontweight="bold")

    for i, tl in enumerate(type_labels):
        ax.text(1.02, i, type_descs[tl], fontsize=7, color="#aaa",
                va="center", transform=ax.get_yaxis_transform())

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Recall@K", fontsize=11, color="#555")
    ax.set_title("Where Each Approach Wins", fontsize=14,
                 fontweight="bold", pad=18, color="#222")
    ax.set_xlim(0.25, 1.05)
    ax.invert_yaxis()

    ax.legend(loc="lower left", framealpha=0.95, fontsize=8.5,
              ncol=4, columnspacing=1.2, edgecolor="#ddd")

    fig.tight_layout(rect=[0, 0.025, 0.76, 0.975])
    _brand(fig)
    path = output_dir / "chart4_approach_strengths.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  Chart 4 saved: {path}")
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
    chart_approach_strengths(data, args.output_dir)
    print("\nDone! All charts saved to:", args.output_dir)


if __name__ == "__main__":
    main()
