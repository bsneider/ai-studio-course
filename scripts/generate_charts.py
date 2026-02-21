#!/usr/bin/env python3
"""Generate benchmark charts from results.json.

Usage:
    uv run python scripts/generate_charts.py [--results path] [--output-dir path]

Design: minimal, data-focused. Each chart tells one story.
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
BG = "#FAFAFA"

# Muted, desaturated palette
C_HYB = "#C0392B"
C_BM25 = "#2980B9"
C_SEM = "#E67E22"
C_LLM = "#27AE60"

COLORS = {
    "Hybrid": C_HYB, "Hybrid+Expand": "#E74C3C", "Hybrid+RRF": "#922B21",
    "BM25": C_BM25, "Semantic": C_SEM,
}
_LLM_PAL = ["#27AE60", "#16A085", "#8E44AD", "#7F8C8D"]


def _c(name):
    if name in COLORS:
        return COLORS[name]
    if name.startswith("LLM:"):
        n = len([k for k in COLORS if k.startswith("LLM:")])
        COLORS[name] = _LLM_PAL[n % len(_LLM_PAL)]
        return COLORS[name]
    return "#555"


COST = {"gemini": 1.15, "nemotron": 11.10, "llama": 1.09}
DPI = 300
FOOTER = "Research: Brandon Sneider | MIT AI Studio (MAS.664/665)"


def _cost(name):
    if not name.startswith("LLM:"):
        return 0.0
    for k, v in COST.items():
        if k in name.lower():
            return v
    return 2.0


def _rc():
    import matplotlib.font_manager as fm
    f = "Helvetica Neue" if "Helvetica Neue" in {x.name for x in fm.fontManager.ttflist} else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [f], "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.5, "axes.edgecolor": "#ccc",
        "axes.grid": True, "grid.alpha": 0.1, "grid.linewidth": 0.4,
        "figure.facecolor": BG, "axes.facecolor": "white",
        "xtick.color": "#666", "ytick.color": "#666", "text.color": "#333",
    })


def _brand(fig):
    fig.patches.append(mpatches.FancyBboxPatch(
        (0, 0.99), 1.0, 0.01, boxstyle="square,pad=0",
        facecolor=MIT_CRIMSON, edgecolor="none",
        transform=fig.transFigure, zorder=100))
    fig.text(0.98, 0.005, FOOTER, ha="right", va="bottom",
             fontsize=6.5, color="#bbb", transform=fig.transFigure)


# ── Chart 1: Quality vs Cost ─────────────────────────────────────────────────


def chart_quality_vs_cost(data, output_dir):
    _rc()
    agg = data["aggregates"]
    apps = data["approaches"]
    ranked = sorted(apps, key=lambda a: agg[a]["Recall@K"])

    recalls = [agg[a]["Recall@K"] for a in ranked]
    costs = [_cost(a) for a in ranked]
    colors = [_c(a) for a in ranked]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(ranked))

    bars = ax.barh(y, recalls, color=colors, alpha=0.82, height=0.55,
                   zorder=3, edgecolor="white", linewidth=0.8)

    # Separator between free and paid
    free_n = sum(1 for c in costs if c == 0)
    if 0 < free_n < len(ranked):
        sep_y = free_n - 0.5
        ax.axhline(sep_y, color="#e0e0e0", linewidth=0.8, zorder=2)

    best = max(recalls)
    for bar, recall, cost in zip(bars, recalls, costs):
        mid = bar.get_y() + bar.get_height() / 2
        w = "bold" if recall == best else "normal"
        # Recall value
        ax.text(recall + 0.002, mid, f"{recall:.1%}", va="center",
                fontsize=9, fontweight=w, color="#333")
        # Cost badge right-aligned inside plot area
        cstr = "Free" if cost == 0 else f"${cost:.2f}/1K"
        cc = "#27AE60" if cost == 0 else "#C0392B"
        ax.text(best + 0.035, mid, cstr, va="center", ha="right",
                fontsize=8, fontweight="bold", color=cc)

    ax.set_yticks(y)
    ax.set_yticklabels(ranked, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("Recall@K", fontsize=10, color="#666")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0.62, best + 0.04)
    ax.set_title("Retrieval Quality vs. Cost", fontsize=13, fontweight="bold",
                 pad=14, color="#222")

    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    _brand(fig)
    p = output_dir / "chart1_quality_vs_cost.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart 1 saved: {p}")


# ── Chart 2: Hybrid Mix ──────────────────────────────────────────────────────


def chart_hybrid_mix(data, output_dir):
    _rc()
    sweep = data.get("weight_sweep", [])
    if not sweep:
        print("  Chart 2 skipped")
        return

    hyb = [p for p in sweep if p["kw"] is not None]
    llm = [p for p in sweep if p["kw"] is None]

    x = [p["sw"] for p in hyb]
    y = [p["mean_recall"] for p in hyb]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.fill_between(x, y, alpha=0.05, color=C_HYB)
    ax.plot(x, y, "o-", color=C_HYB, lw=2, ms=5, zorder=4,
            markeredgecolor="white", markeredgewidth=1)

    bi = max(range(len(y)), key=lambda i: y[i])
    ax.annotate(f"Best: {x[bi]:.0%} semantic\n{y[bi]:.1%} recall",
                xy=(x[bi], y[bi]),
                xytext=(x[bi] - 0.2, y[bi] + 0.025),
                fontsize=8, fontweight="bold", color=C_HYB,
                arrowprops=dict(arrowstyle="->", color=C_HYB, lw=1.2))

    if llm:
        lr = [p["mean_recall"] for p in llm]
        lo, hi = min(lr), max(lr)
        ax.axhspan(lo - 0.001, hi + 0.001, alpha=0.07, color=C_LLM, zorder=1)
        ax.axhline(np.mean(lr), color=C_LLM, ls="--", lw=1.2, alpha=0.5, zorder=2)
        ax.text(0.5, hi + 0.005,
                f"LLM Rerankers ({lo:.1%}–{hi:.1%}, $1–$11/1K)",
                fontsize=7.5, color=C_LLM, va="bottom", fontweight="bold")

    # Endpoint labels inside the axes
    ax.text(0.01, y[0] + 0.008, "Pure BM25", fontsize=7.5,
            color=C_BM25, fontweight="bold", va="bottom")
    ax.text(0.99, y[-1] + 0.008, "Pure Semantic", fontsize=7.5,
            color=C_SEM, fontweight="bold", va="bottom", ha="right")

    ax.set_xlabel("Semantic Weight  (BM25 = 1 − semantic)", fontsize=10, color="#666")
    ax.set_ylabel("Recall@K", fontsize=10, color="#666")
    ax.set_title("Hybrid Search: Finding the Optimal Mix",
                 fontsize=13, fontweight="bold", pad=14, color="#222")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(-0.03, 1.03)
    # Tighter y range — cut the dead space below
    ax.set_ylim(min(y) - 0.02, max(y) + 0.045)

    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    _brand(fig)
    p = output_dir / "chart2_hybrid_mix.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart 2 saved: {p}")


# ── Chart 3: Summary Table ───────────────────────────────────────────────────


def chart_summary_table(data, output_dir):
    _rc()
    apps = data["approaches"]
    agg = data["aggregates"]
    na = len(apps)

    metrics = [
        ("Recall@K",  "How many right answers found?"),
        ("MRR",       "First useful result rank?"),
        ("NDCG@K",    "Best results near the top?"),
        ("pass_rate", "Queries above 40% threshold"),
    ]

    cols = [""] + apps
    rows, rcolors = [], []

    for idx, (mk, md) in enumerate(metrics):
        vals = [agg[a][mk] for a in apps]
        best = max(vals)
        base = "#f5f5f5" if idx % 2 == 0 else "white"
        row = [md]
        rc = [base]
        for v in vals:
            row.append(f"{v:.1%}")
            rc.append("#e8f5e9" if v == best else base)
        rows.append(row)
        rcolors.append(rc)

    # Cost row
    cr, cc = ["Cost / 1K queries"], ["#f5f5f5"]
    costs = [_cost(a) for a in apps]
    for c in costs:
        cr.append(f"${c:.2f}" if c > 0 else "Free")
        cc.append("#e8f5e9" if c == 0 else "#f5f5f5")
    rows.append(cr)
    rcolors.append(cc)

    fw = max(10, 3 + na * 1.5)
    fig, ax = plt.subplots(figsize=(fw, 3.2))
    ax.axis("off")
    ax.set_title(f"Benchmark Summary — {data['n_queries']} queries, {na} approaches",
                 fontsize=12, fontweight="bold", pad=12, color="#222")

    tbl = ax.table(cellText=rows, colLabels=cols, cellColours=rcolors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    fs = 8 if na <= 5 else 6.8 if na <= 8 else 6
    tbl.set_fontsize(fs)
    tbl.scale(1.0, 1.8)

    dw = 0.17
    aw = (1.0 - dw) / na
    for r in range(len(rows) + 1):
        tbl[r, 0].set_width(dw)
        for j in range(1, na + 1):
            tbl[r, j].set_width(aw)

    for j in range(na + 1):
        tbl[0, j].set_facecolor(MIT_CRIMSON)
        tbl[0, j].set_text_props(color="white", fontweight="bold", fontsize=fs)

    for i in range(1, len(rows) + 1):
        tbl[i, 0].get_text().set_ha("left")
        tbl[i, 0].get_text().set_fontsize(max(5.5, fs - 0.5))
        tbl[i, 0].get_text().set_color("#888")

    for i, (mk, _) in enumerate(metrics):
        vals = [agg[a][mk] for a in apps]
        best = max(vals)
        for j, v in enumerate(vals):
            if v == best:
                tbl[i + 1, j + 1].set_text_props(fontweight="bold", color="#1b5e20")

    for j, c in enumerate(costs):
        if c == 0:
            tbl[len(metrics) + 1, j + 1].set_text_props(fontweight="bold", color="#1b5e20")

    fig.tight_layout(rect=[0, 0.02, 1, 0.955])
    _brand(fig)
    p = output_dir / "chart3_summary_table.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart 3 saved: {p}")


# ── Chart 4: Where Each Approach Wins ─────────────────────────────────────────


def chart_approach_strengths(data, output_dir):
    _rc()
    per_level = data["per_level"]
    approaches = data["approaches"]

    # Query types with descriptions explaining what each tests
    query_types = [
        ("Keyword Lookup",          "Exact terms and specific phrases",          ["L1", "L3"]),
        ("Semantic Understanding",  "Meaning-based queries, no exact match",     ["L2", "L4"]),
        ("Cross-Source Reasoning",  "Combining info across multiple documents",  ["L5", "L7"]),
        ("Paraphrased Queries",     "Same question, different wording",          ["L6"]),
        ("Temporal Comparison",     "Time-based and version comparisons",        ["L9", "L12"]),
        ("Complex Reasoning",       "Multi-hop logic and inference chains",      ["L10", "L11"]),
    ]

    families = {
        "Hybrid":     (["Hybrid", "Hybrid+Expand", "Hybrid+RRF"], C_HYB, "o"),
        "BM25":       (["BM25"], C_BM25, "s"),
        "Semantic":   (["Semantic"], C_SEM, "D"),
        "LLM Rerank": ([a for a in approaches if a.startswith("LLM:")], C_LLM, "^"),
    }

    scores = {}
    for label, _desc, levels in query_types:
        present = [lv for lv in levels if lv in per_level]
        if not present:
            continue
        tn = sum(per_level[lv]["n"] for lv in present)
        scores[label] = {}
        for fn, (variants, _, _) in families.items():
            best = 0
            for v in variants:
                if v not in approaches:
                    continue
                w = sum(per_level[lv]["approaches"].get(v, {}).get("mean_recall", 0)
                        * per_level[lv]["n"] for lv in present)
                best = max(best, w / tn if tn else 0)
            scores[label][fn] = best

    labels = [t[0] for t in query_types if t[0] in scores]
    descs = {t[0]: t[1] for t in query_types}
    n = len(labels)

    fig, ax = plt.subplots(figsize=(8, 5.8))
    yp = np.arange(n)

    # Range lines
    for i, lb in enumerate(labels):
        sv = list(scores[lb].values())
        ax.plot([min(sv), max(sv)], [i, i],
                color="#e0e0e0", lw=3, zorder=1, solid_capstyle="round")

    # Dots
    for fn, (_, color, marker) in families.items():
        xv = [scores[lb][fn] for lb in labels]
        ax.scatter(xv, yp, s=130, c=color, marker=marker, label=fn,
                   zorder=5, edgecolors="white", linewidth=1.5, alpha=0.9)
        # Label the winner in each row
        for i, (x, lb) in enumerate(zip(xv, labels)):
            if x == max(scores[lb].values()):
                ax.annotate(f"{x:.0%}", (x, i), fontsize=7.5,
                            fontweight="bold", color=color,
                            xytext=(0, 10), textcoords="offset points",
                            ha="center", va="bottom")

    # Two-line y-labels: bold name + light description
    ax.set_yticks(yp)
    ax.set_yticklabels(labels, fontsize=9, fontweight="bold")
    # Add description text below each label
    for i, lb in enumerate(labels):
        ax.annotate(descs[lb], xy=(0, i), xycoords=("axes fraction", "data"),
                    xytext=(-10, -12), textcoords="offset points",
                    fontsize=7, color="#999", ha="right", va="top")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Recall@K", fontsize=10, color="#666")
    ax.set_title("Where Each Approach Wins", fontsize=13,
                 fontweight="bold", pad=14, color="#222")
    ax.set_xlim(0.28, 1.02)
    ax.invert_yaxis()

    # Legend at top-right, out of the way of data
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8,
              ncol=2, columnspacing=1, edgecolor="#ddd",
              handletextpad=0.4)

    fig.tight_layout(rect=[0, 0.02, 1, 0.975])
    _brand(fig)
    p = output_dir / "chart4_approach_strengths.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Chart 4 saved: {p}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("benchmark_results/results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Error: {args.results} not found.")
        sys.exit(1)

    data = json.loads(args.results.read_text())
    print(f"Loaded {data['n_queries']} queries x {len(data['approaches'])} approaches")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("\nGenerating charts...")
    chart_quality_vs_cost(data, args.output_dir)
    chart_hybrid_mix(data, args.output_dir)
    chart_summary_table(data, args.output_dir)
    chart_approach_strengths(data, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
