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

C_IMP = "#2ECC71"
C_PI = "#9B59B6"

COLORS = {
    "Hybrid": C_HYB, "Hybrid+Expand": "#E74C3C", "Hybrid+RRF": "#922B21",
    "BM25": C_BM25, "Semantic": C_SEM,
    "LLM:improved": C_IMP, "PageIndex": C_PI,
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


COST = {"gemini": 1.15, "nemotron": 11.10, "llama": 1.09, "improved": 1.15, "pageindex": 1.15}
DPI = 192
FOOTER = "Research: Brandon Sneider | MIT AI Studio (MAS.664/665)"

# Google Slides optimized (16:9)
SLIDE_W = 13.33
SLIDE_H = 7.5


def _cost(name):
    nl = name.lower()
    if name == "PageIndex":
        return COST["pageindex"]
    if not name.startswith("LLM:"):
        return 0.0
    for k, v in COST.items():
        if k in nl:
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


APPROACH_DESC = {
    "Hybrid":        "BM25 0.3 + Semantic 0.7, content-type diversity",
    "Hybrid+Expand": "Hybrid + synonym expansion for BM25 queries",
    "Hybrid+RRF":    "Hybrid using Reciprocal Rank Fusion (k=60)",
    "BM25":          "FTS5 keyword search only (no embeddings)",
    "Semantic":      "Vector similarity only (no keyword signal)",
}


def chart_quality_vs_cost(data, output_dir):
    _rc()
    agg = data["aggregates"]
    apps = data["approaches"]
    ranked = sorted(apps, key=lambda a: agg[a]["Recall@K"])

    recalls = [agg[a]["Recall@K"] for a in ranked]
    colors = [_c(a) for a in ranked]

    fig, ax = plt.subplots(figsize=(SLIDE_W, SLIDE_H))
    y = np.arange(len(ranked))

    bars = ax.barh(y, recalls, color=colors, alpha=0.82, height=0.55,
                   zorder=3, edgecolor="white", linewidth=0.8)

    best = max(recalls)
    for bar, recall, name in zip(bars, recalls, ranked):
        mid = bar.get_y() + bar.get_height() / 2
        w = "bold" if recall == best else "normal"
        ax.text(recall + 0.003, mid, f"{recall:.1%}", va="center",
                fontsize=14, fontweight=w, color="#333")
        # Approach description inside the bar
        desc = APPROACH_DESC.get(name, "")
        if desc:
            ax.text(0.625, mid, desc, va="center", fontsize=9,
                    color="white", alpha=0.9, fontstyle="italic")

    ax.set_yticks(y)
    ax.set_yticklabels(ranked, fontsize=14, fontweight="bold")
    ax.set_xlabel("Recall@K", fontsize=14, color="#666", labelpad=8)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlim(0.62, best + 0.05)
    ax.set_title("Retrieval Quality — All Local, Zero API Cost",
                 fontsize=18, fontweight="bold", pad=18, color="#222")

    fig.tight_layout(rect=[0, 0.06, 1, 0.975])
    fig.text(0.5, 0.025,
             "Embedding: all-MiniLM-L6-v2  |  Database: SQLite + FTS5 + sqlite-vec  |  35 queries, 381 documents",
             ha="center", fontsize=10, color="#999")
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

    fig, ax = plt.subplots(figsize=(SLIDE_W, SLIDE_H))

    ax.fill_between(x, y, alpha=0.05, color=C_HYB)
    ax.plot(x, y, "o-", color=C_HYB, lw=3, ms=8, zorder=4,
            markeredgecolor="white", markeredgewidth=1.5)

    bi = max(range(len(y)), key=lambda i: y[i])
    ax.annotate(f"Best: {x[bi]:.0%} semantic\n{y[bi]:.1%} recall",
                xy=(x[bi], y[bi]),
                xytext=(x[bi] - 0.2, y[bi] + 0.025),
                fontsize=12, fontweight="bold", color=C_HYB,
                arrowprops=dict(arrowstyle="->", color=C_HYB, lw=1.5))

    if llm:
        lr = [p["mean_recall"] for p in llm]
        lo, hi = min(lr), max(lr)
        ax.axhspan(lo - 0.001, hi + 0.001, alpha=0.07, color=C_LLM, zorder=1)
        ax.axhline(np.mean(lr), color=C_LLM, ls="--", lw=1.2, alpha=0.5, zorder=2)
        ax.text(0.02, hi + 0.006,
                f"LLM Rerankers ({lo:.1%}\u2013{hi:.1%}, $1\u2013$11/1K)",
                fontsize=10, color=C_LLM, va="bottom", fontweight="bold")

    ax.text(0.01, y[0] - 0.012, "Pure BM25", fontsize=11,
            color=C_BM25, fontweight="bold", va="top")
    ax.text(0.99, y[-1] + 0.012, "Pure Semantic", fontsize=11,
            color=C_SEM, fontweight="bold", va="bottom", ha="right")

    ax.set_xlabel("Semantic Weight  (BM25 = 1 \u2212 semantic)", fontsize=14, color="#666")
    ax.set_ylabel("Recall@K", fontsize=14, color="#666")
    ax.set_title("Hybrid Search: Finding the Optimal Mix",
                 fontsize=18, fontweight="bold", pad=18, color="#222")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(min(y) - 0.025, max(y) + 0.045)

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

    fig, ax = plt.subplots(figsize=(SLIDE_W, SLIDE_H * 0.55))
    ax.axis("off")
    ax.set_title(f"Benchmark Summary \u2014 {data['n_queries']} queries, {na} approaches (all local, zero API cost)",
                 fontsize=16, fontweight="bold", pad=12, color="#222")

    tbl = ax.table(cellText=rows, colLabels=cols, cellColours=rcolors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    fs = 12 if na <= 5 else 10 if na <= 8 else 8
    tbl.set_fontsize(fs)
    tbl.scale(1.0, 2.2)

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
        tbl[i, 0].get_text().set_fontsize(max(8, fs - 1))
        tbl[i, 0].get_text().set_color("#888")

    for i, (mk, _) in enumerate(metrics):
        vals = [agg[a][mk] for a in apps]
        best = max(vals)
        for j, v in enumerate(vals):
            if v == best:
                tbl[i + 1, j + 1].set_text_props(fontweight="bold", color="#1b5e20")

    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
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

    # Query types with descriptions
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
        "PageIndex":  ([a for a in approaches if a == "PageIndex"], C_PI, "P"),
    }
    # Drop families with no matching approaches in the data
    families = {k: v for k, v in families.items() if v[0]}
    family_names = list(families.keys())

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

    # Spacing: 1.5 units between rows for description text
    fig, ax = plt.subplots(figsize=(SLIDE_W, SLIDE_H))
    yp = np.arange(n) * 1.5

    # Range lines (behind dots)
    for i, lb in enumerate(labels):
        sv = list(scores[lb].values())
        ax.plot([min(sv), max(sv)], [yp[i], yp[i]],
                color="#e0e0e0", lw=3, zorder=1, solid_capstyle="round")

    # For each row, compute jitter offsets to prevent marker overlap
    jitter_threshold = 0.025
    offsets = [0.18, -0.18, 0.35, -0.35]  # alternating up/down nudges

    for row_i, lb in enumerate(labels):
        row_vals = sorted(
            [(fn, scores[lb][fn]) for fn in family_names],
            key=lambda t: t[1]
        )

        # Assign y-offsets: cluster dots that are within threshold
        placed = {}  # fn -> (x, y_offset)
        for fn, xval in row_vals:
            # Find how many previously placed dots are close
            close_offsets = [
                placed[pfn][1] for pfn, px in row_vals
                if pfn in placed and abs(px - xval) < jitter_threshold
            ]
            if not close_offsets:
                yo = 0.0
            else:
                # Pick the first unused offset
                for candidate in offsets:
                    if candidate not in close_offsets:
                        yo = candidate
                        break
                else:
                    yo = offsets[len(close_offsets) % len(offsets)]
            placed[fn] = (xval, yo)

        # Plot dots — draw in consistent order, largest markers first (lowest zorder)
        best_val = max(scores[lb].values())
        labeled_x = []  # track labeled x positions to avoid overlaps
        for fn in family_names:
            _, color, marker = families[fn]
            xval, yo = placed[fn]
            yval = yp[row_i] + yo
            ax.scatter([xval], [yval], s=100, c=color, marker=marker,
                       zorder=6, edgecolors="white", linewidth=1.2, alpha=0.92,
                       label=fn if row_i == 0 else None)

            # Label the winner; skip if another label is already close
            if xval == best_val:
                too_close = any(abs(xval - lx) < 0.03 for lx in labeled_x)
                if not too_close:
                    ax.annotate(f"{xval:.0%}", (xval, yval), fontsize=11,
                                fontweight="bold", color=color,
                                xytext=(0, 12), textcoords="offset points",
                                ha="center", va="bottom")
                    labeled_x.append(xval)

    # Y-axis labels
    ax.set_yticks(yp)
    ax.set_yticklabels(labels, fontsize=13, fontweight="bold")
    # Description subtitle below each label
    for i, lb in enumerate(labels):
        ax.annotate(descs[lb], xy=(0, yp[i]),
                    xycoords=("axes fraction", "data"),
                    xytext=(-10, -16), textcoords="offset points",
                    fontsize=9, color="#aaa", ha="right", va="top")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Recall@K", fontsize=14, color="#666")
    ax.tick_params(axis='x', labelsize=12)
    ax.set_title("Where Each Approach Wins", fontsize=18,
                 fontweight="bold", pad=18, color="#222")
    ax.set_xlim(0.28, 1.02)
    ax.invert_yaxis()

    # Legend below the chart, fully outside plot area
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07),
              framealpha=0.95, fontsize=12,
              ncol=4, columnspacing=1.5, edgecolor="#ddd",
              handletextpad=0.4)

    fig.tight_layout(rect=[0, 0.06, 1, 0.975])
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

    # Filter out disabled approaches for clean sharing
    _HIDDEN = {"PageIndex"} | {a for a in data.get("approaches", []) if a.startswith("LLM:")}
    if any(a in _HIDDEN for a in data["approaches"]):
        data["approaches"] = [a for a in data["approaches"] if a not in _HIDDEN]
        data["aggregates"] = {k: v for k, v in data["aggregates"].items() if k not in _HIDDEN}
        for level_data in data.get("per_level", {}).values():
            level_data["approaches"] = {k: v for k, v in level_data.get("approaches", {}).items() if k not in _HIDDEN}
        data["weight_sweep"] = [p for p in data.get("weight_sweep", []) if p.get("label") not in _HIDDEN]

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
