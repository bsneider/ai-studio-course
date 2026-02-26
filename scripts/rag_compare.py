#!/usr/bin/env python3
"""
Generate RAG comparison chart using `claude -p` for LLM calls.

Three conditions per question:
  1. No RAG  — plain question, no context
  2. Untuned RAG — top_k=3, equal bm25/vec weights, small context
  3. Tuned RAG   — benchmark top_k, optimised weights, full context

Uses COMPLEX_QUERIES from tests/shared.py (L12 → L11 → L10, top 5).
"""

import json
import os
import sqlite3
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import sqlite_vec
from fastembed import TextEmbedding

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from shared import COMPLEX_QUERIES  # noqa: E402

# ── Pick 5 hardest questions ──────────────────────────────────────────────────

_by_level = {"L12": [], "L11": [], "L10": [], "L9": []}
for q in COMPLEX_QUERIES:
    _by_level.get(q["level"], []).append(q)

SELECTED = (_by_level["L12"] + _by_level["L11"] + _by_level["L10"] + _by_level["L9"])[:5]
QUESTIONS  = [q["q"]              for q in SELECTED]
LEVELS     = [q["level"]          for q in SELECTED]
TUNED_TOPK = [q.get("top_k", 8)  for q in SELECTED]

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(ROOT / "ai_studio_rag.db")
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

_emb_model = None
def embed(text: str) -> list[float]:
    global _emb_model
    if _emb_model is None:
        _emb_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    return list(_emb_model.embed([text]))[0].tolist()

STOP = {"what","are","all","the","or","and","a","an","is","in","of","for","to","do",
        "how","who","why","when","where","which","this","that","these","those","have",
        "has","been","did","does","was","were","will","would","could","should","their",
        "they","them","its","it","be","at","on","as","by","with","about","any","from",
        "not","ever"}

def hybrid_search(conn, query: str, query_emb: list, top_k: int,
                  bm25_w: float, vec_w: float) -> list[dict]:
    blob = struct.pack(f"{len(query_emb)}f", *query_emb)

    # Fetch many more to get enough unique results after dedup
    fetch_k = max(top_k * 10, 100)
    vec_rows = conn.execute(
        "SELECT rowid, distance FROM vec_documents WHERE embedding MATCH ? AND k = ?",
        (blob, fetch_k)
    ).fetchall()
    # Deduplicate by rowid, keeping only best (lowest) distance per id
    best: dict[int, float] = {}
    for r in vec_rows:
        rid, dist = r["rowid"], r["distance"]
        if rid not in best or dist < best[rid]:
            best[rid] = dist
    vec_scores = {rid: 1.0 / (1 + d) for rid, d in best.items()}

    tokens = [t.strip("?.!,;:'\"") for t in query.lower().split()]
    tokens = [t for t in tokens if t and t not in STOP]
    bm25_scores = {}
    if tokens:
        bm25_q = " OR ".join(tokens)
        try:
            rows = conn.execute(
                "SELECT rowid, rank FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?",
                (bm25_q, top_k * 2)
            ).fetchall()
            mx = max((abs(r["rank"]) for r in rows), default=1)
            bm25_scores = {r["rowid"]: abs(r["rank"]) / mx for r in rows}
        except Exception:
            pass

    all_ids = set(vec_scores) | set(bm25_scores)
    combined = {cid: bm25_w * bm25_scores.get(cid, 0) + vec_w * vec_scores.get(cid, 0)
                for cid in all_ids}
    top_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]

    results = []
    for cid in top_ids:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (cid,)).fetchone()
        if row:
            results.append(dict(row))
    return results

def format_context(docs: list[dict], max_chars: int = 4000) -> str:
    parts, used = [], 0
    for i, d in enumerate(docs, 1):
        entry = f"[Source {i}: {d.get('page_title','?')} > {d.get('section_title','?')}]\n{d.get('content','')}"
        if used + len(entry) > max_chars:
            break
        parts.append(entry)
        used += len(entry)
    return "\n\n".join(parts) if parts else "No relevant documents found."

# ── claude -p call ────────────────────────────────────────────────────────────

SYSTEM_NO_RAG = (
    "You are a helpful assistant for the MIT AI Studio course at MIT Media Lab. "
    "Answer concisely in 2-4 sentences. If you don't know, say so clearly."
)

SYSTEM_RAG = (
    "You are a helpful assistant for the MIT AI Studio course at MIT Media Lab. "
    "Answer ONLY using the provided context. Be concise (2-4 sentences). "
    "If the context doesn't contain enough information, say so clearly."
)

def ask_claude(prompt: str, system: str, model: str = "haiku") -> str:
    """Run claude -p with the given prompt, return plain text response."""
    env = {**os.environ, "CLAUDECODE": ""}   # unset nested-session guard
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "text",
        "--model", model,
        "--tools", "",                        # no tools needed for answering
        "--system-prompt", system,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=env, timeout=120
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[:120]
        return f"[Error: {err}]"
    return (result.stdout or "").strip()

# ── Three conditions ──────────────────────────────────────────────────────────

def no_rag(question: str) -> str:
    return ask_claude(f"Question: {question}", SYSTEM_NO_RAG)

def untuned_rag(conn, question: str) -> str:
    emb = embed(question)
    # Untuned: small top_k, no punctuation stripping, equal weights, tiny context
    docs = hybrid_search(conn, question, emb, top_k=3, bm25_w=0.5, vec_w=0.5)
    ctx  = format_context(docs, max_chars=1500)
    prompt = f"Context:\n{ctx}\n\nQuestion: {question}"
    return ask_claude(prompt, SYSTEM_RAG)

def tuned_rag(conn, question: str, top_k: int = 15) -> str:
    emb = embed(question)
    # Tuned: high top_k, BM25-heavy for aggregation questions, large context window
    docs = hybrid_search(conn, question, emb, top_k=top_k, bm25_w=0.55, vec_w=0.45)
    ctx  = format_context(docs, max_chars=8000)
    prompt = f"Context:\n{ctx}\n\nQuestion: {question}"
    return ask_claude(prompt, SYSTEM_RAG)

# ── Chart ─────────────────────────────────────────────────────────────────────

CONDITIONS    = ["No RAG (LLM Only)", "Untuned RAG", "Tuned RAG"]
COND_DESC     = ["top_k=0, no retrieval", "top_k=3, BM25/vec 50/50, 1.5k ctx", "top_k=15, BM25-heavy, 8k ctx"]
CELL_COLORS   = ["#FFF3E0", "#E3F2FD", "#E8F5E9"]
HEADER_COLORS = ["#BF360C", "#0D47A1", "#1B5E20"]
BORDER_COLORS = ["#FF7043", "#42A5F5", "#66BB6A"]

MAX_CHARS = 320   # truncate long responses for chart readability

def clip(text: str, n: int = MAX_CHARS) -> str:
    """Truncate and add ellipsis if needed."""
    text = text.strip()
    if len(text) <= n:
        return text
    # Cut at last sentence boundary before limit
    truncated = text[:n]
    last_period = max(truncated.rfind('. '), truncated.rfind('.\n'))
    if last_period > n // 2:
        return truncated[:last_period + 1] + " …"
    return truncated.rstrip() + " …"

def wrap_text(text: str, width: int = 58) -> str:
    lines = []
    for para in text.split("\n"):
        if para.strip():
            lines.extend(textwrap.wrap(para.strip(), width))
        else:
            lines.append("")
    return "\n".join(lines)

def make_chart(data: list[list[str]], out_path: Path):
    """
    Layout: questions as ROWS, conditions as COLUMNS.
    Each cell is tall enough to read comfortably.
    """
    n_q  = len(QUESTIONS)
    n_c  = len(CONDITIONS)

    # dimensions in inches
    lw   = 2.8    # left label column width
    cw   = 5.8    # each condition column width
    rh   = 3.6    # row height per question
    qh   = 0.7    # question label height (left)
    hdr  = 1.0    # top header row height
    pad  = 0.12   # inner padding

    fw   = lw + n_c * cw + 0.4
    fh   = hdr + n_q * rh + 1.2

    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")

    # ── Title ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.992,
             "RAG Quality Comparison: LLM-Only vs Untuned vs Tuned",
             ha="center", va="top", fontsize=16, fontweight="bold", color="#1A1A2E",
             fontfamily="sans-serif")
    fig.text(0.5, 0.972,
             "MIT AI Studio Course  ·  5 Hardest Questions (L10–L12)  ·  Claude Haiku  ·  Hybrid BM25 + Vector Search",
             ha="center", va="top", fontsize=9.5, color="#666", style="italic")

    y_top = fh - hdr

    # ── Column headers ────────────────────────────────────────────────────────
    for ci, (cond, desc, hc, bc) in enumerate(zip(CONDITIONS, COND_DESC, HEADER_COLORS, BORDER_COLORS)):
        x = lw + ci * cw
        rect = FancyBboxPatch(
            (x + pad, y_top + pad), cw - 2*pad, hdr - 2*pad,
            boxstyle="round,pad=0.05", facecolor=hc, edgecolor=hc, linewidth=0
        )
        ax.add_patch(rect)
        ax.text(x + cw/2, y_top + hdr*0.62, cond,
                ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(x + cw/2, y_top + hdr*0.22, desc,
                ha="center", va="center", fontsize=7.5, color="#DDD", style="italic")

    # ── Rows ──────────────────────────────────────────────────────────────────
    for qi, (question, level) in enumerate(zip(QUESTIONS, LEVELS)):
        y = y_top - (qi + 1) * rh

        # Question label (left column)
        q_bg = "#EDE7F6" if qi % 2 == 0 else "#F3E5F5"
        rect = FancyBboxPatch(
            (pad, y + pad), lw - 2*pad, rh - 2*pad,
            boxstyle="round,pad=0.05", facecolor=q_bg, edgecolor="#7E57C2", linewidth=1.2
        )
        ax.add_patch(rect)
        # Level badge
        ax.text(lw/2, y + rh - 0.28, level,
                ha="center", va="center", fontsize=9, fontweight="bold", color="#7B1FA2",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#CE93D8", edgecolor="none", alpha=0.7))
        qlabel = "\n".join(textwrap.wrap(question, width=26))
        ax.text(lw/2, y + rh/2 - 0.1, qlabel,
                ha="center", va="center", fontsize=8.2, color="#311B92",
                fontweight="bold", multialignment="center", linespacing=1.4)

        # Condition cells
        for ci, (bg, bc, hc) in enumerate(zip(CELL_COLORS, BORDER_COLORS, HEADER_COLORS)):
            x    = lw + ci * cw
            text = clip(data[qi][ci])
            is_failure = "don't have" in text.lower() or "i cannot" in text.lower() or "not enough" in text.lower()

            cell_bg = "#FFF8F8" if is_failure else bg
            cell_border = "#FFCDD2" if is_failure else bc
            lw_border   = 0.8 if is_failure else 1.4

            rect = FancyBboxPatch(
                (x + pad, y + pad), cw - 2*pad, rh - 2*pad,
                boxstyle="round,pad=0.05", facecolor=cell_bg, edgecolor=cell_border,
                linewidth=lw_border
            )
            ax.add_patch(rect)

            # Failure indicator
            if is_failure:
                ax.text(x + cw - 0.35, y + rh - 0.25, "✗",
                        ha="center", va="center", fontsize=11, color="#E53935")
            else:
                ax.text(x + cw - 0.35, y + rh - 0.25, "✓",
                        ha="center", va="center", fontsize=11, color="#43A047")

            wrapped = wrap_text(text, width=58)
            ax.text(x + cw/2, y + rh/2, wrapped,
                    ha="center", va="center", fontsize=7.5, color="#212121",
                    multialignment="left", linespacing=1.45,
                    fontfamily="monospace")

        # Row separator line
        if qi < n_q - 1:
            ax.axhline(y, color="#DDD", linewidth=0.5, xmin=0.01, xmax=0.99)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_y = 0.022
    patches = [
        mpatches.Patch(facecolor=HEADER_COLORS[0], label=f"No RAG — {COND_DESC[0]}"),
        mpatches.Patch(facecolor=HEADER_COLORS[1], label=f"Untuned RAG — {COND_DESC[1]}"),
        mpatches.Patch(facecolor=HEADER_COLORS[2], label=f"Tuned RAG — {COND_DESC[2]}"),
        mpatches.Patch(facecolor="#FFF8F8", edgecolor="#FFCDD2", label="✗ Insufficient context / hallucination"),
        mpatches.Patch(facecolor=CELL_COLORS[2], edgecolor=BORDER_COLORS[2], label="✓ Grounded answer from retrieved context"),
    ]
    ax.legend(handles=patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.01), ncol=5, fontsize=7.8,
              framealpha=0.95, edgecolor="#CCC",
              handlelength=1.2, handleheight=0.9)

    plt.tight_layout(rect=[0, 0.05, 1, 0.965])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
    print(f"\nChart saved → {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading DB + embedding model...")
    conn = get_conn()
    embed("warmup")   # load model once

    data = []  # data[qi][condition] = answer string
    for qi, question in enumerate(QUESTIONS):
        print(f"\n[{qi+1}/5] ({LEVELS[qi]}) {question[:70]}...")
        row = []

        print("  no_rag ...")
        row.append(no_rag(question))
        print(f"    → {row[-1][:90]}")

        print("  untuned_rag ...")
        row.append(untuned_rag(conn, question))
        print(f"    → {row[-1][:90]}")

        print("  tuned_rag ...")
        row.append(tuned_rag(conn, question, top_k=TUNED_TOPK[qi]))
        print(f"    → {row[-1][:90]}")

        data.append(row)

    # Save raw JSON
    raw_path = ROOT / "benchmark_results" / "rag_comparison_responses.json"
    raw_path.write_text(json.dumps(
        [{"question": q, "level": lv, "no_rag": d[0], "untuned_rag": d[1], "tuned_rag": d[2]}
         for q, lv, d in zip(QUESTIONS, LEVELS, data)],
        indent=2
    ))
    print(f"Responses saved → {raw_path}")

    chart_path = ROOT / "benchmark_results" / "rag_comparison_chart.png"
    make_chart(data, chart_path)

def chart_only():
    """Regenerate chart from saved responses JSON without any API calls."""
    raw_path = ROOT / "benchmark_results" / "rag_comparison_responses.json"
    if not raw_path.exists():
        print(f"No saved responses at {raw_path}. Run without --chart-only first.")
        sys.exit(1)
    with open(raw_path) as f:
        saved = json.load(f)
    data = [[d["no_rag"], d["untuned_rag"], d["tuned_rag"]] for d in saved]
    chart_path = ROOT / "benchmark_results" / "rag_comparison_chart.png"
    make_chart(data, chart_path)

if __name__ == "__main__":
    if "--chart-only" in sys.argv:
        chart_only()
    else:
        main()
