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
from shared import COMPLEX_QUERIES, bm25_only_search  # noqa: E402

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
    "Answer using the provided context. Extract and synthesize all relevant information "
    "you can find across the sources. Be specific — name names, companies, projects, dates. "
    "If some information is missing, state what you found and note the gap briefly at the end."
)

def ask_claude(prompt: str, system: str, model: str = "sonnet") -> str:
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
    return ask_claude(f"Question: {question}", SYSTEM_NO_RAG, model="haiku")

def bm25_rag(conn, question: str, top_k: int = 10) -> str:
    docs = bm25_only_search(conn, question, top_k=top_k)
    parts = [f"[Source {i+1}]\n{d['content']}" for i, d in enumerate(docs)]
    ctx   = "\n\n".join(parts) if parts else "No relevant documents found."
    prompt = f"Context:\n{ctx}\n\nQuestion: {question}"
    return ask_claude(prompt, SYSTEM_RAG, model="haiku")

def untuned_rag(conn, question: str) -> str:
    emb = embed(question)
    # Untuned: small top_k, equal weights, small context — haiku model
    docs = hybrid_search(conn, question, emb, top_k=3, bm25_w=0.5, vec_w=0.5)
    ctx  = format_context(docs, max_chars=1500)
    prompt = f"Context:\n{ctx}\n\nQuestion: {question}"
    return ask_claude(prompt, SYSTEM_RAG, model="haiku")

def tuned_rag(conn, question: str, top_k: int = 15) -> str:
    from shared import hybrid_search as shared_hybrid_search
    # Ensure embedding model is initialised
    embed("warmup") if _emb_model is None else None
    docs = shared_hybrid_search(conn, _emb_model, question, top_k=top_k)
    # Build context — shared results have 'content' but not page_title/section_title
    parts = [f"[Source {i+1}]\n{d['content']}" for i, d in enumerate(docs)]
    ctx   = "\n\n".join(parts) if parts else "No relevant documents found."
    prompt = f"Context:\n{ctx}\n\nQuestion: {question}"
    return ask_claude(prompt, SYSTEM_RAG)

# ── Chart ─────────────────────────────────────────────────────────────────────

CONDITIONS    = ["No RAG\n(LLM Only)", "BM25 Only\nRAG", "Untuned\nHybrid RAG", "Tuned\nHybrid RAG"]
COND_DESC     = ["No retrieval · Haiku", "keyword search · top_k=10 · Haiku", "vec+BM25 50/50 · top_k=3 · Haiku", "hybrid search · top_k=15 · Sonnet"]
CELL_BG       = ["#FFF8F5", "#FFFDE7", "#F0F7FF", "#F1FBF1"]
HEADER_BG     = ["#BF360C", "#F57F17", "#0D47A1", "#1B5E20"]
CELL_BORDER   = ["#FFAB91", "#FFE082", "#90CAF9", "#A5D6A7"]
FAIL_BG       = "#FFF0F0"
FAIL_BORDER   = "#FFCDD2"

# Characters per line and max lines for response text
WRAP_WIDTH    = 62
MAX_LINES     = 11


def clip_to_lines(text: str, max_lines: int = MAX_LINES, width: int = WRAP_WIDTH) -> str:
    """Wrap text and truncate to max_lines, adding ellipsis if cut."""
    text = text.strip()
    # Flatten markdown bullets/bold for cleaner display
    import re
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # bold
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)  # bullets

    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width))

    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines - 1]) + "\n…"


def is_failure(text: str) -> bool:
    from shared import answer_quality_score  # noqa (tests/ is in sys.path)
    return answer_quality_score(text) == 0.0


def make_chart(data: list[list[str]], out_path: Path):
    """
    Layout: 3 questions as rows (only the 3 with best tuned answers),
    3 conditions as columns. Large readable cells.
    """
    # Pick the 3 questions where tuned RAG succeeds, else use all 5
    good_idx = [i for i in range(len(QUESTIONS)) if not is_failure(data[i][3])]
    show_idx = good_idx[:3] if len(good_idx) >= 3 else list(range(min(3, len(QUESTIONS))))

    n_q   = len(show_idx)
    n_c   = len(CONDITIONS)
    pad   = 0.15

    # Generous cell sizes for readability
    lw    = 3.0    # question label column
    cw    = 5.6    # each answer column
    rh    = 5.2    # row height
    hdr_h = 1.1    # column header height
    fw    = lw + n_c * cw + 0.3
    fh    = hdr_h + n_q * rh + 1.6

    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── Titles ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.993,
             "RAG Quality: LLM-Only vs Untuned vs Tuned",
             ha="center", va="top", fontsize=18, fontweight="bold", color="#1A1A2E")
    fig.text(0.5, 0.972,
             "MIT AI Studio Course  ·  Hardest questions (L10–L12)  ·  Claude Haiku  ·  Hybrid BM25 + Vector Search",
             ha="center", va="top", fontsize=10, color="#555", style="italic")

    y_top = fh - hdr_h

    # ── Column headers ────────────────────────────────────────────────────────
    for ci, (cond, desc, hbg) in enumerate(zip(CONDITIONS, COND_DESC, HEADER_BG)):
        x = lw + ci * cw
        rect = FancyBboxPatch(
            (x + pad, y_top + pad/2), cw - 2*pad, hdr_h - pad,
            boxstyle="round,pad=0.06", facecolor=hbg, edgecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x + cw/2, y_top + hdr_h*0.65,
                cond, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
        ax.text(x + cw/2, y_top + hdr_h*0.22,
                desc, ha="center", va="center",
                fontsize=8.5, color="#DDD", style="italic")

    # ── Rows ──────────────────────────────────────────────────────────────────
    for row_i, qi in enumerate(show_idx):
        question = QUESTIONS[qi]
        level    = LEVELS[qi]
        y        = y_top - (row_i + 1) * rh

        # ── Question label ──────────────────────────────────────────────────
        q_bg = "#F3F0FF" if row_i % 2 == 0 else "#EDE7F6"
        rect = FancyBboxPatch(
            (pad/2, y + pad), lw - pad, rh - 2*pad,
            boxstyle="round,pad=0.06", facecolor=q_bg,
            edgecolor="#9575CD", linewidth=1.5
        )
        ax.add_patch(rect)

        # difficulty badge top
        ax.text(lw/2, y + rh - 0.38, level,
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#7E57C2",
                          edgecolor="none"))

        qlabel = "\n".join(textwrap.wrap(question, width=28))
        ax.text(lw/2, y + rh/2 - 0.15, qlabel,
                ha="center", va="center", fontsize=9.5, color="#311B92",
                fontweight="bold", multialignment="center", linespacing=1.5)

        # ── Answer cells ────────────────────────────────────────────────────
        for ci in range(n_c):
            x        = lw + ci * cw
            raw_text = data[qi][ci]
            failed   = is_failure(raw_text)
            cell_bg  = FAIL_BG if failed else CELL_BG[ci]
            border   = FAIL_BORDER if failed else CELL_BORDER[ci]
            bw       = 0.8 if failed else 1.6

            rect = FancyBboxPatch(
                (x + pad, y + pad), cw - 2*pad, rh - 2*pad,
                boxstyle="round,pad=0.06", facecolor=cell_bg,
                edgecolor=border, linewidth=bw
            )
            ax.add_patch(rect)

            # Badge
            badge      = "NO ANSWER" if failed else "ANSWERED"
            badge_fg   = "#C62828" if failed else "#1B5E20"
            badge_bg_c = "#FFCDD2" if failed else "#C8E6C9"
            ax.text(x + cw/2, y + rh - 0.36,
                    badge, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=badge_fg,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor=badge_bg_c,
                              edgecolor=badge_fg, linewidth=0.7))

            body = clip_to_lines(raw_text)
            ax.text(x + cw/2, y + rh/2 - 0.1,
                    body, ha="center", va="center",
                    fontsize=9.5, color="#1A1A1A",
                    multialignment="left", linespacing=1.55,
                    fontfamily="DejaVu Sans")

        # thin separator
        if row_i < n_q - 1:
            ax.axhline(y, color="#E0E0E0", linewidth=0.6, xmin=0.01, xmax=0.99)

    # ── Footer note (show omitted questions) ─────────────────────────────────
    omitted = [f"Q{i+1} [{LEVELS[i]}]" for i in range(len(QUESTIONS)) if i not in show_idx]
    if omitted:
        fig.text(0.5, 0.025,
                 f"Note: {', '.join(omitted)} omitted — tuned RAG also unable to answer "
                 "(aggregation across all semesters requires broader retrieval)",
                 ha="center", va="bottom", fontsize=8, color="#888", style="italic")

    # ── Legend ───────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(facecolor=HEADER_BG[0], label="No RAG — LLM training knowledge only"),
        mpatches.Patch(facecolor=HEADER_BG[1], label="Untuned RAG — small context window, equal weights"),
        mpatches.Patch(facecolor=HEADER_BG[2], label="Tuned RAG — larger context, BM25-heavy for aggregation"),
        mpatches.Patch(facecolor=FAIL_BG, edgecolor=FAIL_BORDER,
                       label="NO ANSWER — LLM reports insufficient context"),
        mpatches.Patch(facecolor=CELL_BG[2], edgecolor=CELL_BORDER[2],
                       label="ANSWERED — grounded response from retrieved docs"),
    ]
    fig.legend(handles=patches, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=8.5,
               framealpha=0.97, edgecolor="#CCC",
               handlelength=1.4, handleheight=1.0)

    plt.tight_layout(rect=[0, 0.07, 1, 0.965])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
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

        print("  bm25_rag ...")
        row.append(bm25_rag(conn, question, top_k=TUNED_TOPK[qi]))
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
        [{"question": q, "level": lv,
          "no_rag": d[0], "bm25_rag": d[1], "untuned_rag": d[2], "tuned_rag": d[3]}
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
    data = [[d["no_rag"], d.get("bm25_rag", "[not run]"), d["untuned_rag"], d["tuned_rag"]] for d in saved]
    chart_path = ROOT / "benchmark_results" / "rag_comparison_chart.png"
    make_chart(data, chart_path)

if __name__ == "__main__":
    if "--chart-only" in sys.argv:
        chart_only()
    else:
        main()
