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

# ── Slide Charts (Google Slides optimized, 16:9) ────────────────────────────

import re

CONDITIONS    = ["No RAG (LLM Only)", "BM25 Keyword RAG", "Tuned Hybrid RAG"]
COND_DESC     = ["No retrieval · Haiku", "FTS5 keyword · top_k=10 · Haiku", "BM25 0.3 + Vec 0.7 · top_k=15 · Sonnet"]
HEADER_BG     = ["#BF360C", "#F57F17", "#1B5E20"]
CELL_BG       = ["#FFF8F5", "#FFFDE7", "#F1FBF1"]
CELL_BORDER   = ["#FFAB91", "#FFE082", "#A5D6A7"]
FAIL_BG       = "#FFF0F0"
FAIL_BORDER   = "#FFCDD2"

# Google Slides is 10" x 5.625" at 16:9
SLIDE_W = 13.33  # inches (wider for readability at 1920px)
SLIDE_H = 7.5


def clean_markdown(text: str) -> str:
    """Strip markdown formatting for plain-text display."""
    text = text.strip()
    text = re.sub(r"#{1,4}\s+", "", text)                        # headings
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                 # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)                     # italic
    text = re.sub(r"^\s*[-*]\s+", "- ", text, flags=re.MULTILINE) # bullets
    text = re.sub(r"^\s*\d+\.\s+", "- ", text, flags=re.MULTILINE) # numbered lists
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)         # links
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)       # hr
    text = re.sub(r"> ", "", text)                                # blockquotes
    text = re.sub(r"\|[^\n]+\|", "", text)                       # tables
    text = re.sub(r"\n{3,}", "\n\n", text)                       # collapse blanks
    return text.strip()


def wrap_to_lines(text: str, width: int, max_lines: int) -> str:
    """Wrap text to width, truncate to max_lines."""
    text = clean_markdown(text)
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width))
    # Remove trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines - 1]) + "\n..."


def is_failure(text: str) -> bool:
    from shared import answer_quality_score
    return answer_quality_score(text) == 0.0


def make_question_slide(qi: int, question: str, level: str,
                        responses: list[str], out_path: Path):
    """Generate one 16:9 slide PNG for a single question with 3 response columns."""
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor("white")

    # ── Title bar ─────────────────────────────────────────────────────────
    title_ax = fig.add_axes([0, 0.88, 1, 0.12])
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis("off")
    title_ax.add_patch(FancyBboxPatch(
        (0.02, 0.1), 0.96, 0.85,
        boxstyle="round,pad=0.02", facecolor="#1A1A2E", edgecolor="none"))

    # Question number + difficulty badge
    title_ax.text(0.03, 0.52, f"  Q{qi+1}", fontsize=22, fontweight="bold",
                  color="white", va="center",
                  fontfamily="monospace")
    title_ax.text(0.08, 0.52, f"  [{level}]", fontsize=16, fontweight="bold",
                  color="#FFD54F", va="center")

    # Question text
    q_wrapped = "\n".join(textwrap.wrap(question, width=80))
    title_ax.text(0.15, 0.52, q_wrapped, fontsize=16, fontweight="bold",
                  color="white", va="center")

    # ── Three response columns ────────────────────────────────────────────
    col_width = 0.3
    col_gap = 0.025
    left_margin = 0.025

    for ci in range(3):
        failed = is_failure(responses[ci])
        x0 = left_margin + ci * (col_width + col_gap)

        # Column header
        hdr_ax = fig.add_axes([x0, 0.80, col_width, 0.07])
        hdr_ax.set_xlim(0, 1)
        hdr_ax.set_ylim(0, 1)
        hdr_ax.axis("off")
        hdr_ax.add_patch(FancyBboxPatch(
            (0.02, 0.05), 0.96, 0.9,
            boxstyle="round,pad=0.03", facecolor=HEADER_BG[ci], edgecolor="none"))
        hdr_ax.text(0.5, 0.62, CONDITIONS[ci], fontsize=13, fontweight="bold",
                    color="white", ha="center", va="center")
        hdr_ax.text(0.5, 0.22, COND_DESC[ci], fontsize=8.5, color="#DDD",
                    ha="center", va="center", style="italic")

        # Response body
        body_ax = fig.add_axes([x0, 0.02, col_width, 0.77])
        body_ax.set_xlim(0, 1)
        body_ax.set_ylim(0, 1)
        body_ax.axis("off")

        cell_bg = FAIL_BG if failed else CELL_BG[ci]
        border = FAIL_BORDER if failed else CELL_BORDER[ci]
        body_ax.add_patch(FancyBboxPatch(
            (0.02, 0.01), 0.96, 0.98,
            boxstyle="round,pad=0.02", facecolor=cell_bg,
            edgecolor=border, linewidth=2))

        # Badge
        badge = "NO ANSWER" if failed else "ANSWERED"
        badge_fg = "#C62828" if failed else "#1B5E20"
        badge_bg = "#FFCDD2" if failed else "#C8E6C9"
        body_ax.text(0.5, 0.96, badge, fontsize=9, fontweight="bold",
                     color=badge_fg, ha="center", va="top",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_bg,
                               edgecolor=badge_fg, linewidth=0.7))

        # Response text — auto-size to fit
        body = wrap_to_lines(responses[ci], width=48, max_lines=50)
        body_ax.text(0.06, 0.91, body, fontsize=7.5, color="#1A1A1A",
                     va="top", ha="left", linespacing=1.25,
                     fontfamily="DejaVu Sans")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=192, facecolor="white")
    plt.close(fig)
    print(f"  Slide saved -> {out_path}")


def make_definitions_slide(out_path: Path):
    """Generate a definitions slide explaining the three retrieval approaches."""
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor("white")

    # Title
    title_ax = fig.add_axes([0, 0.88, 1, 0.12])
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)
    title_ax.axis("off")
    title_ax.add_patch(FancyBboxPatch(
        (0.02, 0.1), 0.96, 0.85,
        boxstyle="round,pad=0.02", facecolor="#1A1A2E", edgecolor="none"))
    title_ax.text(0.5, 0.52, "RAG Retrieval Approaches: Definitions",
                  fontsize=22, fontweight="bold", color="white",
                  ha="center", va="center")

    # Three definition columns
    definitions = [
        {
            "title": "No RAG\n(LLM Only)",
            "color": HEADER_BG[0],
            "items": [
                ("What it does", "Sends the question directly to the LLM\nwith no retrieved context."),
                ("Retrieval", "None"),
                ("Model", "Claude Haiku (fast, cheap)"),
                ("Context window", "Zero documents"),
                ("Strengths", "Fast, zero infrastructure needed"),
                ("Weaknesses", "Relies entirely on training data.\nCannot answer questions about\nspecific course content, names,\ndates, or events."),
            ]
        },
        {
            "title": "BM25 Keyword RAG",
            "color": HEADER_BG[1],
            "items": [
                ("What it does", "Uses keyword matching (BM25/TF-IDF)\nto find documents, then feeds them\nto the LLM as context."),
                ("Retrieval", "BM25 full-text search (SQLite FTS5)"),
                ("Model", "Claude Haiku (fast, cheap)"),
                ("Context window", "top_k=10 documents"),
                ("Strengths", "Great for exact name/term matches.\nNo embedding model needed.\nVery fast retrieval."),
                ("Weaknesses", "Misses semantic similarity.\n\"Who teaches the course?\" won't\nmatch \"Professor Raskar leads\nthe class\" without shared keywords."),
            ]
        },
        {
            "title": "Tuned Hybrid RAG",
            "color": HEADER_BG[2],
            "items": [
                ("What it does", "Combines BM25 keyword search with\nsemantic vector search, using\nbenchmark-optimized weights."),
                ("Retrieval", "Hybrid: BM25 (weight 0.3)\n+ Semantic vectors (weight 0.7)"),
                ("Model", "Claude Sonnet (more capable)"),
                ("Context window", "top_k=15 documents, 4000 chars"),
                ("Strengths", "Catches both exact matches AND\nmeaning-based matches. Optimized\nweights from 35-query benchmark.\nContent-type diversity enforcement."),
                ("Weaknesses", "Requires embedding model + vector\nindex. Slightly slower retrieval.\nMore expensive LLM (Sonnet)."),
            ]
        },
    ]

    col_width = 0.3
    col_gap = 0.025
    left_margin = 0.025

    for ci, defn in enumerate(definitions):
        x0 = left_margin + ci * (col_width + col_gap)

        # Header
        hdr_ax = fig.add_axes([x0, 0.80, col_width, 0.07])
        hdr_ax.set_xlim(0, 1)
        hdr_ax.set_ylim(0, 1)
        hdr_ax.axis("off")
        hdr_ax.add_patch(FancyBboxPatch(
            (0.02, 0.05), 0.96, 0.9,
            boxstyle="round,pad=0.03", facecolor=defn["color"], edgecolor="none"))
        hdr_ax.text(0.5, 0.5, defn["title"], fontsize=14, fontweight="bold",
                    color="white", ha="center", va="center")

        # Body
        body_ax = fig.add_axes([x0, 0.02, col_width, 0.77])
        body_ax.set_xlim(0, 1)
        body_ax.set_ylim(0, 1)
        body_ax.axis("off")
        body_ax.add_patch(FancyBboxPatch(
            (0.02, 0.01), 0.96, 0.98,
            boxstyle="round,pad=0.02", facecolor=CELL_BG[ci],
            edgecolor=CELL_BORDER[ci], linewidth=2))

        y = 0.93
        for label, value in defn["items"]:
            body_ax.text(0.06, y, label, fontsize=10, fontweight="bold",
                         color="#333", va="top")
            y -= 0.04
            body_ax.text(0.06, y, value, fontsize=9.5, color="#444",
                         va="top", linespacing=1.3)
            # Count newlines to figure out spacing
            n_lines = value.count("\n") + 1
            y -= 0.04 * n_lines + 0.03

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=192, facecolor="white")
    plt.close(fig)
    print(f"  Definitions slide saved -> {out_path}")


def make_slides(data: list[list[str]], out_dir: Path):
    """Generate one slide PNG per question + a definitions slide."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Slide 0: Definitions
    make_definitions_slide(out_dir / "slide_0_definitions.png")

    # Slides 1-N: One per question
    for qi in range(len(QUESTIONS)):
        responses = data[qi]  # [no_rag, bm25_rag, tuned_rag]
        make_question_slide(
            qi, QUESTIONS[qi], LEVELS[qi], responses,
            out_dir / f"slide_{qi+1}_q{qi+1}_{LEVELS[qi]}.png"
        )

    print(f"\n{len(QUESTIONS)+1} slides saved to {out_dir}/")

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

    # Generate slides (3-column: no_rag, bm25_rag, tuned_rag)
    slide_data = [[d[0], d[1], d[3]] for d in data]
    slide_dir = ROOT / "benchmark_results" / "slides"
    make_slides(slide_data, slide_dir)

def chart_only():
    """Regenerate slides from saved responses JSON without any API calls."""
    raw_path = ROOT / "benchmark_results" / "rag_comparison_responses.json"
    if not raw_path.exists():
        print(f"No saved responses at {raw_path}. Run without --chart-only first.")
        sys.exit(1)
    with open(raw_path) as f:
        saved = json.load(f)
    # 3 columns: no_rag, bm25_rag, tuned_rag
    data = [[d["no_rag"], d.get("bm25_rag", "[not run]"), d["tuned_rag"]] for d in saved]
    slide_dir = ROOT / "benchmark_results" / "slides"
    make_slides(data, slide_dir)

if __name__ == "__main__":
    if "--chart-only" in sys.argv:
        chart_only()
    else:
        main()
