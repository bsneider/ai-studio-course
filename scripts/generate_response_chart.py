"""
Generate a publishable chart comparing RAG responses to the 5 hardest questions:
- No RAG (LLM only)
- Untuned RAG (default BM25+vector params)
- Tuned RAG (optimized hybrid search)

Uses Gemini Flash via Google API.
"""

import json
import os
import sqlite3
import struct
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import sqlite_vec
from fastembed import TextEmbedding

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY = "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"
MODEL = "google/gemini-2.0-flash-001"
DB_PATH = Path(__file__).parent.parent / "ai_studio_rag.db"

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from shared import COMPLEX_QUERIES

# Pick 5 hardest: all L12, L11 first, then fill with L10
_by_level = {"L12": [], "L11": [], "L10": [], "L9": []}
for q in COMPLEX_QUERIES:
    _by_level.get(q["level"], []).append(q)

_selected = (_by_level["L12"] + _by_level["L11"] + _by_level["L10"] + _by_level["L9"])[:5]
QUESTIONS = [q["q"] for q in _selected]
QUESTION_LEVELS = [q["level"] for q in _selected]
QUESTION_TOP_K = [q.get("top_k", 8) for q in _selected]

# Short labels for chart headers (wrap at ~18 chars)
QUESTION_LABELS = [
    "\n".join(textwrap.wrap(q["q"], width=22)) for q in _selected
]

# ── DB + Embedding setup ───────────────────────────────────────────────────────

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

_model = None
def get_embedding(text):
    global _model
    if _model is None:
        _model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    return list(_model.embed([text]))[0].tolist()

# ── Search functions ───────────────────────────────────────────────────────────

STOP_WORDS = {"what","are","all","the","or","and","a","an","is","in","of","for",
              "to","do","how","who","why","when","where","which","this","that",
              "these","those","have","has","been","did","does","was","were","will",
              "would","could","should","their","they","them","its","it","be","at",
              "on","as","by","with","about","any","from","not","has","ever"}

def bm25_search(conn, query, top_k):
    tokens = [t for t in query.lower().split() if t not in STOP_WORDS]
    if not tokens:
        return []
    bm25_query = " OR ".join(tokens)
    try:
        rows = conn.execute(
            "SELECT rowid, rank FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?",
            (bm25_query, top_k)
        ).fetchall()
        return [(r["rowid"], abs(r["rank"])) for r in rows]
    except Exception:
        return []

def vec_search(conn, query_emb, top_k):
    blob = struct.pack(f"{len(query_emb)}f", *query_emb)
    rows = conn.execute(
        "SELECT rowid, distance FROM vec_documents WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (blob, top_k)
    ).fetchall()
    return [(r["rowid"], r["distance"]) for r in rows]

def hybrid_search(conn, query, query_emb, top_k=5, bm25_w=0.4, vec_w=0.6):
    bm25 = bm25_search(conn, query, top_k * 2)
    vecs = vec_search(conn, query_emb, top_k * 2)

    max_bm25 = max((s for _, s in bm25), default=1)
    bm25_scores = {rid: s / max_bm25 for rid, s in bm25}
    vec_scores = {rid: 1.0 / (1 + d) for rid, d in vecs}

    all_ids = set(bm25_scores) | set(vec_scores)
    combined = {cid: bm25_w * bm25_scores.get(cid, 0) + vec_w * vec_scores.get(cid, 0)
                for cid in all_ids}
    top_ids = sorted(combined, key=combined.get, reverse=True)[:top_k]

    results = []
    for cid in top_ids:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (cid,)).fetchone()
        if row:
            results.append(dict(row))
    return results

def format_context(results, max_chars=4000):
    parts, chars = [], 0
    for i, r in enumerate(results, 1):
        entry = f"[Source {i}: {r.get('page_title','?')} > {r.get('section_title','?')}]\n{r.get('content','')}"
        if chars + len(entry) > max_chars:
            break
        parts.append(entry)
        chars += len(entry)
    return "\n\n".join(parts) if parts else "No relevant documents found."

# ── LLM call ─────────────────────────────────────────────────────────────────

def call_gemini(prompt, system=None):
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        messages=messages,
    )
    return response.choices[0].message.content.strip()

SYSTEM = """You are a helpful assistant for the MIT AI Studio course at MIT Media Lab.
Answer concisely in 2-4 sentences. If you don't know, say so clearly."""

SYSTEM_RAG = """You are a helpful assistant for the MIT AI Studio course at MIT Media Lab.
Answer ONLY using the provided context. Be concise (2-4 sentences).
If the context doesn't contain enough information, say so clearly."""

def no_rag_answer(question):
    return call_gemini(f"Question: {question}", system=SYSTEM)

def untuned_rag_answer(conn, question):
    emb = get_embedding(question)
    # Untuned: low top_k, equal bm25/vec weights
    results = hybrid_search(conn, question, emb, top_k=3, bm25_w=0.5, vec_w=0.5)
    context = format_context(results, max_chars=2000)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return call_gemini(prompt, system=SYSTEM_RAG)

def tuned_rag_answer(conn, question, top_k=8):
    emb = get_embedding(question)
    # Tuned: benchmark top_k, optimized weights favoring semantic
    results = hybrid_search(conn, question, emb, top_k=top_k, bm25_w=0.35, vec_w=0.65)
    context = format_context(results, max_chars=4000)
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return call_gemini(prompt, system=SYSTEM_RAG)

# ── Chart generation ──────────────────────────────────────────────────────────

def wrap(text, width=52):
    return "\n".join(textwrap.wrap(text, width))

def generate_chart(data, output_path):
    n_questions = len(QUESTIONS)
    conditions = ["No RAG\n(LLM Only)", "Untuned RAG", "Tuned RAG"]
    colors = ["#E8D5D5", "#D5E8F0", "#C8E6C9"]
    header_colors = ["#C0392B", "#2980B9", "#27AE60"]

    fig_width = 22
    col_width = fig_width / (n_questions + 1)
    row_height = 3.2
    header_height = 1.0
    q_header_height = 1.2
    n_rows = len(conditions)
    fig_height = q_header_height + n_rows * row_height + header_height + 1.2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")

    # Title
    fig.text(0.5, 0.98, "RAG Quality Comparison: Hardest Questions",
             ha="center", va="top", fontsize=18, fontweight="bold", color="#1a1a2e")
    levels_str = " · ".join(f"Q{i+1}={lv}" for i, lv in enumerate(QUESTION_LEVELS))
    fig.text(0.5, 0.955, f"MIT AI Studio Course  ·  Gemini 2.0 Flash  ·  {levels_str}",
             ha="center", va="top", fontsize=10, color="#555", style="italic")

    y_top = fig_height - header_height - 0.3

    # Row labels (left column)
    for i, (cond, hc, bg) in enumerate(zip(conditions, header_colors, colors)):
        y = y_top - q_header_height - i * row_height
        rect = FancyBboxPatch((0.1, y - row_height + 0.1), col_width - 0.25, row_height - 0.2,
                               boxstyle="round,pad=0.05", facecolor=bg, edgecolor=hc, linewidth=2)
        ax.add_patch(rect)
        ax.text(col_width / 2, y - row_height / 2, cond,
                ha="center", va="center", fontsize=10, fontweight="bold", color=hc)

    # Question headers
    for j, (q, label) in enumerate(zip(QUESTIONS, QUESTION_LABELS)):
        x = col_width * (j + 1)
        rect = FancyBboxPatch((x + 0.1, y_top - q_header_height + 0.1), col_width - 0.25, q_header_height - 0.2,
                               boxstyle="round,pad=0.05", facecolor="#F0F0F8", edgecolor="#7986CB", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + col_width / 2, y_top - q_header_height / 2, label,
                ha="center", va="center", fontsize=9, fontweight="bold", color="#3949AB",
                multialignment="center")

    # Response cells
    for i, (cond, bg, hc) in enumerate(zip(conditions, colors, header_colors)):
        for j, q in enumerate(QUESTIONS):
            x = col_width * (j + 1)
            y = y_top - q_header_height - i * row_height
            response = data[j][i]

            rect = FancyBboxPatch((x + 0.1, y - row_height + 0.1), col_width - 0.25, row_height - 0.2,
                                   boxstyle="round,pad=0.05", facecolor=bg, edgecolor=hc, linewidth=0.8, alpha=0.6)
            ax.add_patch(rect)

            wrapped = wrap(response, width=48)
            ax.text(x + col_width / 2, y - row_height / 2, wrapped,
                    ha="center", va="center", fontsize=7.2, color="#222",
                    multialignment="left", linespacing=1.4)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=colors[0], edgecolor=header_colors[0], label="No RAG — LLM uses only training knowledge"),
        mpatches.Patch(facecolor=colors[1], edgecolor=header_colors[1], label="Untuned RAG — top_k=3, equal BM25/vector weights"),
        mpatches.Patch(facecolor=colors[2], edgecolor=header_colors[2], label="Tuned RAG — top_k=8, optimized weights, larger context"),
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=9,
              framealpha=0.9, edgecolor="#ccc")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Chart saved to {output_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Connecting to DB...")
    conn = get_conn()

    print("Generating embeddings model...")
    get_embedding("warmup")  # load model

    data = []  # data[question_idx][condition_idx] = response text
    for qi, question in enumerate(QUESTIONS):
        print(f"\n[{qi+1}/{len(QUESTIONS)}] {question[:60]}...")
        row = []

        print("  -> No RAG...")
        try:
            row.append(no_rag_answer(question))
        except Exception as e:
            row.append(f"[Error: {e}]")

        print("  -> Untuned RAG...")
        try:
            row.append(untuned_rag_answer(conn, question))
        except Exception as e:
            row.append(f"[Error: {e}]")

        print("  -> Tuned RAG...")
        try:
            row.append(tuned_rag_answer(conn, question, top_k=QUESTION_TOP_K[qi]))
        except Exception as e:
            row.append(f"[Error: {e}]")

        data.append(row)
        print(f"  No RAG:     {row[0][:80]}...")
        print(f"  Untuned:    {row[1][:80]}...")
        print(f"  Tuned:      {row[2][:80]}...")

    out = Path(__file__).parent.parent / "benchmark_results" / "rag_comparison_chart.png"
    generate_chart(data, out)

    # Also save raw responses
    raw_out = Path(__file__).parent.parent / "benchmark_results" / "rag_comparison_responses.json"
    with open(raw_out, "w") as f:
        json.dump([{"question": q, "no_rag": d[0], "untuned_rag": d[1], "tuned_rag": d[2]}
                   for q, d in zip(QUESTIONS, data)], f, indent=2)
    print(f"Raw responses saved to {raw_out}")

if __name__ == "__main__":
    main()
