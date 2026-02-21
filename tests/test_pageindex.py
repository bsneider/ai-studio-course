"""Comparison benchmark: 4 retrieval approaches on hard queries (L3-L5).

Compares:
  1. Hybrid Search (BM25 + semantic vectors) -- our current production approach
  2. BM25-only -- pure keyword search, no embeddings
  3. Semantic-only -- pure vector cosine similarity, no keywords
  4. LLM-Reranking -- hybrid candidates reranked by an LLM for relevance
     (conceptually similar to PageIndex's reasoning-based retrieval)

Background on PageIndex:
  PageIndex (github.com/VectifyAI/PageIndex) is a vectorless, reasoning-based
  RAG system that builds a hierarchical tree index and uses LLM reasoning to
  navigate it. It primarily supports PDF and Markdown (not raw HTML), requires
  either a PageIndex cloud API key or OpenAI API key for local processing.

  Since our data is 7 crawled HTML pages + YouTube transcripts, and we use
  OpenRouter (not OpenAI directly), we implement LLM-reranking as an
  approximation: retrieve broad candidates via hybrid search, then ask an LLM
  to score each for relevance.

Run:
    uv run pytest tests/test_pageindex.py -v -s

Requires:
  - Network access to crawl the site
  - OPENROUTER_API_KEY in .env (for LLM-reranking tests)
  - Pre-downloaded transcripts in data/transcripts/*.txt
"""

import json
import os
import textwrap
from pathlib import Path

import pytest
from openai import OpenAI

from tests.shared import (
    BENCHMARK_QUERIES,
    DIFFERENTIATING_QUERIES,
    HARD_QUERIES,
    bm25_only_search,
    build_rag_db,
    hybrid_search,
    mrr_score,
    ndcg_score,
    pass_at_k,
    pass_power_k,
    recall_at_k,
    score_retrieval,
    semantic_only_search,
)

# ── Configuration ────────────────────────────────────────────────────────────

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_RERANK_MODEL = "google/gemini-2.0-flash-001"


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rag_db(tmp_path_factory):
    """Build full RAG database: crawl site + load transcripts + embed + store."""
    db_path = tmp_path_factory.mktemp("rag_comparison") / "benchmark.db"
    conn, emb_model, all_chunks = build_rag_db(db_path)
    return conn, emb_model, all_chunks


@pytest.fixture(scope="session")
def openrouter_client():
    """Create OpenRouter client (or skip if no API key)."""
    if not OPENROUTER_API_KEY:
        pytest.skip("OPENROUTER_API_KEY not set -- skipping LLM-reranking tests")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


# ── LLM-Reranking Implementation ────────────────────────────────────────────


def llm_rerank_search(conn, emb_model, openrouter_client, query, top_k=5, candidate_k=50):
    """Hybrid retrieval + LLM-based reranking.

    1. Retrieve candidate_k chunks via hybrid search (broad recall from both
       BM25 keywords AND semantic vectors)
    2. Ask an LLM to score each candidate for relevance to the query
    3. Return the top_k highest-scored candidates
    """
    candidates = hybrid_search(conn, emb_model, query, top_k=candidate_k)
    if not candidates:
        return []

    # Build reranking prompt with numbered chunks
    chunks_text = ""
    for i, c in enumerate(candidates):
        content = c["content"][:600]
        chunks_text += f"\n--- Chunk {i} (type: {c['content_type']}) ---\n{content}\n"

    rerank_prompt = textwrap.dedent(f"""\
        You are a retrieval quality judge. Given a user query and {len(candidates)}
        candidate text chunks, score EVERY chunk from 0 to 10 for relevance to
        answering the query. Consider:
        - Does the chunk contain information that directly answers the query?
        - Does it contain keywords, names, or concepts mentioned in the query?
        - Would this chunk be useful as context for generating an answer?

        Return a JSON array of objects with "chunk_id" (int) and "score" (int 0-10).
        You MUST score all {len(candidates)} chunks. Include ALL chunks, even those
        with score 0.

        Query: {query}

        Candidate chunks:
        {chunks_text}

        JSON response (array of {len(candidates)} objects):""")

    try:
        response = openrouter_client.chat.completions.create(
            model=LLM_RERANK_MODEL,
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores = json.loads(raw)
    except (json.JSONDecodeError, IndexError, KeyError, Exception):
        return candidates[:top_k]

    score_map = {s["chunk_id"]: s["score"] for s in scores if isinstance(s, dict)}
    reranked = []
    for i, c in enumerate(candidates):
        llm_score = score_map.get(i, 0)
        reranked.append({**c, "score": llm_score})

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


# ── Tests: BM25-only baseline ───────────────────────────────────────────────


class TestBM25OnlyBaseline:
    """BM25-only search -- measurement baseline, no assertions."""

    @pytest.mark.parametrize("bq", HARD_QUERIES, ids=[e["q"][:50] for e in HARD_QUERIES])
    def test_bm25_hard_queries(self, rag_db, bq):
        conn, _emb_model, _chunks = rag_db
        top_k = bq.get("top_k", 5)
        results = bm25_only_search(conn, bq["q"], top_k=top_k)
        score = score_retrieval(results, bq["keywords"])
        print(f"  BM25-only [{bq['level']}] '{bq['q']}': {score:.0%} ({bq['note']})")

    def test_bm25_overall_hard_pass_rate(self, rag_db):
        conn, _emb_model, _chunks = rag_db
        passed = 0
        for bq in HARD_QUERIES:
            top_k = bq.get("top_k", 5)
            results = bm25_only_search(conn, bq["q"], top_k=top_k)
            score = score_retrieval(results, bq["keywords"])
            if score >= 0.4:
                passed += 1
        rate = passed / len(HARD_QUERIES)
        print(f"\n  BM25-only hard-query pass rate: {rate:.0%} ({passed}/{len(HARD_QUERIES)})")


# ── Tests: Semantic-only baseline ────────────────────────────────────────────


class TestSemanticOnlyBaseline:
    """Pure vector search -- shows where embeddings help or hurt vs keywords."""

    @pytest.mark.parametrize("bq", HARD_QUERIES, ids=[e["q"][:50] for e in HARD_QUERIES])
    def test_semantic_hard_queries(self, rag_db, bq):
        conn, emb_model, _chunks = rag_db
        top_k = bq.get("top_k", 5)
        results = semantic_only_search(conn, emb_model, bq["q"], top_k=top_k)
        score = score_retrieval(results, bq["keywords"])
        print(f"  Semantic-only [{bq['level']}] '{bq['q']}': {score:.0%} ({bq['note']})")

    def test_semantic_overall_hard_pass_rate(self, rag_db):
        conn, emb_model, _chunks = rag_db
        passed = 0
        for bq in HARD_QUERIES:
            top_k = bq.get("top_k", 5)
            results = semantic_only_search(conn, emb_model, bq["q"], top_k=top_k)
            score = score_retrieval(results, bq["keywords"])
            if score >= 0.4:
                passed += 1
        rate = passed / len(HARD_QUERIES)
        print(f"\n  Semantic-only hard-query pass rate: {rate:.0%} ({passed}/{len(HARD_QUERIES)})")


# ── Tests: LLM-Reranking ────────────────────────────────────────────────────


class TestLLMReranking:
    """LLM-reranking: hybrid candidates reranked by LLM reasoning."""

    @pytest.mark.parametrize("bq", HARD_QUERIES, ids=[e["q"][:50] for e in HARD_QUERIES])
    def test_llm_rerank_hard_queries(self, rag_db, openrouter_client, bq):
        conn, emb_model, _chunks = rag_db
        top_k = bq.get("top_k", 5)
        results = llm_rerank_search(conn, emb_model, openrouter_client, bq["q"], top_k=top_k)
        score = score_retrieval(results, bq["keywords"])
        print(f"  LLM-rerank [{bq['level']}] '{bq['q']}': {score:.0%} ({bq['note']})")
        assert score >= 0.3, (
            f"[{bq['level']}] LLM-rerank '{bq['q']}' scored {score:.0%} "
            f"(top_k={top_k}, {bq['note']}). Expected >=30% of {bq['keywords']}"
        )

    def test_llm_rerank_overall_hard_pass_rate(self, rag_db, openrouter_client):
        conn, emb_model, _chunks = rag_db
        passed = 0
        for bq in HARD_QUERIES:
            top_k = bq.get("top_k", 5)
            results = llm_rerank_search(conn, emb_model, openrouter_client, bq["q"], top_k=top_k)
            score = score_retrieval(results, bq["keywords"])
            if score >= 0.4:
                passed += 1
        rate = passed / len(HARD_QUERIES)
        print(f"\n  LLM-rerank hard-query pass rate: {rate:.0%} ({passed}/{len(HARD_QUERIES)})")
        assert rate >= 0.50, (
            f"LLM-rerank hard-query pass rate {rate:.0%} ({passed}/{len(HARD_QUERIES)}) below 50%"
        )


# ── Tests: Side-by-side comparison ──────────────────────────────────────────


class TestSideBySideComparison:
    """Compare all four approaches on the same hard queries."""

    def test_comparison_table(self, rag_db, openrouter_client):
        """Print a side-by-side comparison of all four retrieval approaches."""
        conn, emb_model, _chunks = rag_db

        print("\n" + "=" * 120)
        print("RETRIEVAL COMPARISON: Hard Queries (L3-L5)")
        print("=" * 120)
        print(f"{'Query':<55} {'Level':<5} {'Hybrid':<10} {'BM25':<10} {'Semantic':<10} {'LLM-Rerank':<12} {'Winner'}")
        print("-" * 120)

        totals = {"hybrid": 0, "bm25": 0, "semantic": 0, "llm": 0}
        passes = {"hybrid": 0, "bm25": 0, "semantic": 0, "llm": 0}

        for bq in HARD_QUERIES:
            top_k = bq.get("top_k", 5)

            h_score = score_retrieval(hybrid_search(conn, emb_model, bq["q"], top_k=top_k), bq["keywords"])
            b_score = score_retrieval(bm25_only_search(conn, bq["q"], top_k=top_k), bq["keywords"])
            s_score = score_retrieval(semantic_only_search(conn, emb_model, bq["q"], top_k=top_k), bq["keywords"])
            l_score = score_retrieval(llm_rerank_search(conn, emb_model, openrouter_client, bq["q"], top_k=top_k), bq["keywords"])

            for key, val in [("hybrid", h_score), ("bm25", b_score), ("semantic", s_score), ("llm", l_score)]:
                totals[key] += val
                if val >= 0.4:
                    passes[key] += 1

            best = max(h_score, b_score, s_score, l_score)
            names = {"hybrid": "Hybrid", "bm25": "BM25", "semantic": "Semantic", "llm": "LLM"}
            winners = [names[k] for k, v in [("hybrid", h_score), ("bm25", b_score), ("semantic", s_score), ("llm", l_score)] if v == best]

            q_short = bq["q"][:53]
            print(f"{q_short:<55} {bq['level']:<5} {h_score:<10.0%} {b_score:<10.0%} {s_score:<10.0%} {l_score:<12.0%} {' + '.join(winners)}")

        n = len(HARD_QUERIES)
        print("-" * 120)
        print(f"{'AVERAGE':<55} {'':5} {totals['hybrid']/n:<10.0%} {totals['bm25']/n:<10.0%} {totals['semantic']/n:<10.0%} {totals['llm']/n:<12.0%}")
        print(f"{'PASS RATE (>=40%)':<55} {'':5} {passes['hybrid']/n:<10.0%} {passes['bm25']/n:<10.0%} {passes['semantic']/n:<10.0%} {passes['llm']/n:<12.0%}")
        print("=" * 120)

    def test_full_benchmark_comparison(self, rag_db, openrouter_client):
        """Compare on ALL queries (L1-L5) to show where semantic helps."""
        conn, emb_model, _chunks = rag_db

        print("\n" + "=" * 120)
        print("FULL BENCHMARK COMPARISON: All Queries (L1-L5)")
        print("=" * 120)
        print(f"{'Query':<55} {'Level':<5} {'Hybrid':<10} {'BM25':<10} {'Semantic':<10} {'LLM-Rerank':<12} {'Winner'}")
        print("-" * 120)

        totals = {"hybrid": 0, "bm25": 0, "semantic": 0, "llm": 0}
        passes = {"hybrid": 0, "bm25": 0, "semantic": 0, "llm": 0}

        for bq in BENCHMARK_QUERIES:
            top_k = bq.get("top_k", 5)

            h_score = score_retrieval(hybrid_search(conn, emb_model, bq["q"], top_k=top_k), bq["keywords"])
            b_score = score_retrieval(bm25_only_search(conn, bq["q"], top_k=top_k), bq["keywords"])
            s_score = score_retrieval(semantic_only_search(conn, emb_model, bq["q"], top_k=top_k), bq["keywords"])
            l_score = score_retrieval(llm_rerank_search(conn, emb_model, openrouter_client, bq["q"], top_k=top_k), bq["keywords"])

            for key, val in [("hybrid", h_score), ("bm25", b_score), ("semantic", s_score), ("llm", l_score)]:
                totals[key] += val
                if val >= 0.4:
                    passes[key] += 1

            best = max(h_score, b_score, s_score, l_score)
            names = {"hybrid": "Hybrid", "bm25": "BM25", "semantic": "Semantic", "llm": "LLM"}
            winners = [names[k] for k, v in [("hybrid", h_score), ("bm25", b_score), ("semantic", s_score), ("llm", l_score)] if v == best]

            q_short = bq["q"][:53]
            print(f"{q_short:<55} {bq['level']:<5} {h_score:<10.0%} {b_score:<10.0%} {s_score:<10.0%} {l_score:<12.0%} {' + '.join(winners)}")

        n = len(BENCHMARK_QUERIES)
        print("-" * 120)
        print(f"{'AVERAGE':<55} {'':5} {totals['hybrid']/n:<10.0%} {totals['bm25']/n:<10.0%} {totals['semantic']/n:<10.0%} {totals['llm']/n:<12.0%}")
        print(f"{'PASS RATE (>=40%)':<55} {'':5} {passes['hybrid']/n:<10.0%} {passes['bm25']/n:<10.0%} {passes['semantic']/n:<10.0%} {passes['llm']/n:<12.0%}")
        print("=" * 120)


# ── Tests: IR Evaluation Metrics ─────────────────────────────────────────────


class TestRetrievalMetrics:
    """Compute standard IR evaluation metrics across all 4 approaches on HARD_QUERIES."""

    def test_metrics_summary(self, rag_db, openrouter_client):
        """Compute MRR, NDCG@5, Recall@5, pass@1, pass@3, pass^3 for each approach."""
        conn, emb_model, _chunks = rag_db

        approaches = {
            "Hybrid": lambda bq, k: hybrid_search(conn, emb_model, bq["q"], top_k=k),
            "BM25": lambda bq, k: bm25_only_search(conn, bq["q"], top_k=k),
            "Semantic": lambda bq, k: semantic_only_search(conn, emb_model, bq["q"], top_k=k),
            "LLM-Rerank": lambda bq, k: llm_rerank_search(conn, emb_model, openrouter_client, bq["q"], top_k=k),
        }

        # Collect per-query results for each approach
        approach_results = {}
        for name, search_fn in approaches.items():
            query_data = []
            for bq in HARD_QUERIES:
                top_k = bq.get("top_k", 5)
                results = search_fn(bq, top_k)
                recall = recall_at_k(results, bq["keywords"])
                mrr = mrr_score(results, bq["keywords"])
                ndcg = ndcg_score(results, bq["keywords"])
                query_data.append({
                    "query": bq["q"],
                    "level": bq["level"],
                    "recall": recall,
                    "mrr": mrr,
                    "ndcg": ndcg,
                    "passed": recall >= 0.4,
                })
            approach_results[name] = query_data

        # Compute aggregate metrics per approach
        n = len(HARD_QUERIES)
        metrics = {}
        for name, qdata in approach_results.items():
            c = sum(1 for qd in qdata if qd["passed"])
            metrics[name] = {
                "Recall@K": sum(qd["recall"] for qd in qdata) / n,
                "MRR": sum(qd["mrr"] for qd in qdata) / n,
                "NDCG@K": sum(qd["ndcg"] for qd in qdata) / n,
                "pass@1": pass_at_k(n, c, 1),
                "pass@3": pass_at_k(n, c, 3),
                "pass^3": pass_power_k(n, c, 3),
            }

        # Print per-query detail table
        metric_names = ["Recall@K", "MRR", "NDCG@K", "pass@1", "pass@3", "pass^3"]
        approach_names = list(approaches.keys())

        print("\n" + "=" * 130)
        print("IR EVALUATION METRICS: Hard Queries (L3-L5)")
        print("=" * 130)

        # Per-query breakdown (Recall, MRR, NDCG per approach)
        header = f"{'Query':<50} {'Lvl':<4}"
        for aname in approach_names:
            header += f" {'Rec':>5} {'MRR':>5} {'NDCG':>5} |"
        print(header)
        print(f"{'':50} {'':4}", end="")
        for aname in approach_names:
            label = f"--- {aname} ---"
            print(f" {label:>19}|", end="")
        print()
        print("-" * 130)

        for i, bq in enumerate(HARD_QUERIES):
            q_short = bq["q"][:48]
            line = f"{q_short:<50} {bq['level']:<4}"
            for aname in approach_names:
                qd = approach_results[aname][i]
                line += f" {qd['recall']:5.2f} {qd['mrr']:5.2f} {qd['ndcg']:5.2f} |"
            print(line)

        # Aggregate summary table
        print("\n" + "=" * 90)
        print("AGGREGATE METRICS SUMMARY")
        print("=" * 90)
        header = f"{'Metric':<12}"
        for aname in approach_names:
            header += f" {aname:>12}"
        print(header)
        print("-" * 90)

        for mname in metric_names:
            line = f"{mname:<12}"
            for aname in approach_names:
                line += f" {metrics[aname][mname]:12.4f}"
            print(line)
        print("=" * 90)

        # Identify best approach per metric
        print("\nBest approach per metric:")
        for mname in metric_names:
            best_val = max(metrics[aname][mname] for aname in approach_names)
            best_approaches = [aname for aname in approach_names if metrics[aname][mname] == best_val]
            print(f"  {mname:<12} -> {' + '.join(best_approaches)} ({best_val:.4f})")


# ── Benchmark data capture for visualization ─────────────────────────────────


class TestBenchmarkCapture:
    """Trigger benchmark_capture fixture to save results.json for chart generation."""

    def test_capture_benchmark_results(self, benchmark_capture):
        """Run all 4 approaches on ALL_QUERIES and save to benchmark_results/results.json."""
        n = benchmark_capture["n_queries"]
        approaches = benchmark_capture["approaches"]
        print(f"\n  Captured benchmark data: {n} queries x {len(approaches)} approaches")
        for aname in approaches:
            agg = benchmark_capture["aggregates"][aname]
            print(f"    {aname}: Recall={agg['Recall@K']:.2%}  MRR={agg['MRR']:.2%}  pass_rate={agg['pass_rate']:.0%}")
        assert n > 0
