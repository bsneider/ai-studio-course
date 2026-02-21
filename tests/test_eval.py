"""RAG retrieval quality benchmark.

Tests that the ingestion pipeline + hybrid search retrieves relevant content
for queries at 5 difficulty levels (L1-L5). No LLM calls needed -- this only
tests the retrieval component.

Run:
    uv run pytest tests/test_eval.py -v

Requires network access to crawl the site. Uses pre-downloaded transcripts
from data/transcripts/*.txt.
"""

import pytest

from tests.shared import (
    BENCHMARK_QUERIES,
    build_rag_db,
    hybrid_search,
    score_retrieval,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rag_db(tmp_path_factory):
    """Build full RAG database: crawl site + load transcripts + embed + store."""
    db_path = tmp_path_factory.mktemp("rag") / "benchmark.db"
    conn, emb_model, _chunks = build_rag_db(db_path)
    return conn, emb_model


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRAGRetrieval:
    """Benchmark retrieval quality at L1-L5 difficulty."""

    @pytest.mark.parametrize("bq", BENCHMARK_QUERIES, ids=[e["q"][:50] for e in BENCHMARK_QUERIES])
    def test_retrieval_quality(self, rag_db, bq):
        conn, emb_model = rag_db
        top_k = bq.get("top_k", 5)
        results = hybrid_search(conn, emb_model, bq["q"], top_k=top_k)
        score = score_retrieval(results, bq["keywords"])
        assert score >= 0.4, (
            f"[{bq['level']}] '{bq['q']}' scored {score:.0%} "
            f"(top_k={top_k}, {bq['note']}). Expected >=40% of {bq['keywords']}"
        )

    def test_overall_pass_rate(self, rag_db):
        """At least 70% of all benchmark queries should pass."""
        conn, emb_model = rag_db
        passed = 0
        for bq in BENCHMARK_QUERIES:
            top_k = bq.get("top_k", 5)
            results = hybrid_search(conn, emb_model, bq["q"], top_k=top_k)
            score = score_retrieval(results, bq["keywords"])
            if score >= 0.4:
                passed += 1
        rate = passed / len(BENCHMARK_QUERIES)
        assert rate >= 0.70, f"Overall pass rate {rate:.0%} ({passed}/{len(BENCHMARK_QUERIES)}) below 70% threshold"

    def test_transcript_queries_pass(self, rag_db):
        """L4+ queries (transcript-dependent) must pass -- this is the key value-add."""
        conn, emb_model = rag_db
        l4_plus = [e for e in BENCHMARK_QUERIES if e["level"] in ("L4", "L5")]
        passed = 0
        for bq in l4_plus:
            top_k = bq.get("top_k", 5)
            results = hybrid_search(conn, emb_model, bq["q"], top_k=top_k)
            score = score_retrieval(results, bq["keywords"])
            if score >= 0.4:
                passed += 1
        rate = passed / len(l4_plus)
        assert rate >= 0.60, f"L4+L5 pass rate {rate:.0%} ({passed}/{len(l4_plus)}) below 60% threshold"
