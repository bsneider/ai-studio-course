"""Pytest plugin: capture benchmark data from all 4 retrieval approaches.

Runs after all tests in test_pageindex.py complete. Evaluates every query in
ALL_QUERIES against Hybrid, BM25, Semantic, and LLM-Rerank search, computes
per-query IR metrics (recall, MRR, NDCG), and writes structured results to
benchmark_results/results.json for downstream chart generation.

The JSON capture is triggered by the pytest_sessionfinish hook and only runs
when test_pageindex.py tests were collected.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Lazy imports -- these are only needed when the benchmark fixture actually runs.
# We import at call time to avoid import errors when pytest collects other files.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
_RESULTS_DIR = _ROOT / "benchmark_results"

# Store results on the pytest session object so pytest_sessionfinish can write them
_BENCHMARK_DATA = {}


def _ensure_openrouter_client():
    """Build an OpenRouter client, or return None if no key."""
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


@pytest.fixture(scope="session")
def benchmark_capture(rag_db, request):
    """Session fixture that runs all 4 approaches on ALL_QUERIES and stores results.

    This fixture is *requested* by a dedicated test so it runs exactly once.
    Results are stashed on the session for pytest_sessionfinish to write out.
    """
    from tests.shared import (
        ALL_QUERIES,
        bm25_only_search,
        hybrid_search,
        hybrid_search_expanded,
        hybrid_search_rrf,
        improved_llm_rerank_search,
        mrr_score,
        ndcg_score,
        pageindex_tree_search,
        pass_at_k,
        pass_power_k,
        recall_at_k,
        semantic_only_search,
    )
    # Late import of llm_rerank_search and model list from the test module
    from tests.test_pageindex import LLM_RERANK_MODELS, llm_rerank_search

    conn, emb_model, _chunks = rag_db
    openrouter = _ensure_openrouter_client()

    approaches = {
        "Hybrid": lambda bq, k: hybrid_search(conn, emb_model, bq["q"], top_k=k),
        "Hybrid+Expand": lambda bq, k: hybrid_search_expanded(conn, emb_model, bq["q"], top_k=k),
        "Hybrid+RRF": lambda bq, k: hybrid_search_rrf(conn, emb_model, bq["q"], top_k=k),
        "BM25": lambda bq, k: bm25_only_search(conn, bq["q"], top_k=k),
        "Semantic": lambda bq, k: semantic_only_search(conn, emb_model, bq["q"], top_k=k),
    }
    if openrouter:
        for model_id, display_name in LLM_RERANK_MODELS.items():
            label = f"LLM:{display_name}"
            # Capture model_id in closure
            approaches[label] = (lambda mid: lambda bq, k: llm_rerank_search(
                conn, emb_model, openrouter, bq["q"], top_k=k, model=mid
            ))(model_id)
        approaches["LLM:improved"] = lambda bq, k: improved_llm_rerank_search(
            conn, emb_model, openrouter, bq["q"], top_k=k
        )
        approaches["PageIndex"] = lambda bq, k: pageindex_tree_search(
            openrouter, bq["q"], top_k=k
        )

    per_query = []
    for bq in ALL_QUERIES:
        top_k = bq.get("top_k", 5)
        inverse = bq.get("scoring") == "inverse"
        query_row = {
            "query": bq["q"],
            "level": bq["level"],
            "scoring": "inverse" if inverse else "normal",
            "keywords": bq["keywords"],
            "top_k": top_k,
            "approaches": {},
        }
        for aname, search_fn in approaches.items():
            results = search_fn(bq, top_k)
            rec = recall_at_k(results, bq["keywords"])
            mrr = mrr_score(results, bq["keywords"])
            ndcg = ndcg_score(results, bq["keywords"])

            # For inverse-scored queries (L8), finding keywords is bad.
            # Invert ALL metrics so that lower raw = higher effective.
            if inverse:
                eff_recall = 1.0 - rec
                eff_mrr = 1.0 - mrr
                eff_ndcg = 1.0 - ndcg
            else:
                eff_recall = rec
                eff_mrr = mrr
                eff_ndcg = ndcg

            query_row["approaches"][aname] = {
                "recall": rec,
                "mrr": mrr,
                "ndcg": ndcg,
                "effective_recall": eff_recall,
                "effective_mrr": eff_mrr,
                "effective_ndcg": eff_ndcg,
                "passed": eff_recall >= 0.4,
            }
        per_query.append(query_row)

    # Aggregate metrics per approach
    approach_names = list(approaches.keys())
    n = len(ALL_QUERIES)
    aggregates = {}
    for aname in approach_names:
        recalls = [q["approaches"][aname]["effective_recall"] for q in per_query]
        mrrs = [q["approaches"][aname]["effective_mrr"] for q in per_query]
        ndcgs = [q["approaches"][aname]["effective_ndcg"] for q in per_query]
        c = sum(1 for q in per_query if q["approaches"][aname]["passed"])
        aggregates[aname] = {
            "Recall@K": sum(recalls) / n,
            "MRR": sum(mrrs) / n,
            "NDCG@K": sum(ndcgs) / n,
            "pass@1": pass_at_k(n, c, 1),
            "pass@3": pass_at_k(n, c, 3),
            "pass^3": pass_power_k(n, c, 3),
            "pass_rate": c / n,
            "n_passed": c,
            "n_total": n,
        }

    # Per-level aggregates
    levels = sorted(set(q["level"] for q in per_query), key=lambda x: int(x[1:]))
    per_level = {}
    for level in levels:
        level_queries = [q for q in per_query if q["level"] == level]
        nl = len(level_queries)
        per_level[level] = {"n": nl, "approaches": {}}
        for aname in approach_names:
            level_recalls = [q["approaches"][aname]["effective_recall"] for q in level_queries]
            level_mrrs = [q["approaches"][aname]["effective_mrr"] for q in level_queries]
            level_ndcgs = [q["approaches"][aname]["effective_ndcg"] for q in level_queries]
            per_level[level]["approaches"][aname] = {
                "mean_recall": sum(level_recalls) / nl,
                "mean_mrr": sum(level_mrrs) / nl,
                "mean_ndcg": sum(level_ndcgs) / nl,
            }

    # ── Weight sweep: vary BM25/Semantic ratio in hybrid search ──────────
    # Sweep from pure BM25 (kw=1.0) to pure semantic (sw=1.0) in 11 steps,
    # plus the LLM-rerank approach as a separate data point.
    weight_sweep = []
    for step in range(11):  # 0.0, 0.1, ..., 1.0 semantic weight
        sw = round(step * 0.1, 1)
        kw = round(1.0 - sw, 1)
        label = f"BM25={kw:.0%} Sem={sw:.0%}"

        sweep_recalls = []
        for bq in ALL_QUERIES:
            top_k = bq.get("top_k", 5)
            inverse = bq.get("scoring") == "inverse"
            results = hybrid_search(conn, emb_model, bq["q"], top_k=top_k, kw=kw, sw=sw)
            rec = recall_at_k(results, bq["keywords"])
            eff = (1.0 - rec) if inverse else rec
            sweep_recalls.append(eff)

        mean_recall = sum(sweep_recalls) / len(sweep_recalls)
        pass_count = sum(1 for r in sweep_recalls if r >= 0.4)
        weight_sweep.append({
            "kw": kw,
            "sw": sw,
            "label": label,
            "mean_recall": mean_recall,
            "pass_rate": pass_count / len(sweep_recalls),
        })

    # Also record LLM/PageIndex approaches as separate points for the progression chart
    for aname in approach_names:
        if aname.startswith("LLM:") or aname == "PageIndex":
            agg_entry = aggregates[aname]
            weight_sweep.append({
                "kw": None,
                "sw": None,
                "label": aname,
                "mean_recall": agg_entry["Recall@K"],
                "pass_rate": agg_entry["pass_rate"],
            })

    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_queries": n,
        "approaches": approach_names,
        "per_query": per_query,
        "aggregates": aggregates,
        "per_level": per_level,
        "weight_sweep": weight_sweep,
    }

    # Stash on the module-level dict for sessionfinish
    _BENCHMARK_DATA.update(data)

    return data


def pytest_sessionfinish(session, exitstatus):
    """Write benchmark results to JSON after all tests complete."""
    if not _BENCHMARK_DATA:
        return
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RESULTS_DIR / "results.json"
    out.write_text(json.dumps(_BENCHMARK_DATA, indent=2))
    print(f"\n[conftest] Benchmark results written to {out}")
