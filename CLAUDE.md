# CLAUDE.md

## Project overview

RAG (Retrieval-Augmented Generation) workshop for the MIT AI Studio course. Students build a system that crawls the course website, stores it in a vector database, and chats with the data via a Jupyter notebook.

## Key commands

```bash
uv sync                              # Install dependencies
uv run jupyter notebook rag_workshop.ipynb  # Open the workshop
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/test_eval.py -v  # Run RAG evaluation tests only
uv run ruff check .                  # Lint
```

## Architecture

- **Database**: SQLite with FTS5 (BM25 keyword search) + sqlite-vec (vector similarity search)
- **Embeddings**: fastembed (BAAI/bge-small-en-v1.5)
- **LLM**: OpenRouter API (OpenAI-compatible client) — key in `.env`
- **Notebook**: `rag_workshop.ipynb` — 8 parts, from setup to interactive chat
- **Pre-built DB**: `ai_studio_rag.db` ships with the repo so students can skip to chat

## File layout

- `rag_workshop.ipynb` — Main workshop notebook (Parts 1-8)
- `ai_studio_rag.db` — Pre-built SQLite vector database (381 docs, ~13 MB)
- `data/transcripts/` — YouTube transcript source files
- `data/trees/` — Hierarchical page structures (built by `scripts/build_trees.py`)
- `scripts/` — Data generation, chart generation, tree parsing
- `tests/shared.py` — 35 benchmark queries with expected results, search implementations
- `tests/conftest.py` — Pytest plugin that runs full benchmark suite after tests
- `benchmark_results/` — Pre-computed results and charts

## Important patterns

- The database uses hybrid search: `bm25_score * bm25_weight + semantic_score * (1 - bm25_weight)`
- Optimal weights: BM25 = 0.4, Semantic = 0.6
- All LLM calls go through OpenRouter using the `openai` Python client with `base_url="https://openrouter.ai/api/v1"`
- Environment variable: `OPENROUTER_API_KEY` (from `.env`)
- The notebook can rebuild the database from scratch (Parts 2-4) or use the pre-built one

## Testing

Tests in `tests/test_pageindex.py` compare 10 retrieval approaches. The conftest plugin automatically runs a full benchmark sweep after the test session and writes results to `benchmark_results/results.json`.
