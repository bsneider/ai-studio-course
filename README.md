# MIT AI Studio RAG Workshop

Build a Retrieval-Augmented Generation (RAG) system that lets you **chat with the MIT AI Studio course website** — all inside a Jupyter notebook.

**Course:** MAS.664 / MAS.665 / EC.731 / IDS.865 — MIT Media Lab
**Lead Professor:** Ramesh Raskar
**Workshop Instructor:** [Brandon Sneider](https://linkedin.com/in/brandonsneider)

## Quickstart (5 minutes)

### 1. Clone and install

```bash
git clone https://github.com/bsneider/ai-studio-course.git
cd ai-studio-course
uv sync
```

> No `uv`? Install it: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. Get a free API key

Sign up at [OpenRouter](https://openrouter.ai/keys) (free tier available) and create a `.env` file:

```bash
cp .env.example .env
# Edit .env and paste your key
```

### 3. Open the notebook

```bash
uv run jupyter notebook rag_workshop.ipynb
```

### 4. Chat immediately

The repo ships with a **pre-built database** (`ai_studio_rag.db`, ~13 MB) containing 381 indexed documents from the course website and YouTube transcripts. You can skip straight to **Part 5** (Hybrid Search) or **Part 7** (Interactive Chat) to start asking questions right away.

Or run through Parts 1-4 to rebuild the database from scratch and learn how RAG indexing works.

## What's in the notebook

| Section | What you'll learn | Time |
|---------|-------------------|------|
| Part 1 | Setup & installation | 2 min |
| Part 2 | Initialize a SQLite vector database (FTS5 + sqlite-vec) | 3 min |
| Part 3 | Crawl the course website & YouTube transcripts | 5 min |
| Part 4 | Embedding models — how text becomes vectors | 5 min |
| Part 5 | Hybrid search — combining BM25 keyword + semantic vector search | 5 min |
| Part 6 | Connect an LLM for grounded generation | 3 min |
| Part 7 | Interactive chat interface | 5 min |
| Part 8 | RAG recap & benchmark results | — |

## Project structure

```
ai-studio-course/
├── rag_workshop.ipynb          # Main workshop notebook
├── ai_studio_rag.db            # Pre-built vector database (ready to query)
├── .env.example                # API key template
├── pyproject.toml              # Dependencies (use `uv sync` to install)
├── data/
│   ├── transcripts/            # YouTube video transcripts (.json3 + .txt)
│   └── trees/                  # Hierarchical page structures for tree search
├── scripts/
│   ├── build_trees.py          # Crawl HTML pages → tree JSON structures
│   ├── rag_compare.py          # Generate RAG quality comparison charts
│   ├── generate_charts.py      # Benchmark visualization charts
│   └── generate_response_chart.py
├── tests/
│   ├── shared.py               # 35 benchmark queries across 12 difficulty levels
│   ├── conftest.py             # Pytest plugin for benchmark capture
│   ├── test_pageindex.py       # Retrieval approach comparison benchmarks
│   ├── test_eval.py            # RAG evaluation tests
│   └── test_notebook.py        # Notebook integration tests
└── benchmark_results/          # Pre-computed benchmark outputs
    ├── results.json            # Full benchmark results (10 approaches, 35 queries)
    ├── rag_comparison_responses.json
    └── *.png                   # Visualization charts
```

## The database

`ai_studio_rag.db` is a SQLite database with three search capabilities:

- **BM25 keyword search** via FTS5 — great for exact names and terms
- **Semantic vector search** via sqlite-vec — great for meaning-based queries
- **Hybrid search** — combines both with tunable weights (best overall)

**Data sources indexed:**
- 7 course web pages from [aiforimpact.github.io](https://aiforimpact.github.io/)
- 7 YouTube video transcripts (Demo Days, talks)
- 381 unique document chunks total

## Benchmark results

The `benchmark_results/` directory contains evaluations of 10 retrieval approaches across 35 queries at 12 difficulty levels. Key finding: **tuned hybrid search (BM25 weight 0.4, semantic weight 0.6) beats pure keyword or pure semantic search, and matches LLM-reranking approaches at zero additional API cost.**

## Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An API key from [OpenRouter](https://openrouter.ai/) (free) or [OpenAI](https://platform.openai.com/)

## Running tests

```bash
uv run pytest tests/ -v
```

## License

Educational use for MIT AI Studio course participants.
