"""Tests for the RAG workshop notebook.

Two test modes:
  1. Structural validation (always runs) — checks format, syntax, imports
  2. Full e2e execution via nbmake (requires network + API key)

Run structural tests:
    uv run pytest tests/test_notebook.py -v

Run full e2e (executes every cell, needs OPENROUTER_API_KEY env var):
    uv run pytest --nbmake rag_workshop.ipynb --nbmake-timeout=300
"""

import ast
import json
import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOK_PATH = Path(__file__).parent.parent / "rag_workshop.ipynb"


@pytest.fixture
def notebook():
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


# ── Structural Tests ────────────────────────────────────────────────────────


class TestNotebookStructure:
    """Validate the notebook is well-formed and students won't hit structural issues."""

    def test_valid_nbformat(self, notebook):
        """Notebook conforms to the Jupyter notebook schema."""
        nbformat.validate(notebook)

    def test_has_expected_cell_count(self, notebook):
        """Notebook has the right number of cells (catches accidental deletions)."""
        assert len(notebook.cells) >= 20, f"Expected 20+ cells, got {len(notebook.cells)}"

    def test_has_markdown_and_code(self, notebook):
        """Notebook has both explanation and executable cells."""
        types = {c.cell_type for c in notebook.cells}
        assert "markdown" in types, "No markdown cells found"
        assert "code" in types, "No code cells found"

    def test_starts_with_title(self, notebook):
        """First cell is the branding/title markdown."""
        first = notebook.cells[0]
        assert first.cell_type == "markdown"
        assert "AI STUDIO" in first.source or "RAG Workshop" in first.source

    def test_no_empty_code_cells(self, notebook):
        """No code cells are completely empty (students would be confused)."""
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                source = cell.source.strip()
                assert source, f"Cell {i} is an empty code cell"

    def test_all_code_cells_parse(self, notebook):
        """All code cells are valid Python syntax (catches typos/broken edits)."""
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                source = cell.source
                # Skip cells with IPython magics (%%capture, !pip)
                lines = source.split("\n")
                py_lines = [
                    l for l in lines
                    if not l.strip().startswith(("!", "%%", "%"))
                ]
                py_source = "\n".join(py_lines)
                if py_source.strip():
                    try:
                        ast.parse(py_source)
                    except SyntaxError as e:
                        pytest.fail(f"Cell {i} has a syntax error: {e}")

    def test_no_hardcoded_api_keys(self, notebook):
        """No API keys are hardcoded in the notebook."""
        for i, cell in enumerate(notebook.cells):
            source = cell.source
            # Markdown cells may reference key prefixes in docs — only check code cells
            if cell.cell_type != "code":
                continue
            # Check for actual key values (prefix + long alphanumeric string)
            assert "sk-or-v1-" not in source, f"Cell {i} contains a hardcoded OpenRouter key"
            # Allow prefix checks like startswith("sk-proj-") but catch actual keys
            if re.search(r'sk-proj-[A-Za-z0-9]{20,}', source):
                pytest.fail(f"Cell {i} contains a hardcoded OpenAI key")
            if re.search(r'sk-ant-[A-Za-z0-9]{20,}', source):
                pytest.fail(f"Cell {i} contains a hardcoded Anthropic key")


class TestNotebookContent:
    """Validate the workshop content is complete and well-ordered."""

    def test_has_all_parts(self, notebook):
        """All 8 parts of the workshop are present."""
        all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        expected_parts = [
            "Part 1",
            "Part 2",
            "Part 3",
            "Part 4",
            "Part 5",
            "Part 6",
            "Part 7",
            "Part 8",
        ]
        for part in expected_parts:
            assert part in all_text, f"Missing section: {part}"

    def test_has_exercises(self, notebook):
        """Exercises section exists for student practice."""
        all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        assert "Exercise" in all_text, "No exercises found"

    def test_has_academic_references(self, notebook):
        """Academic papers are cited (required for MBA audience)."""
        all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        assert "Lewis et al" in all_text, "Missing Lewis et al. (2020) RAG paper"
        assert "Karpukhin" in all_text, "Missing Karpukhin et al. (2020) DPR paper"
        assert "arxiv.org" in all_text, "No arxiv links found"

    def test_has_embedding_model_comparison(self, notebook):
        """Embedding model comparison table is present."""
        all_text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        assert "BGE-M3" in all_text, "Missing BGE-M3 in model comparison"
        assert "OpenAI" in all_text, "Missing OpenAI in model comparison"
        assert "fine-tune" in all_text.lower() or "fine-tuning" in all_text.lower(), (
            "Missing fine-tuning discussion"
        )

    def test_has_learning_outcomes(self, notebook):
        """Learning outcomes are stated upfront."""
        first_md = "\n".join(
            c.source for c in notebook.cells[:3] if c.cell_type == "markdown"
        )
        assert "Learning Outcomes" in first_md, "Missing learning outcomes in intro"

    def test_has_instructor_info(self, notebook):
        """Instructor branding is present."""
        first_md = notebook.cells[0].source
        assert "Brandon Sneider" in first_md, "Missing instructor name"
        assert "linkedin.com" in first_md, "Missing LinkedIn link"

    def test_pip_install_is_first_code_cell(self, notebook):
        """First code cell is the pip install (so deps are available for all other cells)."""
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert "pip install" in code_cells[0].source, (
            "First code cell should be pip install"
        )

    def test_imports_before_usage(self, notebook):
        """Import cell comes before any cells that use the libraries."""
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        # Second code cell should have the imports
        assert "import sqlite3" in code_cells[1].source, (
            "Second code cell should import core libraries"
        )
        assert "import json" in code_cells[1].source


class TestNotebookDependencies:
    """Validate all required packages are importable."""

    @pytest.mark.parametrize(
        "module",
        [
            "json",
            "sqlite3",
            "struct",
            "re",
            "requests",
            "bs4",
            "numpy",
            "sqlite_vec",
            "fastembed",
            "openai",
            "nbformat",
            "ipywidgets",
        ],
    )
    def test_import(self, module):
        """All required Python packages can be imported."""
        __import__(module)
