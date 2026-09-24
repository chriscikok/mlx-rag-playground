# MLX Modular RAG

Modular local RAG that can switch between:
- Basic
- Hybrid (Vector + BM25)
- Corrective (CRAG)
- Adaptive
- Graph RAG
- Agentic / Self-RAG

## Structure

### Entry Points (What you run)
- `app.py` - Streamlit UI with checkboxes to toggle Hybrid / CRAG / Reranker / Adaptive
- `cli.py` - Terminal test with streaming: `python cli.py "What is our refund policy?"`

### Library / Package (What you import)
```
mlx-rag-modular/
├── config.py          # All models & params in one place - edit here to switch models globally
├── requirements.txt
├── data/              # Put PDFs, DOCX, MD here
├── storage/           # Persisted index (auto-created)
├── chroma_db/         # Persisted vector DB (auto-created)
└── core/              # Reusable modules
    ├── __init__.py    # Makes core a Python package + clean re-exports (see below)
    ├── ingestion.py   # Loader + Chunker
    ├── embeddings.py  # BGE-M3 on MPS
    ├── vector_store.py# Chroma wrapper
    ├── retrievers.py  # Vector, BM25, Hybrid, Reranker
    ├── grader.py      # CRAG relevance grader
    ├── router.py      # Adaptive router
    ├── llm.py         # MLXLLM wrapper with streaming
    ├── graph.py       # Graph RAG (optional)
    └── pipeline.py    # Final RAGPipeline that composes everything
```

### What is `core/__init__.py` for?
It is NOT an entry point. It has 2 purposes:

1.  **Makes `core/` a Python package** so `from core.retrievers import ...` works from `app.py` and `cli.py`.
2.  **Clean public API via re-exports** - so you can write:
    ```python
    from core import RAGPipeline, MLXLLM, get_hybrid_retriever
    ```
    instead of remembering deep paths. See `core/__init__.py` for the export list.

Without it, the modular imports would fail.

## Usage

```bash
pip install -r requirements.txt

# Put docs in ./data
# First run will index
streamlit run app.py

# Or CLI
python cli.py "What is our refund policy?"
```

## Switching RAG types

In `pipeline.py`:

```python
pipeline = RAGPipeline(
  index=index,
  llm=llm,
  use_hybrid=True,      # Hybrid RAG
  use_crag=True,        # Corrective RAG
  use_reranker=True,
  use_adaptive=False    # Adaptive RAG
)

# For agentic:
pipeline.agentic_answer("Compare 2023 vs 2024 policy")
```

All modules are reusable - import `get_hybrid_retriever` or `RelevanceGrader` in your own project.
