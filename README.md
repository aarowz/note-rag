# note-rag

Small **local-only** CLI pipeline for Retrieval-Augmented Generation (RAG) over a folder of Markdown files:

1. **`note-rag index`** — chunk by `##` headings, embed with [sentence-transformers](https://www.sbert.net/), persist vectors in [Chroma](https://www.trychroma.com/).
2. **`note-rag ask`** — retrieve top‑k chunks, call [Ollama](https://ollama.com/) `/api/chat`, print an answer with numbered citations and source paths.

**Limitations (v0.1):** single-machine CLI; full index rebuild each run; Markdown only; English-oriented defaults; citations are best-effort LLM behavior plus a simple bracket-id sanity check.

## Prerequisites

- Python **3.11+**
- [Ollama](https://ollama.com/) installed and running, with a chat model pulled (examples below use `llama3.2`)

## Install

```bash
cd note-rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

First run downloads the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) from the Hugging Face Hub (you may see a harmless warning about unauthenticated requests).

## Demo

From the repo root (after install):

```bash
# 1) Build / rebuild the index over ./corpus
note-rag index corpus --db ./.chroma

# 2) Inspect retrieval only (no LLM)
note-rag ask "What is safety stock?" --dry-run --db ./.chroma

# 3) Answer with Ollama (requires: ollama pull llama3.2)
note-rag ask "Define SKU and how it differs from a product family." --db ./.chroma --model llama3.2
```

Useful flags:

- `--k` — number of chunks to retrieve (default `5`)
- `--embed-model` — sentence-transformers model id
- `--ollama-host` — override base URL (default `OLLAMA_HOST` or `http://127.0.0.1:11434`)
- `--max-chars` — per-section split size before sub-chunking (default `2000`)

## Chunking rules

- New chunks start at Markdown **`##` headings**. A leading `#` title groups content until the first `##`.
- Text before any heading goes under **`Preamble`**.
- Very long sections are split into multiple parts with the same heading path (see `heading_slug` metadata).

## Tests

```bash
pytest
```

## Layout

| Module | Role |
|--------|------|
| `chunking.py` | `##`-aware splits + max-length subchunks |
| `embedder.py` | sentence-transformers wrapper |
| `store.py` | Chroma persistent collection |
| `llm.py` | Ollama HTTP client |
| `pipeline.py` | index / retrieve / prompt assembly |
| `cli.py` | Typer commands |

## License

Specify your preferred license when you publish the repo (this starter ships without a `LICENSE` file).
