"""End-to-end index and query orchestration."""

from __future__ import annotations

from pathlib import Path

from note_rag.chunking import chunk_markdown_tree
from note_rag.embedder import embed_texts
from note_rag.llm import chat
from note_rag.store import (
    COLLECTION_NAME_DEFAULT,
    get_client,
    parse_query_result,
    query_collection,
    reset_collection,
    upsert_chunks,
)


def build_index(
    corpus_dir: Path,
    *,
    persist_path: str,
    collection_name: str = COLLECTION_NAME_DEFAULT,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_chars: int = 2000,
) -> int:
    chunks = chunk_markdown_tree(corpus_dir, max_chars=max_chars)
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model_name=embed_model)
    client = get_client(persist_path)
    collection = reset_collection(client, collection_name)
    upsert_chunks(collection, chunks, embeddings)
    return len(chunks)


def retrieve(
    question: str,
    *,
    persist_path: str,
    collection_name: str = COLLECTION_NAME_DEFAULT,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 5,
) -> list[dict]:
    client = get_client(persist_path)
    collection = client.get_collection(name=collection_name)
    q_emb = embed_texts([question], model_name=embed_model)[0]
    raw = query_collection(collection, q_emb, k)
    return parse_query_result(raw)


def format_context_blocks(hits: list[dict]) -> tuple[str, list[dict]]:
    """Numbered context for the prompt and a parallel source list for printing."""
    blocks: list[str] = []
    sources: list[dict] = []
    for i, h in enumerate(hits, start=1):
        src = h["source_path"]
        section = h.get("heading_slug") or " > ".join(h.get("heading_path") or [])
        blocks.append(f"[{i}] Source: {src} | Section: {section}\n{h['document']}")
        sources.append(
            {
                "ref": i,
                "chunk_id": h["chunk_id"],
                "source_path": src,
                "section": section,
                "distance": h.get("distance"),
            }
        )
    return "\n\n---\n\n".join(blocks), sources


def answer_from_hits(
    question: str,
    hits: list[dict],
    *,
    ollama_model: str = "llama3.2",
    ollama_host: str | None = None,
) -> tuple[str, list[dict]]:
    """Call Ollama using numbered context built from retrieval hits."""
    context, sources = format_context_blocks(hits)
    system = (
        "You are a careful assistant. Answer ONLY using the provided context blocks. "
        "Each block is labeled [n]. For every factual claim, cite the bracket id(s) "
        "like [1] or [1][2]. If the context is insufficient, say you cannot answer "
        "from the context and explain what is missing."
    )
    user = f"Context:\n\n{context}\n\nQuestion: {question}\n\nAnswer with citations."
    text = chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=ollama_model,
        base_url=ollama_host,
    )
    return text, sources


def answer_question(
    question: str,
    *,
    persist_path: str,
    collection_name: str = COLLECTION_NAME_DEFAULT,
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    k: int = 5,
    ollama_model: str = "llama3.2",
    ollama_host: str | None = None,
) -> tuple[str, list[dict]]:
    hits = retrieve(
        question,
        persist_path=persist_path,
        collection_name=collection_name,
        embed_model=embed_model,
        k=k,
    )
    if not hits:
        return (
            "No indexed chunks found. Run `note-rag index <corpus>` first.",
            [],
        )
    return answer_from_hits(
        question,
        hits,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )


def validate_citation_refs(answer: str, k: int) -> list[str]:
    """Warn if answer cites [n] outside 1..k."""
    import re

    refs = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    bad = sorted(n for n in refs if n < 1 or n > k)
    if not bad:
        return []
    return [f"Citation id(s) out of retrieved range 1..{k}: {bad}"]
