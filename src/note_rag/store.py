"""Chroma persistent vector store."""

from __future__ import annotations

import json
from typing import Any

import chromadb
from chromadb.api.types import QueryResult


COLLECTION_NAME_DEFAULT = "note_rag_md"


def get_client(persist_path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_path)


def reset_collection(client: chromadb.PersistentClient, name: str) -> Any:
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def upsert_chunks(
    collection: Any,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> None:
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        metadatas.append(
            {
                "source_path": c["source_path"],
                "heading_path_json": json.dumps(c["heading_path"], ensure_ascii=False),
                "heading_slug": c["heading_slug"],
                "part_index": int(c["part_index"]),
            }
        )
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_collection(
    collection: Any,
    query_embedding: list[float],
    k: int,
) -> QueryResult:
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )


def parse_query_result(result: QueryResult) -> list[dict[str, Any]]:
    """Normalize Chroma query result to a list of hit dicts."""
    if not result["ids"] or not result["ids"][0]:
        return []
    hits: list[dict[str, Any]] = []
    ids = result["ids"][0]
    dists = (result.get("distances") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    for i, cid in enumerate(ids):
        meta = dict(metas[i]) if metas[i] else {}
        hp = json.loads(meta.get("heading_path_json") or "[]")
        hits.append(
            {
                "chunk_id": cid,
                "distance": float(dists[i]) if dists is not None and i < len(dists) else None,
                "document": docs[i] if i < len(docs) else "",
                "source_path": meta.get("source_path", ""),
                "heading_path": hp,
                "heading_slug": meta.get("heading_slug", ""),
                "part_index": int(meta.get("part_index", 0)),
            }
        )
    return hits
