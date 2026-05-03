"""Local sentence-transformers embedding wrapper."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: list[str], *, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> list[list[float]]:
    """Return embeddings for each text (same order)."""
    if not texts:
        return []
    model = _model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 32,
    )
    return [v.tolist() for v in vectors]
