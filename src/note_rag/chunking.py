"""Structure-aware Markdown chunking (## boundaries + max-length splits)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug_heading_path(heading_path: list[str], part_index: int) -> str:
    base = " > ".join(heading_path)
    if part_index > 0:
        return f"{base} (part {part_index + 1})"
    return base


def _chunk_id(source_path: str, heading_path: list[str], part_index: int) -> str:
    payload = json.dumps(
        {"source": source_path, "headings": heading_path, "part": part_index},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _split_body(body: str, max_chars: int) -> list[str]:
    body = body.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]
    parts: list[str] = []
    start = 0
    while start < len(body):
        parts.append(body[start : start + max_chars])
        start += max_chars
    return parts


def _emit_chunks_for_body(
    rel: str,
    heading_path: list[str],
    raw: str,
    *,
    max_chars: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, body in enumerate(_split_body(raw, max_chars)):
        out.append(
            {
                "chunk_id": _chunk_id(rel, heading_path, i),
                "source_path": rel,
                "heading_path": heading_path.copy(),
                "heading_slug": _slug_heading_path(heading_path, i),
                "part_index": i,
                "text": body,
            }
        )
    return out


def chunk_markdown_file(
    path: Path,
    corpus_root: Path,
    *,
    max_chars: int = 2000,
) -> list[dict[str, Any]]:
    """Split a Markdown file into chunks at ``##`` headings; sub-split long bodies."""
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(corpus_root).as_posix()
    lines = text.splitlines()

    h1: str | None = None
    h2: str | None = None
    buf: list[str] = []
    chunks: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buf
        raw = "\n".join(buf).strip()
        buf = []
        if not raw:
            return
        if h2 is not None:
            heading_path = [h1, h2] if h1 else [h2]
        elif h1 is not None:
            heading_path = [h1]
        else:
            heading_path = ["Preamble"]
        chunks.extend(_emit_chunks_for_body(rel, heading_path, raw, max_chars=max_chars))

    for line in lines:
        m = HEADING_RE.match(line.rstrip())
        if not m:
            buf.append(line)
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        if level == 1:
            flush()
            h1 = title
            h2 = None
        elif level == 2:
            flush()
            h2 = title
        else:
            buf.append(line)

    flush()
    return chunks


def chunk_markdown_tree(corpus_root: Path, *, max_chars: int = 2000) -> list[dict[str, Any]]:
    """Chunk all ``*.md`` files under corpus_root (recursive)."""
    root = corpus_root.resolve()
    all_chunks: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.md")):
        if path.is_file():
            all_chunks.extend(chunk_markdown_file(path, root, max_chars=max_chars))
    return all_chunks
