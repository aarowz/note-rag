"""Tests for Markdown chunking."""

from __future__ import annotations

from pathlib import Path

from note_rag.chunking import chunk_markdown_file, chunk_markdown_tree


def test_h1_intro_and_h2_sections(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text(
        "# Doc Title\n\nIntro line here.\n\n## Section A\n\nAlpha body.\n\n## Section B\n\nBeta body.\n",
        encoding="utf-8",
    )
    chunks = chunk_markdown_file(p, tmp_path)
    assert len(chunks) == 3
    assert chunks[0]["heading_path"] == ["Doc Title"]
    assert "Intro line" in chunks[0]["text"]
    assert chunks[1]["heading_path"] == ["Doc Title", "Section A"]
    assert "Alpha body" in chunks[1]["text"]
    assert chunks[2]["heading_path"] == ["Doc Title", "Section B"]
    assert "Beta body" in chunks[2]["text"]


def test_preamble_then_h2(tmp_path: Path) -> None:
    p = tmp_path / "pre.md"
    p.write_text("Preamble only.\n\n## First\n\nInside.\n", encoding="utf-8")
    chunks = chunk_markdown_file(p, tmp_path)
    assert chunks[0]["heading_path"] == ["Preamble"]
    assert "Preamble only" in chunks[0]["text"]
    assert chunks[1]["heading_path"] == ["First"]
    assert "Inside" in chunks[1]["text"]


def test_long_section_splits(tmp_path: Path) -> None:
    p = tmp_path / "long.md"
    body = "word " * 1500  # >> default max_chars in test
    p.write_text(f"# T\n\n## Big\n\n{body}\n", encoding="utf-8")
    chunks = chunk_markdown_file(p, tmp_path, max_chars=500)
    big_chunks = [c for c in chunks if c["heading_path"] == ["T", "Big"]]
    assert len(big_chunks) >= 2
    assert big_chunks[0]["part_index"] == 0
    assert big_chunks[1]["part_index"] == 1


def test_chunk_markdown_tree_finds_md(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# A\n\n## X\n\nx.\n", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("## Only\n\ny.\n", encoding="utf-8")
    chunks = chunk_markdown_tree(tmp_path)
    sources = {c["source_path"] for c in chunks}
    assert "a.md" in sources
    assert "sub/b.md" in sources
