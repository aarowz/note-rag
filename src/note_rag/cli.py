"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer

from note_rag.pipeline import answer_from_hits, build_index, retrieve, validate_citation_refs
from note_rag.store import COLLECTION_NAME_DEFAULT

app = typer.Typer(no_args_is_help=True, help="Local Markdown RAG: index corpus, ask with citations.")


@app.command("index")
def index_command(
    corpus: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    db: Path = typer.Option(Path(".chroma"), "--db", help="Chroma persistence directory."),
    collection: str = typer.Option(COLLECTION_NAME_DEFAULT, "--collection"),
    embed_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embed-model",
    ),
    max_chars: int = typer.Option(2000, "--max-chars", min=256),
) -> None:
    """Chunk and embed all *.md under CORPUS (recursive), then rebuild the vector index."""
    n = build_index(
        corpus,
        persist_path=str(db.resolve()),
        collection_name=collection,
        embed_model=embed_model,
        max_chars=max_chars,
    )
    typer.echo(f"Indexed {n} chunk(s) into {db} (collection={collection}).")


@app.command("ask")
def ask_command(
    question: str = typer.Argument(..., metavar="QUESTION"),
    db: Path = typer.Option(Path(".chroma"), "--db"),
    collection: str = typer.Option(COLLECTION_NAME_DEFAULT, "--collection"),
    embed_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embed-model",
    ),
    k: int = typer.Option(5, "--k", min=1, max=50),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print top-k chunks only (no LLM)."),
    model: str = typer.Option("llama3.2", "--model", help="Ollama chat model name."),
    ollama_host: str | None = typer.Option(
        None,
        "--ollama-host",
        help="Ollama base URL (default: env OLLAMA_HOST or http://127.0.0.1:11434).",
    ),
) -> None:
    """Retrieve context for QUESTION and answer with Ollama (unless --dry-run)."""
    hits = retrieve(
        question,
        persist_path=str(db.resolve()),
        collection_name=collection,
        embed_model=embed_model,
        k=k,
    )
    if not hits:
        typer.echo("No chunks retrieved. Index the corpus first: note-rag index <dir>")
        raise typer.Exit(code=1)

    if dry_run:
        typer.echo(typer.style("Retrieval (dry run)", bold=True))
        for h in hits:
            dist = h.get("distance")
            dist_s = f"{dist:.4f}" if dist is not None else "n/a"
            typer.echo(f"\n--- score(distance)={dist_s} id={h['chunk_id']}")
            typer.echo(f"source={h['source_path']} section={h.get('heading_slug')}")
            body = h["document"]
            typer.echo(body[:2000])
            if len(body) > 2000:
                typer.echo("…")
        raise typer.Exit(code=0)

    answer, sources = answer_from_hits(
        question,
        hits,
        ollama_model=model,
        ollama_host=ollama_host,
    )

    typer.echo(typer.style("Answer", bold=True))
    typer.echo(answer)

    for w in validate_citation_refs(answer, len(hits)):
        typer.echo(typer.style(w, fg=typer.colors.YELLOW))

    typer.echo(typer.style("\nSources", bold=True))
    for s in sources:
        dist = s.get("distance")
        dist_s = f"{dist:.4f}" if dist is not None else "n/a"
        typer.echo(
            f"  [{s['ref']}] {s['source_path']} — {s['section']} "
            f"(chunk={s['chunk_id'][:12]}… score={dist_s})"
        )


def main() -> None:
    app()
