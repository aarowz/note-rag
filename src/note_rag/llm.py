"""Ollama HTTP chat client."""

from __future__ import annotations

import os
from typing import Any

import httpx


def chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    base_url: str | None = None,
    timeout: float = 120.0,
) -> str:
    url = (base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    endpoint = f"{url}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(endpoint, json=payload)
        r.raise_for_status()
        data = r.json()
    msg = data.get("message") or {}
    return (msg.get("content") or "").strip()
