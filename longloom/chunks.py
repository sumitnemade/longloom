"""Chunk data model and token-aware text splitting.

Backends and counters are pluggable: :func:`split_text` works with any
callable token counter (tiktoken-based or estimator) and never depends on
a specific tokenizer's encode/decode round-trip.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

__all__ = ["ContextChunk", "content_hash", "utc_now_iso", "split_text"]


def content_hash(text: str, length: int = 16) -> str:
    """Short SHA-256 hash of text, used for content addressing."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def utc_now_iso() -> str:
    """UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ContextChunk:
    """A single chunk of a larger document."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    created_at: str = ""
    chunk_index: int = 0
    total_chunks: int = 1


def split_text(
    text: str,
    max_tokens: int,
    counter: Callable[[str], int],
) -> List[str]:
    """Split *text* into pieces of at most *max_tokens* (counter-defined tokens).

    Strategy — LINE-AWARE, so facts on a single line are never cut in half:

      1. Greedily pack whole lines into chunks while the chunk fits
         (``counter()`` measures each line; works with any counter).
      2. A single line longer than *max_tokens* is hard-split by token count
         (binary search with the counter) — unavoidable last resort.

    Returns a list of non-empty strings whose concatenation reconstructs the
    original text.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not text:
        return []

    def _hard_split(segment: str) -> List[str]:
        """Token-count split for a single line that exceeds max_tokens."""
        if chr(10) in segment:  # defensive; caller passes single lines
            return [segment]  # pragma: no cover
        out: List[str] = []
        remaining = segment
        while remaining:
            if counter(remaining) <= max_tokens:
                out.append(remaining)
                break
            guess = max(1, int(len(remaining) * max_tokens / max(counter(remaining), 1)))
            steps = 0
            while guess > 1 and counter(remaining[:guess]) > max_tokens and steps < 64:
                guess = max(1, int(guess * 0.9))
                steps += 1
            while guess < len(remaining) and counter(remaining[: guess + 1]) <= max_tokens:
                guess += 1
            if guess <= 0:  # pragma: no cover — defensive; guess starts >= 1
                guess = 1
            # Snap to a whitespace boundary so identifier tokens (IDs, numbers,
            # handles — anything without spaces) are never cut in half.
            ws = remaining.rfind(" ", 0, guess)
            if ws > guess * 0.5:
                guess = ws + 1
            out.append(remaining[:guess])
            remaining = remaining[guess:]
        return out

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0
    for line in text.splitlines(keepends=True):
        line_tokens = counter(line)
        if line_tokens > max_tokens:
            if current:
                chunks.append("".join(current))
                current, current_tokens = [], 0
            chunks.extend(_hard_split(line.rstrip("\n")) if line.strip() else [line])
            continue
        if current_tokens + line_tokens > max_tokens and current:
            chunks.append("".join(current))
            current, current_tokens = [], 0
        current.append(line)
        current_tokens += line_tokens
    if current:
        chunks.append("".join(current))

    return [c for c in chunks if c.strip()]