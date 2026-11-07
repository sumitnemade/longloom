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

    Strategy:
      1. Fast path — if the counter exposes ``split(text, max_tokens)`` it is
         used directly (tiktoken counters do, via exact encode/decode).
      2. Fallback — analytic estimate of the cut point, then a bounded
         shrink/extend correction loop, snapping to whitespace where possible.

    Returns a list of non-empty strings whose concatenation reconstructs the
    original text (whitespace-stripped at chunk seams).
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if not text:
        return []

    fast = getattr(counter, "split", None)
    if callable(fast):
        return [c for c in fast(text, max_tokens) if c.strip()]

    chunks: List[str] = []
    remaining = text
    while remaining:
        total = counter(remaining)
        if total <= max_tokens:
            chunks.append(remaining)
            break

        # Analytic first guess from measured token density.
        guess = max(1, int(len(remaining) * max_tokens / max(total, 1)))
        # Shrink until the prefix fits (bounded, ~log-ish via 5% steps).
        steps = 0
        while guess > 1 and counter(remaining[:guess]) > max_tokens and steps < 64:
            guess = max(1, int(guess * 0.9))
            steps += 1
        # Extend while it still fits.
        while guess < len(remaining) and counter(remaining[: guess + 1]) <= max_tokens:
            guess += 1

        cut = guess
        # Snap to a whitespace boundary, but never backtrack too far.
        ws = remaining.rfind(" ", 0, cut)
        if ws > cut * 3 // 4:
            cut = ws + 1

        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    return [c for c in chunks if c.strip()]