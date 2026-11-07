"""Pluggable token counting.

The library never hardcodes a tokenizer. Two built-in counters are provided:

* :class:`TiktokenCounter` — exact for OpenAI-style BPE; used when ``tiktoken``
  is installed. Note it only *approximates* provider tokenizers (Nemotron,
  LLaMA, Claude...) — chunk with headroom against the real context window.
* :class:`EstimateCounter` — chars-per-token heuristic, zero dependencies.

``default_counter()`` picks tiktoken when available, otherwise falls back to
the estimator, so the library works out of the box either way.
"""
from __future__ import annotations

import math
from typing import List, Protocol, runtime_checkable

__all__ = ["TokenCounter", "TiktokenCounter", "EstimateCounter", "default_counter"]


@runtime_checkable
class TokenCounter(Protocol):
    """Anything callable returning the token count of a string."""

    def __call__(self, text: str) -> int: ...


class TiktokenCounter:
    """Exact token counting via tiktoken (fast encode/decode split path)."""

    def __init__(self, encoding: str = "cl100k_base"):
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "tiktoken is not installed. Install it with "
                "'pip install tiktoken' or use EstimateCounter instead."
            ) from exc
        self._enc = tiktoken.get_encoding(encoding)

    def __call__(self, text: str) -> int:
        return len(self._enc.encode(text))

    def split(self, text: str, max_tokens: int) -> List[str]:
        """Exact tokenizer-aware split (fast path used by split_text)."""
        tokens = self._enc.encode(text)
        return [self._enc.decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


class EstimateCounter:
    """Dependency-free chars-per-token estimator."""

    def __init__(self, chars_per_token: float = 4.0):
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be > 0")
        self.chars_per_token = chars_per_token

    def __call__(self, text: str) -> int:
        return max(1, math.ceil(len(text) / self.chars_per_token))


def default_counter() -> TokenCounter:
    """tiktoken when installed, else the zero-dependency estimator."""
    try:
        return TiktokenCounter()  # type: ignore[return-value]
    except ImportError:
        return EstimateCounter()  # type: ignore[return-value]