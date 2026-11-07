"""Storage abstraction for long-context chunks.

To plug in your own backend (Redis, Postgres, S3, ...), subclass
:class:`BaseContextStore` and implement four primitives:

* ``_persist(chunks)``       — write chunk objects
* ``_load_chunks(...)``      — read chunk objects back (ordered by chunk_index)
* ``_delete_session(...)``   — remove all chunks of a session
* ``search_context(...)``    — backend-appropriate search (substring is fine)

``store_context``, ``retrieve_context``, and ``get_session_info`` are template
methods shared by every backend, so chunking/id/hashing semantics stay
identical across storage engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..chunks import ContextChunk, content_hash, split_text, utc_now_iso
from ..tokens import TokenCounter, default_counter

__all__ = ["BaseContextStore"]


class BaseContextStore(ABC):
    """Common contract for all storage backends."""

    def __init__(self, token_counter: Optional[TokenCounter] = None):
        self.counter: TokenCounter = token_counter or default_counter()

    # ------------------------------------------------------------------
    # Template methods (shared across backends)
    # ------------------------------------------------------------------

    def store_context(
        self,
        content: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_chunk_tokens: int = 3000,
    ) -> List[ContextChunk]:
        """Split *content* into token-bounded chunks and persist them."""
        if not content:
            raise ValueError("content must be a non-empty string")
        if not session_id:
            raise ValueError("session_id must be a non-empty string")

        pieces = split_text(content, max_chunk_tokens, self.counter)
        total = len(pieces)
        base_meta = dict(metadata or {})

        chunks: List[ContextChunk] = []
        for i, piece in enumerate(pieces):
            h = content_hash(piece)
            chunks.append(
                ContextChunk(
                    id=f"{session_id}_{i:04d}_{h}",
                    content=piece,
                    metadata={
                        **base_meta,
                        "chunk_index": i,
                        "total_chunks": total,
                        "content_hash": h,
                    },
                    token_count=self.counter(piece),
                    created_at=utc_now_iso(),
                    chunk_index=i,
                    total_chunks=total,
                )
            )
        self._persist(chunks)
        return chunks

    def retrieve_context(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        """Load chunks in order; stop before exceeding *max_tokens*."""
        chunks = self._load_chunks(session_id, chunk_indices=chunk_indices)
        if max_tokens is None:
            return chunks

        out: List[ContextChunk] = []
        budget = 0
        for chunk in chunks:
            if budget + chunk.token_count > max_tokens:
                break
            out.append(chunk)
            budget += chunk.token_count
        return out

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Aggregate stats for a session (backends may override for speed)."""
        chunks = self._load_chunks(session_id)
        if not chunks:
            return {"session_id": session_id, "chunk_count": 0, "total_tokens": 0}
        return {
            "session_id": session_id,
            "chunk_count": len(chunks),
            "total_tokens": sum(c.token_count for c in chunks),
            "first_chunk_at": min(c.created_at for c in chunks),
            "last_chunk_at": max(c.created_at for c in chunks),
        }

    def delete_session(self, session_id: str) -> None:
        """Remove every chunk belonging to a session."""
        self._delete_session(session_id)

    # ------------------------------------------------------------------
    # Backend primitives
    # ------------------------------------------------------------------

    @abstractmethod
    def _persist(self, chunks: List[ContextChunk]) -> None:
        """Write the given chunks (upsert semantics)."""

    @abstractmethod
    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        """Return chunks for a session ordered by chunk_index."""

    @abstractmethod
    def _delete_session(self, session_id: str) -> None:
        """Remove every chunk belonging to a session."""

    @abstractmethod
    def search_context(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ContextChunk]:
        """Find chunks in a session matching *query* (case-insensitive)."""