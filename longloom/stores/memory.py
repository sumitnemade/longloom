"""In-memory storage backend — the zero-dependency library default."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from .base import BaseContextStore
from ..chunks import ContextChunk

__all__ = ["InMemoryContextStore"]


class InMemoryContextStore(BaseContextStore):
    """Stores everything in a process-local dict. Data is lost on exit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sessions: Dict[str, List[ContextChunk]] = defaultdict(list)

    def _persist(self, chunks: List[ContextChunk]) -> None:
        for chunk in chunks:
            # Session id is the id prefix: "{session_id}_{index:04d}_{hash}".
            session_id = chunk.id.rsplit("_", 2)[0]
            bucket = self._sessions[session_id]
            # Upsert by (session, chunk_index).
            replaced = False
            for i, existing in enumerate(bucket):
                if existing.chunk_index == chunk.chunk_index:
                    bucket[i] = chunk
                    replaced = True
                    break
            if not replaced:
                bucket.append(chunk)
            bucket.sort(key=lambda c: c.chunk_index)

    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        chunks = list(self._sessions.get(session_id, []))
        if chunk_indices is not None:
            wanted = set(chunk_indices)
            chunks = [c for c in chunks if c.chunk_index in wanted]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def _delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def search_context(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ContextChunk]:
        q = query.lower()
        hits: List[ContextChunk] = []
        for chunk in self._load_chunks(session_id):
            if q in chunk.content.lower():
                hits.append(chunk)
                if len(hits) >= limit:
                    break
        return hits