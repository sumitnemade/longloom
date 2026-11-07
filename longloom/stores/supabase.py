"""Supabase storage backend (supabase-py, optional dependency).

    pip install supabase

Create the table once in the Supabase SQL editor (or via ``table_sql()``)::

    from longloom.stores.supabase import table_sql
    print(table_sql())          # CREATE TABLE context_chunks (...)

Usage::

    store = SupabaseContextStore(
        url="https://<project>.supabase.co",
        key="<service-or-anon-key>",     # or set SUPABASE_KEY env var
    )
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import BaseContextStore
from ..chunks import ContextChunk

__all__ = ["SupabaseContextStore", "table_sql"]

_COLS = (
    "id", "content_hash", "metadata", "token_count", "created_at",
    "chunk_index", "total_chunks", "session_id", "content",
)


def table_sql(table: str = "context_chunks") -> str:
    """DDL to create the chunks table in a Supabase/Postgres project."""
    return f"""CREATE TABLE IF NOT EXISTS {table} (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    metadata JSONB NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT
);
CREATE INDEX IF NOT EXISTS idx_{table}_session ON {table} (session_id, chunk_index);"""


class SupabaseContextStore(BaseContextStore):
    """REST-based backend via supabase-py — no direct DB connection needed."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 table: str = "context_chunks", **kwargs):
        super().__init__(**kwargs)
        url = url or os.environ.get("SUPABASE_URL")
        key = key or os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError(
                "Supabase URL and key required — pass url=/key= or set "
                "SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        try:
            from supabase import create_client
        except ImportError as exc:
            raise ImportError(
                "supabase-py is required for the Supabase backend: pip install supabase"
            ) from exc
        self.client = create_client(url, key)
        self.table = table

    # -- row (de)serialization ------------------------------------------------

    def _to_row(self, chunk: ContextChunk) -> Dict[str, Any]:
        return {
            "id": chunk.id,
            "content_hash": chunk.metadata["content_hash"],
            "metadata": chunk.metadata,          # JSONB — client serializes
            "token_count": chunk.token_count,
            "created_at": chunk.created_at,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "session_id": chunk.id.rsplit("_", 2)[0],
            "content": chunk.content,
        }

    def _from_row(self, row: Dict[str, Any]) -> ContextChunk:
        return ContextChunk(
            id=row["id"],
            content=row.get("content") or "",
            metadata=row.get("metadata") or {},
            token_count=row.get("token_count", 0),
            created_at=row.get("created_at", ""),
            chunk_index=row.get("chunk_index", 0),
            total_chunks=row.get("total_chunks", 1),
        )

    # -- backend primitives ------------------------------------------------------

    def _persist(self, chunks: List[ContextChunk]) -> None:
        if chunks:
            self.client.table(self.table).upsert([self._to_row(c) for c in chunks]).execute()

    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        q = self.client.table(self.table).select("*").eq("session_id", session_id)
        if chunk_indices is not None:
            if not chunk_indices:
                return []
            q = q.in_("chunk_index", list(chunk_indices))
        data = q.order("chunk_index").execute().data or []
        return [self._from_row(r) for r in data]

    def _delete_session(self, session_id: str) -> None:
        self.client.table(self.table).delete().eq("session_id", session_id).execute()

    # -- search / info --------------------------------------------------------------

    def search_context(self, session_id: str, query: str, limit: int = 10) -> List[ContextChunk]:
        q = query.lower()
        hits: List[ContextChunk] = []
        for chunk in self._load_chunks(session_id):
            if q in (chunk.content or "").lower():
                hits.append(chunk)
                if len(hits) >= limit:
                    break
        return hits

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        rows = (
            self.client.table(self.table)
            .select("token_count,created_at")
            .eq("session_id", session_id)
            .execute()
            .data or []
        )
        if not rows:
            return {"session_id": session_id, "chunk_count": 0, "total_tokens": 0}
        return {
            "session_id": session_id,
            "chunk_count": len(rows),
            "total_tokens": sum(r.get("token_count", 0) for r in rows),
            "first_chunk_at": min(r.get("created_at", "") for r in rows),
            "last_chunk_at": max(r.get("created_at", "") for r in rows),
        }