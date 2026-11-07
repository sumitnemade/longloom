"""Supabase storage backend (supabase-py, optional dependency).

    pip install supabase

Create the table once in the Supabase SQL editor — the DDL adapts to your
naming::

    from longloom.stores.supabase import table_sql
    print(table_sql())                                   # default names
    print(table_sql("app_chunks", {"content": "body"}))  # your names

Embedding in an existing project::

    store = SupabaseContextStore(
        url="https://<project>.supabase.co",
        key="<service-or-anon-key>",
        table="app_chunks",                       # your table
        columns={"id": "chunk_id", "content": "body"},  # your columns
        namespace="support-agent",                # share a table across apps
    )
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import BaseContextStore
from .naming import FieldMap
from ..chunks import ContextChunk

__all__ = ["SupabaseContextStore", "table_sql"]


def table_sql(table: str = "context_chunks", columns: Optional[Dict[str, str]] = None,
              namespace: Optional[str] = None) -> str:
    """DDL to create the (custom-named) chunks table in a Supabase/Postgres project."""
    names = FieldMap(table=table, columns=columns or {}, namespace=namespace)
    c = names.all_columns
    defs = [
        f"{c['id']} TEXT PRIMARY KEY",
        f"{c['content_hash']} TEXT NOT NULL",
        f"{c['metadata']} JSONB NOT NULL",
        f"{c['token_count']} INTEGER NOT NULL",
        f"{c['created_at']} TEXT NOT NULL",
        f"{c['chunk_index']} INTEGER NOT NULL",
        f"{c['total_chunks']} INTEGER NOT NULL",
        f"{c['session_id']} TEXT NOT NULL",
        f"{c['content']} TEXT",
    ]
    if namespace is not None:
        defs.append(f"{c['namespace']} TEXT NOT NULL DEFAULT ''")
    return (
        f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(defs)});\n"
        f"CREATE INDEX IF NOT EXISTS idx_{table}_session "
        f"ON {table} ({c['session_id']}, {c['chunk_index']});"
    )


class SupabaseContextStore(BaseContextStore):
    """REST-based backend via supabase-py — no direct DB connection needed."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None,
                 table: str = "context_chunks", columns: Optional[Dict[str, str]] = None,
                 namespace: Optional[str] = None, namespace_field: str = "namespace",
                 **kwargs):
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
        self.names = FieldMap(table=table, columns=columns or {},
                              namespace=namespace, namespace_field=namespace_field)
        self.table = self.names.table

    # -- row (de)serialization ------------------------------------------------

    def _to_row(self, chunk: ContextChunk) -> Dict[str, Any]:
        c = self.names.col
        row: Dict[str, Any] = {
            c("id"): chunk.id,
            c("content_hash"): chunk.metadata["content_hash"],
            c("metadata"): chunk.metadata,       # JSONB — client serializes
            c("token_count"): chunk.token_count,
            c("created_at"): chunk.created_at,
            c("chunk_index"): chunk.chunk_index,
            c("total_chunks"): chunk.total_chunks,
            c("session_id"): chunk.id.rsplit("_", 2)[0],
            c("content"): chunk.content,
        }
        if self.names.namespace is not None:
            row[c("namespace")] = self.names.namespace
        return row

    def _from_row(self, row: Dict[str, Any]) -> ContextChunk:
        c = self.names.col
        return ContextChunk(
            id=row[c("id")],
            content=row.get(c("content")) or "",
            metadata=row.get(c("metadata")) or {},
            token_count=row.get(c("token_count"), 0),
            created_at=row.get(c("created_at"), ""),
            chunk_index=row.get(c("chunk_index"), 0),
            total_chunks=row.get(c("total_chunks"), 1),
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
        c = self.names.col
        q = self.client.table(self.table).select("*").eq(c("session_id"), session_id)
        if self.names.namespace is not None:
            q = q.eq(c("namespace"), self.names.namespace)
        if chunk_indices is not None:
            if not chunk_indices:
                return []
            q = q.in_(c("chunk_index"), list(chunk_indices))
        data = q.order(c("chunk_index")).execute().data or []
        return [self._from_row(r) for r in data]

    def _delete_session(self, session_id: str) -> None:
        c = self.names.col
        q = self.client.table(self.table).delete().eq(c("session_id"), session_id)
        if self.names.namespace is not None:
            q = q.eq(c("namespace"), self.names.namespace)
        q.execute()

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
        c = self.names.col
        q = (self.client.table(self.table)
             .select(f"{c('token_count')},{c('created_at')}")
             .eq(c("session_id"), session_id))
        if self.names.namespace is not None:
            q = q.eq(c("namespace"), self.names.namespace)
        rows = q.execute().data or []
        if not rows:
            return {"session_id": session_id, "chunk_count": 0, "total_tokens": 0}
        return {
            "session_id": session_id,
            "chunk_count": len(rows),
            "total_tokens": sum(r.get(c("token_count"), 0) for r in rows),
            "first_chunk_at": min(r.get(c("created_at"), "") for r in rows),
            "last_chunk_at": max(r.get(c("created_at"), "") for r in rows),
        }
