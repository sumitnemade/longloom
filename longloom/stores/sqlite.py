"""SQLite storage backend — single-file persistence.

Two content modes:
* ``content_dir=None`` (default): chunk text is stored inline in the DB —
  fully portable single file, no sidecar files.
* ``content_dir=<path>``: chunk text is written to hash-addressed files and
  the DB keeps only metadata (the original project's layout, good when text
  blobs are large and you want them greppable on disk).
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from .base import BaseContextStore
from ..chunks import ContextChunk

__all__ = ["SQLiteContextStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_chunks (
    id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    metadata TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    content TEXT
);
CREATE INDEX IF NOT EXISTS idx_session_id ON context_chunks(session_id);
CREATE INDEX IF NOT EXISTS idx_content_hash ON context_chunks(content_hash);
"""


class SQLiteContextStore(BaseContextStore):
    def __init__(
        self,
        db_path: str = "longloom.db",
        content_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.db_path = db_path
        self.content_dir = content_dir
        if content_dir:
            os.makedirs(content_dir, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- persistence ---------------------------------------------------

    def _persist(self, chunks: List[ContextChunk]) -> None:
        for chunk in chunks:
            content = chunk.content
            if self.content_dir:
                path = self._content_path(chunk.metadata["content_hash"])
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(chunk.content)
                content = None  # keep DB as metadata-only
            self._conn.execute(
                """
                INSERT OR REPLACE INTO context_chunks
                (id, content_hash, metadata, token_count, created_at,
                 chunk_index, total_chunks, session_id, content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.metadata["content_hash"],
                    json.dumps(chunk.metadata),
                    chunk.token_count,
                    chunk.created_at,
                    chunk.chunk_index,
                    chunk.total_chunks,
                    self._session_of(chunk.id),
                    content,
                ),
            )
        self._conn.commit()

    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        query = """
            SELECT id, content_hash, metadata, token_count, created_at,
                   chunk_index, total_chunks, content
            FROM context_chunks WHERE session_id = ?
        """
        params: List[Any] = [session_id]
        if chunk_indices is not None:
            if not chunk_indices:
                return []
            marks = ",".join("?" * len(chunk_indices))
            query += f" AND chunk_index IN ({marks})"
            params += list(chunk_indices)
        query += " ORDER BY chunk_index"

        rows = self._conn.execute(query, params).fetchall()
        chunks: List[ContextChunk] = []
        for row in rows:
            (
                cid,
                content_hash,
                metadata_json,
                token_count,
                created_at,
                chunk_index,
                total_chunks,
                content,
            ) = row
            if content is None:  # file-backed mode
                path = self._content_path(content_hash)
                content = ""
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as fh:
                        content = fh.read()
            chunks.append(
                ContextChunk(
                    id=cid,
                    content=content,
                    metadata=json.loads(metadata_json),
                    token_count=token_count,
                    created_at=created_at,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
            )
        return chunks

    def _delete_session(self, session_id: str) -> None:
        # Remove sidecar files first (file-backed mode).
        rows = self._conn.execute(
            "SELECT content_hash FROM context_chunks WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        for (content_hash,) in rows:
            path = self._content_path(content_hash)
            if os.path.exists(path):
                os.remove(path)
        self._conn.execute(
            "DELETE FROM context_chunks WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()

    # -- search ---------------------------------------------------------

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

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        row = self._conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(token_count), 0),
                   MIN(created_at), MAX(created_at)
            FROM context_chunks WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        count, tokens, first, last = row
        if count == 0:
            return {"session_id": session_id, "chunk_count": 0, "total_tokens": 0}
        return {
            "session_id": session_id,
            "chunk_count": count,
            "total_tokens": tokens,
            "first_chunk_at": first,
            "last_chunk_at": last,
        }

    def close(self) -> None:
        self._conn.close()

    # -- helpers ---------------------------------------------------------

    def _content_path(self, content_hash: str) -> str:
        return os.path.join(self.content_dir or "", f"{content_hash}.txt")

    @staticmethod
    def _session_of(chunk_id: str) -> str:
        return chunk_id.rsplit("_", 2)[0]