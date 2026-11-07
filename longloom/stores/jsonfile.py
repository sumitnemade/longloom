"""JSONL file storage backend — one human-readable file per session.

Proves the storage contract is truly pluggable: no SQL, no files-for-content
splitting — just a plain JSON-Lines document per session under ``root_dir``.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from .base import BaseContextStore
from ..chunks import ContextChunk

__all__ = ["JSONFileContextStore"]


class JSONFileContextStore(BaseContextStore):
    def __init__(self, root_dir: str = "longloom_data", **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)

    # -- persistence ---------------------------------------------------

    def _session_path(self, session_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)
        return os.path.join(self.root_dir, f"{safe}.jsonl")

    def _persist(self, chunks: List[ContextChunk]) -> None:
        if not chunks:
            return
        session_id = chunks[0].id.rsplit("_", 2)[0]
        path = self._session_path(session_id)

        existing = {
            c.chunk_index: c for c in self._load_chunks(session_id)
        }
        for chunk in chunks:
            existing[chunk.chunk_index] = chunk

        ordered = sorted(existing.values(), key=lambda c: c.chunk_index)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for chunk in ordered:
                fh.write(
                    json.dumps(
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "token_count": chunk.token_count,
                            "created_at": chunk.created_at,
                            "chunk_index": chunk.chunk_index,
                            "total_chunks": chunk.total_chunks,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        os.replace(tmp, path)

    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return []
        wanted = set(chunk_indices) if chunk_indices is not None else None
        chunks: List[ContextChunk] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if wanted is not None and rec["chunk_index"] not in wanted:
                    continue
                chunks.append(
                    ContextChunk(
                        id=rec["id"],
                        content=rec["content"],
                        metadata=rec["metadata"],
                        token_count=rec["token_count"],
                        created_at=rec["created_at"],
                        chunk_index=rec["chunk_index"],
                        total_chunks=rec["total_chunks"],
                    )
                )
        return sorted(chunks, key=lambda c: c.chunk_index)

    def _delete_session(self, session_id: str) -> None:
        path = self._session_path(session_id)
        if os.path.exists(path):
            os.remove(path)

    def list_sessions(self) -> List[str]:
        return [f[:-6] for f in os.listdir(self.root_dir) if f.endswith(".jsonl")]

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