"""MongoDB storage backend (pymongo, optional dependency).

    pip install pymongo
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Union

from .base import BaseContextStore
from ..chunks import ContextChunk

__all__ = ["MongoContextStore"]


class MongoContextStore(BaseContextStore):
    """One document per chunk: {_id, session_id, chunk_index, content, ...}."""

    def __init__(
        self,
        client_or_uri: Union[Any, str],
        db_name: str = "longloom",
        collection: str = "context_chunks",
        **kwargs,
    ):
        super().__init__(**kwargs)
        try:
            from pymongo import MongoClient, ReplaceOne
        except ImportError as exc:
            raise ImportError(
                "pymongo is required for the MongoDB backend: pip install pymongo"
            ) from exc
        self._ReplaceOne = ReplaceOne
        if isinstance(client_or_uri, str):
            client_or_uri = MongoClient(client_or_uri)
        self.client = client_or_uri
        self.collection = self.client[db_name][collection]
        try:
            self.collection.create_index([("session_id", 1), ("chunk_index", 1)])
        except Exception:
            pass  # read-only/Atlas-restricted contexts still work for CRUD

    # -- backend primitives -------------------------------------------------

    def _persist(self, chunks: List[ContextChunk]) -> None:
        ops = []
        for chunk in chunks:
            ops.append(
                self._ReplaceOne(
                    {"_id": chunk.id},
                    {
                        "_id": chunk.id,
                        "session_id": chunk.id.rsplit("_", 2)[0],
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "token_count": chunk.token_count,
                        "created_at": chunk.created_at,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                    },
                    upsert=True,
                )
            )
        if ops:
            self.collection.bulk_write(ops)

    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        query: dict = {"session_id": session_id}
        if chunk_indices is not None:
            if not chunk_indices:
                return []
            query["chunk_index"] = {"$in": list(chunk_indices)}
        docs = self.collection.find(query).sort("chunk_index", 1)
        return [
            ContextChunk(
                id=doc["_id"],
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                token_count=doc.get("token_count", 0),
                created_at=doc.get("created_at", ""),
                chunk_index=doc.get("chunk_index", 0),
                total_chunks=doc.get("total_chunks", 1),
            )
            for doc in docs
        ]

    def _delete_session(self, session_id: str) -> None:
        self.collection.delete_many({"session_id": session_id})

    # -- search / info --------------------------------------------------------

    def search_context(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ContextChunk]:
        pattern = re.escape(query)  # substring match, regex-escaped server-side
        docs = self.collection.find(
            {
                "session_id": session_id,
                "content": {"$regex": pattern, "$options": "i"},
            }
        ).sort("chunk_index", 1).limit(limit)
        return [
            ContextChunk(
                id=doc["_id"],
                content=doc.get("content", ""),
                metadata=doc.get("metadata", {}),
                token_count=doc.get("token_count", 0),
                created_at=doc.get("created_at", ""),
                chunk_index=doc.get("chunk_index", 0),
                total_chunks=doc.get("total_chunks", 1),
            )
            for doc in docs
        ]

    def get_session_info(self, session_id: str) -> dict:
        pipeline = [
            {"$match": {"session_id": session_id}},
            {
                "$group": {
                    "_id": None,
                    "chunk_count": {"$sum": 1},
                    "total_tokens": {"$sum": "$token_count"},
                    "first_chunk_at": {"$min": "$created_at"},
                    "last_chunk_at": {"$max": "$created_at"},
                }
            },
        ]
        rows = list(self.collection.aggregate(pipeline))
        if not rows:
            return {"session_id": session_id, "chunk_count": 0, "total_tokens": 0}
        info = rows[0]
        return {
            "session_id": session_id,
            "chunk_count": info["chunk_count"],
            "total_tokens": info["total_tokens"],
            "first_chunk_at": info.get("first_chunk_at", ""),
            "last_chunk_at": info.get("last_chunk_at", ""),
        }

    def close(self) -> None:
        self.client.close()