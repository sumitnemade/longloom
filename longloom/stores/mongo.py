"""MongoDB storage backend (pymongo, optional dependency).

    pip install pymongo

Embedding in an existing database::

    store = MongoContextStore(
        "mongodb://host:27017", db_name="myapp",
        collection="app_chunks",                      # your collection
        columns={"content": "body", "session_id": "sess"},  # your field names
        namespace="support-agent",                    # share a collection across apps
    )
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from .base import BaseContextStore
from .naming import FieldMap
from ..chunks import ContextChunk

__all__ = ["MongoContextStore"]

#: Mongo keeps ``_id`` fixed; these logical fields can be remapped.
MAPPABLE = ("content", "content_hash", "metadata", "token_count",
            "created_at", "chunk_index", "total_chunks", "session_id")


class MongoContextStore(BaseContextStore):
    """One document per chunk (customizable field names + namespace)."""

    def __init__(
        self,
        client_or_uri: Union[Any, str],
        db_name: str = "longloom",
        collection: str = "context_chunks",
        columns: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
        namespace_field: str = "namespace",
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
        self.names = FieldMap(
            table=collection,
            columns={k: v for k, v in (columns or {}).items() if k in MAPPABLE},
            namespace=namespace,
            namespace_field=namespace_field,
        )
        self.collection = self.client[db_name][self.names.table]
        try:
            self.collection.create_index(
                [(self.names.col("session_id"), 1), (self.names.col("chunk_index"), 1)])
        except Exception:
            pass  # read-only/Atlas-restricted contexts still work for CRUD

    # -- (de)serialization ---------------------------------------------------

    def _to_doc(self, chunk: ContextChunk) -> Dict[str, Any]:
        c = self.names.col
        doc: Dict[str, Any] = {
            "_id": chunk.id,
            c("session_id"): chunk.id.rsplit("_", 2)[0],
            c("content"): chunk.content,
            c("metadata"): chunk.metadata,
            c("token_count"): chunk.token_count,
            c("created_at"): chunk.created_at,
            c("chunk_index"): chunk.chunk_index,
            c("total_chunks"): chunk.total_chunks,
        }
        if self.names.namespace is not None:
            doc[self.names.namespace_field] = self.names.namespace
        return doc

    def _from_doc(self, doc: Dict[str, Any]) -> ContextChunk:
        c = self.names.col
        return ContextChunk(
            id=doc["_id"],
            content=doc.get(c("content"), ""),
            metadata=doc.get(c("metadata"), {}),
            token_count=doc.get(c("token_count"), 0),
            created_at=doc.get(c("created_at"), ""),
            chunk_index=doc.get(c("chunk_index"), 0),
            total_chunks=doc.get(c("total_chunks"), 1),
        )

    def _query(self, session_id: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        q: Dict[str, Any] = {self.names.col("session_id"): session_id}
        if self.names.namespace is not None:
            q[self.names.namespace_field] = self.names.namespace
        if extra:
            q.update(extra)
        return q

    # -- backend primitives -------------------------------------------------

    def _persist(self, chunks: List[ContextChunk]) -> None:
        ops = [self._ReplaceOne({"_id": chunk.id}, self._to_doc(chunk), upsert=True)
               for chunk in chunks]
        if ops:
            self.collection.bulk_write(ops)

    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        extra: Optional[Dict[str, Any]] = None
        if chunk_indices is not None:
            if not chunk_indices:
                return []
            extra = {self.names.col("chunk_index"): {"$in": list(chunk_indices)}}
        docs = self.collection.find(self._query(session_id, extra)).sort(
            self.names.col("chunk_index"), 1)
        return [self._from_doc(doc) for doc in docs]

    def _delete_session(self, session_id: str) -> None:
        self.collection.delete_many(self._query(session_id))

    # -- search / info --------------------------------------------------------

    def search_context(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ContextChunk]:
        pattern = re.escape(query)  # substring match, regex-escaped server-side
        docs = self.collection.find(
            self._query(session_id, {self.names.col("content"): {
                "$regex": pattern, "$options": "i"}})
        ).sort(self.names.col("chunk_index"), 1).limit(limit)
        return [self._from_doc(doc) for doc in docs]

    def get_session_info(self, session_id: str) -> dict:
        c = self.names.col
        pipeline = [
            {"$match": self._query(session_id)},
            {"$group": {
                "_id": None,
                "chunk_count": {"$sum": 1},
                "total_tokens": {"$sum": f"${c('token_count')}"},
                "first_chunk_at": {"$min": f"${c('created_at')}"},
                "last_chunk_at": {"$max": f"${c('created_at')}"},
            }},
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
