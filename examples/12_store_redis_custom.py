#!/usr/bin/env python3
"""12 — Write YOUR OWN storage backend: a real Redis store in ~40 lines.

Any backend just implements four primitives on BaseContextStore; chunking,
ids, hashing, token math and the whole workflow are inherited for free.

Requires: Redis on localhost:6379 and redis-py (pip install redis)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import redis

from _common import DemoLLM, SAMPLE_DOC, guard, show
from longloom import BaseContextStore, MapReduceProcessor


class RedisContextStore(BaseContextStore):
    """One Redis hash per session: field = chunk_index, value = chunk JSON."""

    def __init__(self, url: str = "redis://127.0.0.1:6379/0", prefix: str = "longloom", **kw):
        super().__init__(**kw)
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:session:{session_id}"

    # -- the four required primitives ------------------------------------

    def _persist(self, chunks) -> None:
        pipe = self.r.pipeline()
        for c in chunks:
            pipe.hset(self._key(c.id.rsplit("_", 2)[0]), str(c.chunk_index),
                      json.dumps({"id": c.id, "content": c.content,
                                  "metadata": c.metadata, "token_count": c.token_count,
                                  "created_at": c.created_at, "chunk_index": c.chunk_index,
                                  "total_chunks": c.total_chunks}))
        pipe.execute()

    def _load_chunks(self, session_id, chunk_indices=None):
        raw = self.r.hgetall(self._key(session_id))
        wanted = set(chunk_indices) if chunk_indices is not None else None
        chunks = [self._decode(v) for v in raw.values()]
        return sorted((c for c in chunks if wanted is None or c.chunk_index in wanted),
                      key=lambda c: c.chunk_index)

    def _delete_session(self, session_id) -> None:
        self.r.delete(self._key(session_id))

    def search_context(self, session_id, query, limit=10):
        q = query.lower()
        hits = [c for c in self._load_chunks(session_id) if q in c.content.lower()]
        return hits[:limit]

    # -- helper ------------------------------------------------------------

    def _decode(self, raw: str):
        d = json.loads(raw)
        from longloom import ContextChunk
        return ContextChunk(**d)


def main() -> int:
    try:
        store = RedisContextStore()
        store.r.ping()
    except Exception as exc:
        guard(f"Redis reachable on localhost:6379 (start: 'redis-server'; "
              f"pip install redis) — {type(exc).__name__}", False)

    processor = MapReduceProcessor(
        llm=DemoLLM(), store=store,
        max_chunk_tokens=600, max_context_tokens=8192,
    )
    result = processor.process(SAMPLE_DOC, task="Summarize key themes",
                               session_id="redis-demo")
    show("Custom RedisContextStore (your own backend!)", result, processor)
    print(f"  redis keys used: {store.r.keys('longloom:session:*')}")
    processor.delete_session("redis-demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())