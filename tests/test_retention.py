"""Retention (list_sessions, delete_sessions_older_than) and PII scrubbing."""
from __future__ import annotations

import uuid

import pytest

from longloom import InMemoryContextStore, JSONFileContextStore, MapReduceProcessor
from longloom.chunks import ContextChunk, content_hash, utc_now_iso
from longloom.tokens import default_counter
from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc


def make_stores(tmp_path):
    return [
        ("memory", InMemoryContextStore()),
        ("sqlite", __import__("longloom").SQLiteContextStore(str(tmp_path / "r.db"))),
        ("jsonfile", JSONFileContextStore(str(tmp_path / "rj"))),
    ]


def test_list_sessions_all_builtins(tmp_path):
    for name, store in make_stores(tmp_path):
        store.store_context("AlphaFold document one. " * 30, session_id="s-one",
                            max_chunk_tokens=200)
        store.store_context("Sepsis document two. " * 30, session_id="s-two",
                            max_chunk_tokens=200)
        assert sorted(store.list_sessions()) == ["s-one", "s-two"], name


def test_delete_sessions_older_than(tmp_path):
    import datetime as dt

    for name, store in make_stores(tmp_path):
        old_chunk = ContextChunk(
            id=f"old_{uuid.uuid4().hex[:8]}_0000_abcd1234",
            content="ancient document about AlphaFold",
            metadata={"content_hash": "abcd1234"}, token_count=5,
            created_at="2020-01-01T00:00:00+00:00", chunk_index=0, total_chunks=1,
        )
        store._persist([old_chunk])  # backdated directly
        fresh_sid = f"fresh_{uuid.uuid4().hex[:8]}"
        store.store_context("recent sepsis document. " * 20,
                            session_id=fresh_sid, max_chunk_tokens=200)

        deleted = store.delete_sessions_older_than(days=30)
        assert deleted == 1, name
        assert store.get_session_info("old_" + uuid.uuid4().hex[:8])["chunk_count"] == 0
        # the fresh session survives
        assert store.get_session_info(fresh_sid)["chunk_count"] > 0


def test_retention_respects_namespace(tmp_path):
    from longloom import SQLiteContextStore

    db = str(tmp_path / "shared.db")
    store_a = SQLiteContextStore(db_path=db, namespace="app-a")
    store_b = SQLiteContextStore(db_path=db, namespace="app-b")
    old_chunk = ContextChunk(
        id="app-a-sess_0000_beefcafe", content="old app-a data about AlphaFold",
        metadata={"content_hash": "beefcafe"}, token_count=5,
        created_at="2020-06-01T00:00:00+00:00", chunk_index=0, total_chunks=1,
    )
    store_a._persist([old_chunk])
    store_b.store_context("app-b fresh data. " * 20, session_id="b-sess",
                          max_chunk_tokens=200)

    assert store_a.list_sessions() == ["app-a-sess"]
    assert store_a.delete_sessions_older_than(days=30) == 1
    assert len(store_b.retrieve_context("b-sess")) > 0  # app-b untouched


def test_custom_backend_without_list_sessions_raises():
    from longloom import BaseContextStore, InMemoryContextStore, MapReduceProcessor

    class MinimalStore(InMemoryContextStore):
        pass

    # built-ins implement it; a custom backend that opts out surfaces clearly
    class Opaque(BaseContextStore):
        def _persist(self, chunks): ...
        def _load_chunks(self, session_id, chunk_indices=None): return []
        def _delete_session(self, session_id): ...
        def search_context(self, session_id, query, limit=10): return []

    with pytest.raises(NotImplementedError, match="list_sessions"):
        Opaque().list_sessions()


def test_scrubber_runs_before_anything_is_persisted(tmp_path):
    from longloom import SQLiteContextStore

    seen: list[str] = []

    def scrub(text: str) -> str:
        seen.append(text)
        return text.replace("SECRET-", "REDACTED-")

    store = SQLiteContextStore(str(tmp_path / "s.db"))
    processor = MapReduceProcessor(
        llm=PhaseAwareFakeChatModel(), store=store,
        max_chunk_tokens=400, max_context_tokens=8192,
        scrubber=scrub,
    )
    doc = ("Incident log: ticket SECRET-77123 involves patient data. " * 40)
    result = processor.process(doc, task="Summarize", session_id="scrubbed")

    # nothing with the raw marker was ever stored
    chunks = processor.retrieve_chunks(result["session_id"])
    assert all("SECRET-" not in c.content for c in chunks)
    assert all("REDACTED-" in c.content for c in chunks)
    assert "SECRET-" not in result["final_result"]
    assert seen and "SECRET-" in seen[0]  # scrubber received the raw text