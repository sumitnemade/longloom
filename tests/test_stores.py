"""Round-trip contract tests — every backend must behave identically."""
from __future__ import annotations

import pytest

from longloom import (
    InMemoryContextStore,
    JSONFileContextStore,
    SQLiteContextStore,
)


def make_stores(tmp_path):
    return [
        ("memory", InMemoryContextStore()),
        ("sqlite-inline", SQLiteContextStore(db_path=str(tmp_path / "a.db"))),
        (
            "sqlite-filebacked",
            SQLiteContextStore(
                db_path=str(tmp_path / "b.db"), content_dir=str(tmp_path / "bfiles")
            ),
        ),
        ("jsonfile", JSONFileContextStore(root_dir=str(tmp_path / "json"))),
    ]


DOC = (
    "Chapter about AlphaFold and drug discovery. " * 120
) + "Final chapter mentions sepsis and FDA oversight."


def test_store_roundtrip_all_backends(tmp_path):
    for name, store in make_stores(tmp_path):
        chunks = store.store_context(DOC, session_id="s1", max_chunk_tokens=200)
        assert len(chunks) >= 2, name
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        assert all(c.total_chunks == len(chunks) for c in chunks)

        loaded = store.retrieve_context("s1")
        assert [c.id for c in loaded] == [c.id for c in chunks], name
        assert "".join(c.content for c in loaded).replace(" ", "") == DOC.replace(" ", ""), name

        # token accounting is consistent
        info = store.get_session_info("s1")
        assert info["chunk_count"] == len(chunks)
        assert info["total_tokens"] == sum(c.token_count for c in chunks)


def test_retrieve_filters_and_budget(tmp_path):
    for name, store in make_stores(tmp_path):
        store.store_context(DOC, session_id="s2", max_chunk_tokens=200)
        first_two = store.retrieve_context("s2", chunk_indices=[0, 1])
        assert [c.chunk_index for c in first_two] == [0, 1], name

        budgeted = store.retrieve_context("s2", max_tokens=first_two[0].token_count)
        assert len(budgeted) == 1, name


def test_search_and_delete(tmp_path):
    for name, store in make_stores(tmp_path):
        store.store_context(DOC, session_id="s3", max_chunk_tokens=200)
        hits = store.search_context("s3", "sepsis")
        assert hits, name
        assert all("sepsis" in c.content.lower() for c in hits)

        store.delete_session("s3")
        assert store.get_session_info("s3")["chunk_count"] == 0
        assert store.search_context("s3", "sepsis") == []


def test_store_context_validation(tmp_path):
    store = InMemoryContextStore()
    with pytest.raises(ValueError):
        store.store_context("", "s")
    with pytest.raises(ValueError):
        store.store_context("x", "")
    with pytest.raises(ValueError):
        store.store_context("x", "s", max_chunk_tokens=0)