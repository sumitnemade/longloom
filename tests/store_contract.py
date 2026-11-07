"""Shared storage-contract assertions used by both unit and DB-integration tests."""
from __future__ import annotations

import uuid

from longloom import BaseContextStore

DOC = (
    "Chapter about AlphaFold and drug discovery. " * 120
) + "Final chapter mentions sepsis and FDA oversight."


def run_store_contract(store: BaseContextStore, label: str) -> None:
    """Full round-trip contract: store → retrieve → search → info → delete."""
    session_id = f"contract_{uuid.uuid4().hex[:10]}"

    try:
        chunks = store.store_context(DOC, session_id=session_id, max_chunk_tokens=200)
        assert len(chunks) >= 2, f"{label}: expected multiple chunks"
        assert [c.chunk_index for c in chunks] == list(range(len(chunks))), label
        assert all(c.total_chunks == len(chunks) for c in chunks), label

        loaded = store.retrieve_context(session_id)
        assert [c.id for c in loaded] == [c.id for c in chunks], f"{label}: id round-trip"
        assert "".join(c.content for c in loaded).replace(" ", "") == DOC.replace(" ", ""), \
            f"{label}: content round-trip"

        # filters + budget
        first_two = store.retrieve_context(session_id, chunk_indices=[0, 1])
        assert [c.chunk_index for c in first_two] == [0, 1], label
        budgeted = store.retrieve_context(session_id, max_tokens=first_two[0].token_count)
        assert len(budgeted) == 1, f"{label}: max_tokens budget"

        # search
        hits = store.search_context(session_id, "sepsis")
        assert hits, f"{label}: search"
        assert all("sepsis" in c.content.lower() for c in hits), label

        # info
        info = store.get_session_info(session_id)
        assert info["chunk_count"] == len(chunks), f"{label}: info chunk_count"
        assert info["total_tokens"] == sum(c.token_count for c in chunks), \
            f"{label}: info total_tokens"

        # upsert (re-store same session → replace, no duplicates)
        again = store.store_context(DOC, session_id=session_id, max_chunk_tokens=200)
        assert len(again) == len(chunks), f"{label}: upsert idempotent"
        assert store.get_session_info(session_id)["chunk_count"] == len(chunks), label
    finally:
        store.delete_session(session_id)
        assert store.get_session_info(session_id)["chunk_count"] == 0, \
            f"{label}: delete_session"