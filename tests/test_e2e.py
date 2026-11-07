"""End-to-end workflow tests: same graph, every storage backend, fake LLM.

Proves the library works with ANY LLM and ANY storage — the identical
MapReduceProcessor code path runs over memory / SQLite / JSONL stores.
"""
from __future__ import annotations

import pytest

from longloom import (
    InMemoryContextStore,
    JSONFileContextStore,
    MapReduceProcessor,
    SQLiteContextStore,
)
from longloom.llm import LLMAdapter

from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc


def make_stores(tmp_path):
    return [
        ("memory", InMemoryContextStore()),
        ("sqlite", SQLiteContextStore(db_path=str(tmp_path / "e.db"))),
        ("jsonfile", JSONFileContextStore(root_dir=str(tmp_path / "ej"))),
    ]


def run_no_repair(store):
    fake = PhaseAwareFakeChatModel()
    processor = MapReduceProcessor(
        llm=fake,
        store=store,
        max_chunk_tokens=400,
        max_context_tokens=8192,
        max_repair_attempts=3,
    )
    result = processor.process(
        make_doc(KEY_CONCEPTS, repeats=30),
        task="Summarize key findings",
        session_id="e2e-no-repair",
    )
    return fake, result, processor


@pytest.mark.parametrize("backend", ["memory", "sqlite", "jsonfile"])
def test_e2e_no_repair(tmp_path, backend):
    store = dict(make_stores(tmp_path))[backend]
    fake, result, processor = run_no_repair(store)

    assert result["num_chunks"] >= 2, backend
    assert fake.phase_counts["map"] == result["num_chunks"]
    assert fake.phase_counts["merge"] == 1
    assert fake.phase_counts["repair"] == 0
    assert result["repair_attempts"] == 0

    # context preservation: every tracked concept survives the merge
    final = result["final_result"].lower()
    missing = [c for c in KEY_CONCEPTS if c.lower() not in final]
    assert not missing, f"{backend}: missing concepts {missing}"

    # storage consistency
    info = processor.get_session_info(result["session_id"])
    assert info["total_tokens"] == result["total_input_tokens"]
    chunks = processor.retrieve_chunks(result["session_id"], max_tokens=10_000)
    assert sum(c.token_count for c in chunks) == result["total_input_tokens"]
    assert processor.search_session(result["session_id"], "AlphaFold"), backend


@pytest.mark.parametrize("backend", ["memory", "sqlite", "jsonfile"])
def test_e2e_forced_repair(tmp_path, backend):
    store = dict(make_stores(tmp_path))[backend]
    fake = PhaseAwareFakeChatModel(
        merge_response="[MERGE RESULT] " + "; ".join(KEY_CONCEPTS) * 30
    )
    processor = MapReduceProcessor(
        llm=fake,
        store=store,
        max_chunk_tokens=400,
        max_context_tokens=300,
        max_repair_attempts=3,
    )
    result = processor.process(
        make_doc(KEY_CONCEPTS, repeats=30),
        task="Summarize key findings",
        session_id="e2e-repair",
    )

    assert fake.phase_counts["repair"] >= 1, backend
    assert result["repair_attempts"] == fake.phase_counts["repair"]
    assert result["merge_token_count"] <= 300, backend  # repair brought it under the window
    final = result["final_result"].lower()
    missing = [c for c in KEY_CONCEPTS if c.lower() not in final]
    assert not missing, f"{backend}: missing concepts {missing}"
    assert any("Repair attempt" in line for line in result["processing_log"])


def test_e2e_with_plain_callable_llm():
    """Even a bare function works as the LLM — ultimate provider-agnosticism."""
    def tiny_llm(prompt: str) -> str:
        if "CHUNK RESULTS:" in prompt:
            return "; ".join(KEY_CONCEPTS)
        if "CURRENT RESULT:" in prompt:
            return "repaired: " + "; ".join(KEY_CONCEPTS)
        return "[MAP RESULT] chunk ok"

    processor = MapReduceProcessor(
        llm=tiny_llm,
        store=InMemoryContextStore(),
        max_chunk_tokens=400,
        max_context_tokens=8192,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30), task="Summarize")
    assert result["num_chunks"] >= 2
    final = result["final_result"].lower()
    assert all(c.lower() in final for c in KEY_CONCEPTS)
    assert LLMAdapter(tiny_llm).backend == "callable"