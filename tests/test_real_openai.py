"""REAL OpenAI + SQLite integration test — runs only when OPENAI_API_KEY is set.

    OPENAI_API_KEY=... pytest tests/test_real_openai.py -v

Uses the lowest-context chat model available (gpt-4o-mini) configured as a
deliberately low-context deployment, with a persistent SQLite store.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — real-API test skipped",
)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def _doc() -> str:
    body = (
        "AI in healthcare report. Medical imaging, drug discovery, personalized "
        "medicine, and FDA oversight are key themes. Sepsis early warning systems "
        "save lives. AlphaFold accelerated protein structure prediction. "
    )
    return body * 40


def test_real_openai_sqlite_end_to_end(tmp_path):
    from longloom import MapReduceProcessor, SQLiteContextStore, init_llm

    llm = init_llm("openai", MODEL, temperature=0.2, max_tokens=1024, timeout=120)
    store = SQLiteContextStore(db_path=str(tmp_path / "openai_real.db"))
    processor = MapReduceProcessor(
        llm=llm,
        store=store,
        max_chunk_tokens=1200,
        max_context_tokens=8192,   # low-context deployment
        max_repair_attempts=2,
    )

    result = processor.process(_doc(), task="Summarize the key themes",
                               session_id="real-openai-sqlite")

    assert result["num_chunks"] >= 1
    assert result["final_result"].strip(), "empty summary from real API"
    assert result["merge_token_count"] <= 8192

    # SQLite persistence
    info = processor.get_session_info(result["session_id"])
    assert info["chunk_count"] == result["num_chunks"]
    assert info["total_tokens"] == result["total_input_tokens"]
    assert len(processor.retrieve_chunks(result["session_id"])) == result["num_chunks"]

    # Content sanity
    final = result["final_result"].lower()
    for concept in ("medical imaging", "drug discovery", "fda"):
        assert concept in final, f"concept missing from real summary: {concept}"

    # Data survives a fresh store instance over the same file
    from longloom import SQLiteContextStore as S
    reopened = S(db_path=str(tmp_path / "openai_real.db"))
    assert reopened.get_session_info(result["session_id"])["chunk_count"] == \
        result["num_chunks"]
    reopened.close()
    store.close()

    print("\n--- REAL OPENAI + SQLITE RESULT (truncated) ---")
    print(result["final_result"][:600])