"""REAL NVIDIA + SQLite integration test — runs only when NVIDIA_API_KEY is set.

    NVIDIA_API_KEY=... pytest tests/test_real_nvidia.py -v

Uses the real nemotron model with ``thinking=False`` (reasoning off) so the
whole max_tokens budget goes to the actual output, and a persistent SQLite
store — the exact production configuration.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NVIDIA_NIM_API_KEY")
        or os.environ.get("NEMOTRON_API_KEY")
    ),
    reason="NVIDIA_API_KEY not set — real-API test skipped",
)

MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


def _doc() -> str:
    body = (
        "AI in healthcare report. Medical imaging, drug discovery, personalized "
        "medicine, and FDA oversight are key themes. Sepsis early warning systems "
        "save lives. AlphaFold accelerated protein structure prediction. "
    )
    return body * 40  # ~4-5k tokens → multiple chunks at 1200-token budget


def test_real_nvidia_sqlite_end_to_end(tmp_path):
    from longloom import MapReduceProcessor, SQLiteContextStore, init_llm

    llm = init_llm(
        "nvidia", MODEL,
        api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
        or os.environ.get("NEMOTRON_API_KEY"),
        temperature=0.2, max_tokens=1024, thinking=False,
    )
    store = SQLiteContextStore(db_path=str(tmp_path / "nvidia_real.db"))
    processor = MapReduceProcessor(
        llm=llm,
        store=store,
        max_chunk_tokens=1200,
        max_context_tokens=8192,
        max_repair_attempts=2,
    )

    result = processor.process(_doc(), task="Summarize the key themes",
                               session_id="real-nvidia-sqlite")

    # Workflow sanity
    assert result["num_chunks"] >= 1
    assert result["final_result"].strip(), "empty summary from real API"
    assert result["merge_token_count"] <= 8192

    # SQLite persistence: everything retrievable after the run
    info = processor.get_session_info(result["session_id"])
    assert info["chunk_count"] == result["num_chunks"]
    assert info["total_tokens"] == result["total_input_tokens"]
    chunks = processor.retrieve_chunks(result["session_id"])
    assert len(chunks) == result["num_chunks"]
    assert sum(c.token_count for c in chunks) == result["total_input_tokens"]

    # Content sanity: key topics from the source doc appear in the summary
    final = result["final_result"].lower()
    for concept in ("medical imaging", "drug discovery", "fda"):
        assert concept in final, f"concept missing from real summary: {concept}"

    # New processor instance over the same SQLite file — data survives restart
    from longloom import SQLiteContextStore as S
    reopened = S(db_path=str(tmp_path / "nvidia_real.db"))
    assert reopened.get_session_info(result["session_id"])["chunk_count"] == \
        result["num_chunks"]
    reopened.close()
    store.close()

    print("\n--- REAL NVIDIA + SQLITE RESULT (truncated) ---")
    print(result["final_result"][:800])
    print(f"\nchunks={result['num_chunks']} repair={result['repair_attempts']} "
          f"tokens={result['total_input_tokens']:,}")