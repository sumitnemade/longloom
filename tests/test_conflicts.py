"""Opt-in advisory conflict detection."""
from __future__ import annotations

from longloom import InMemoryContextStore, MapReduceProcessor
from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc


def test_conflicts_detected_and_parsed():
    fake = PhaseAwareFakeChatModel(
        conflicts_response="CONFLICT: revenue is 5M || revenue is 6M\n"
                           "CONFLICT: launch 2026 || launch 2027")
    processor = MapReduceProcessor(
        llm=fake, store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
        detect_conflicts=True,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Summarize", session_id="conflicts")
    assert result["conflicts"] == ["revenue is 5M || revenue is 6M",
                                   "launch 2026 || launch 2027"]
    # advisory: final_result is untouched by the conflicts pass
    assert result["final_result"].startswith("[MERGE RESULT]")
    assert fake.calls.count("conflicts") == 1


def test_no_conflicts_returns_empty_list():
    fake = PhaseAwareFakeChatModel()  # conflicts_response = NONE
    processor = MapReduceProcessor(
        llm=fake, store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
        detect_conflicts=True,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Summarize", session_id="clean")
    assert result["conflicts"] == []


def test_conflicts_opt_in_by_default():
    fake = PhaseAwareFakeChatModel(
        conflicts_response="CONFLICT: a || b")
    processor = MapReduceProcessor(
        llm=fake, store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Summarize", session_id="opt-out")
    assert "conflicts" not in result
    assert fake.calls.count("conflicts") == 0  # no extra LLM call
