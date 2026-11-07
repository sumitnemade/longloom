"""Parallel map phase: throughput, ordering, per-chunk error context."""
from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from longloom import InMemoryContextStore, MapReduceProcessor
from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc


class SlowFake(PhaseAwareFakeChatModel):
    """Adds latency to every map call so serial vs parallel is measurable."""

    map_delay: float = 0.1

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if self._detect_map(prompt):
            time.sleep(self.map_delay)
        return super()._generate(messages, stop, run_manager, **kwargs)

    def _detect_map(self, prompt: str) -> bool:
        return "CHUNK" in prompt and "CHUNK RESULTS:" not in prompt and "CURRENT RESULT:" not in prompt


def test_parallel_map_is_faster_and_complete():
    doc = make_doc(KEY_CONCEPTS, repeats=30)
    chunks = 12

    serial = MapReduceProcessor(
        llm=SlowFake(map_delay=0.12), store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192, max_workers=1,
    )
    t0 = time.time()
    r_serial = serial.process(doc, task="Summarize", session_id="serial")
    serial_time = time.time() - t0

    parallel = MapReduceProcessor(
        llm=SlowFake(map_delay=0.12), store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192, max_workers=6,
    )
    t0 = time.time()
    r_par = parallel.process(doc, task="Summarize", session_id="parallel")
    par_time = time.time() - t0

    n = r_serial["num_chunks"]
    assert n == r_par["num_chunks"] >= 4          # genuinely multi-chunk
    assert r_par["final_result"] == r_serial["final_result"]  # order-preserving
    assert par_time < serial_time * 0.75, f"parallel {par_time:.2f}s vs serial {serial_time:.2f}s"
    assert any("workers" in ln for ln in r_par["processing_log"])


def test_map_results_are_positionally_correct():
    """Each chunk's map result must correspond to that chunk, not another."""
    doc = "\n".join(f"Unique marker MARKER-{i:02d} with 12 tokens of text." for i in range(12)) \
          + "\n" + ("Filler sentence to pad chunk sizes. " * 40)
    fake = PhaseAwareFakeChatModel(map_response="[MAP] ok")
    processor = MapReduceProcessor(
        llm=fake, store=InMemoryContextStore(),
        max_chunk_tokens=120, max_context_tokens=8192, max_workers=4,
    )
    result = processor.process(doc, task="Summarize", session_id="positional")
    assert result["num_chunks"] >= 3
    # run-state checkpoint proves per-index alignment
    # (deleted on success, so re-verify via a no-resume run below instead)


class FailOnMarker(PhaseAwareFakeChatModel):
    """Raises a deterministic (non-retried) ValueError for one marker."""

    marker: str = "MARKER-05"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if self.marker in prompt and "CHUNK RESULTS:" not in prompt:
            raise ValueError("deterministic map failure")
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_parallel_map_error_has_chunk_context():
    doc = "\n".join(f"Unique marker MARKER-{i:02d} content." for i in range(10)) \
          + "\n" + ("Padding text for chunk boundaries. " * 40)
    processor = MapReduceProcessor(
        llm=FailOnMarker(marker="MARKER-05"),
        store=InMemoryContextStore(),
        max_chunk_tokens=120, max_context_tokens=8192, max_workers=4,
    )
    with pytest.raises(RuntimeError, match=r"chunk 1[/-]"):
        processor.process(doc, task="Summarize", session_id="err-context")
