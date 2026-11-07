"""Checkpoint/resume: a crashed run must be resumable without re-mapping
completed chunks, and resume must be a no-op for changed content."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from longloom import InMemoryContextStore, MapReduceProcessor
from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc


class CrashOnNthMap(PhaseAwareFakeChatModel):
    """Raises (deterministically) on the Nth map call, then behaves."""

    crash_on: int = 3
    map_calls: int = Field(default_factory=int, exclude=True)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if "CHUNK RESULTS:" not in prompt and "CURRENT RESULT:" not in prompt:
            self.map_calls += 1
            if self.map_calls == self.crash_on:
                raise ValueError("simulated worker crash")
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_crashed_run_resumes_without_remapping(tmp_path):
    doc = make_doc(KEY_CONCEPTS, repeats=30)
    fake = CrashOnNthMap(crash_on=3)
    processor = MapReduceProcessor(
        llm=fake, store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192, max_workers=1,
    )

    # run 1: crashes on the 3rd chunk — chunks 1-2 are already checkpointed
    with pytest.raises(RuntimeError, match="chunk 3"):
        processor.process(doc, task="Summarize", session_id="resumable")
    assert fake.map_calls == 3

    # run 2: resume — only un-checkpointed chunks are mapped
    fake.crash_on = 10**9  # defuse
    restored = processor.get_session_info("resumable::run")["chunk_count"]
    assert restored >= 1
    result = processor.process(doc, task="Summarize", session_id="resumable", resume=True)
    n = result["num_chunks"]
    assert n >= 4
    assert fake.map_calls == 3 + (n - restored), (
        f"resume re-mapped checkpointed chunks: {fake.map_calls} calls for {n} chunks")
    assert any(f"Resuming: {restored}/" in ln for ln in result["processing_log"])
    final = result["final_result"].lower()
    assert all(c.lower() in final for c in KEY_CONCEPTS)  # nothing lost

    # checkpoint removed on success
    assert processor.get_session_info("resumable::run")["chunk_count"] == 0


def test_resume_is_noop_without_checkpoint(tmp_path):
    doc = make_doc(KEY_CONCEPTS, repeats=30)
    fake = CrashOnNthMap(crash_on=10**9)
    processor = MapReduceProcessor(llm=fake, store=InMemoryContextStore(),
                                   max_chunk_tokens=400, max_context_tokens=8192)
    result = processor.process(doc, task="Summarize", session_id="fresh", resume=True)
    assert fake.map_calls == result["num_chunks"]  # everything mapped normally


def test_resume_remaps_when_content_changed(tmp_path):
    doc = make_doc(KEY_CONCEPTS, repeats=30)
    fake = CrashOnNthMap(crash_on=10**9)
    processor = MapReduceProcessor(llm=fake, store=InMemoryContextStore(),
                                   max_chunk_tokens=400, max_context_tokens=8192)
    first = processor.process(doc, task="Summarize", session_id="changed")
    calls_before = fake.map_calls
    # same session id, DIFFERENT content -> doc_hash mismatch -> full re-map
    second = processor.process(doc + "Extra tail content. " * 50, task="Summarize",
                               session_id="changed", resume=True)
    assert fake.map_calls - calls_before == second["num_chunks"]
    assert second["num_chunks"] >= first["num_chunks"]


def test_resume_works_in_parallel(tmp_path):
    doc = make_doc(KEY_CONCEPTS, repeats=30)
    fake = CrashOnNthMap(crash_on=4)
    processor = MapReduceProcessor(
        llm=fake, store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192, max_workers=4,
    )
    with pytest.raises(RuntimeError):
        processor.process(doc, task="Summarize", session_id="par-resume")
    partial = fake.map_calls
    restored = processor.get_session_info("par-resume::run")["chunk_count"]
    fake.crash_on = 10**9
    result = processor.process(doc, task="Summarize", session_id="par-resume", resume=True)
    assert fake.map_calls - partial == result["num_chunks"] - restored
    assert any("Resuming:" in ln for ln in result["processing_log"])