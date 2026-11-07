"""Production-hardening tests: the real-world breaks, fixed.

1. Merge-input overflow at scale (verbatim extraction, ~45k-token doc, 8k window)
   -> hierarchical merge completes where a single merge call was API-rejected.
2. Transient API failures -> LLMAdapter retries with backoff.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fake_llm import PhaseAwareFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from longloom import InMemoryContextStore, MapReduceProcessor


# ---------------------------------------------------------------------------
# 1) merge-input overflow at scale -> hierarchical merge
# ---------------------------------------------------------------------------

class FaithfulExtractor(PhaseAwareFakeChatModel):
    """Map output ~ proportional to input (verbatim-extraction behaviour);
    merge behaves like a real condensing LLM — deduplicates repeated lines."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if "CURRENT RESULT:" in prompt:
            self.calls.append("repair")
            body = prompt.split("CURRENT RESULT:", 1)[1]
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="\n".join(sorted(set(body.splitlines())))))])
        if "CHUNK RESULTS:" in prompt:
            self.calls.append("merge")
            body = prompt.split("CHUNK RESULTS:", 1)[1]
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="\n".join(sorted(set(body.splitlines())))))])
        self.calls.append("map")
        fact_lines = [ln for ln in prompt.splitlines() if any(c.isdigit() for c in ln)]
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="\n".join(fact_lines)))])


class WindowEnforcing(FaithfulExtractor):
    """Rejects oversized prompts exactly like a real inference API."""

    max_input_tokens: int = 8000

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if len(prompt) // 4 > self.max_input_tokens:
            raise ValueError(
                f"context_length_exceeded: prompt ~{len(prompt) // 4} tokens")
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_hierarchical_merge_survives_large_verbatim_run():
    doc = ("INC-1001 payments-svc timeout trace_id=tr-a latency=12ms\n" * 4000)
    llm = WindowEnforcing()
    processor = MapReduceProcessor(
        llm=llm, store=InMemoryContextStore(),
        max_chunk_tokens=2000, max_context_tokens=8192,
    )
    result = processor.process(doc, task="Extract all incidents verbatim",
                               session_id="scale-test")

    assert result["num_chunks"] >= 15          # genuinely large run
    assert any("Hierarchical merge pass" in ln
               for ln in result["processing_log"]), "hierarchy did not engage"
    assert "INC-1001" in result["final_result"]  # no fact lost in the tree
    # no combine call ever exceeded the enforced window (would have raised)
    assert result["final_result"].strip()


def test_small_runs_skip_hierarchy():
    from fake_llm import KEY_CONCEPTS, make_doc
    llm = FaithfulExtractor()
    processor = MapReduceProcessor(
        llm=llm, store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Extract", session_id="small")
    assert not any("Hierarchical merge pass" in ln
                   for ln in result["processing_log"])


# ---------------------------------------------------------------------------
# 2) transient failures -> retry with backoff
# ---------------------------------------------------------------------------

class FlakyThenOK(PhaseAwareFakeChatModel):
    failures_left: int = 2

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        if self.failures_left > 0 and "CHUNK" in prompt and "RESULTS" not in prompt:
            self.failures_left -= 1
            raise ConnectionError("503 service temporarily unavailable")
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_llm_adapter_retries_transient_failures(monkeypatch):
    from longloom import LLMAdapter
    from longloom.llm import LLMAdapter as A

    sleeps: list[float] = []
    monkeypatch.setattr(A, "complete", A.complete)  # keep real logic
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))

    flaky = FlakyThenOK()
    adapter = LLMAdapter(flaky, retry_attempts=3, retry_backoff=0.1)
    out = adapter.complete("CHUNK 1/2:\nfact line 123")
    assert out == flaky.map_response
    assert len(sleeps) == 2                      # two retries happened
    assert sleeps[1] > sleeps[0]                 # exponential backoff


def test_llm_adapter_exhausts_retries(monkeypatch):
    import time as _time
    from longloom import LLMAdapter

    monkeypatch.setattr(_time, "sleep", lambda s: None)

    class AlwaysDown(PhaseAwareFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            raise ConnectionError("down")

    adapter = LLMAdapter(AlwaysDown(), retry_attempts=2, retry_backoff=0.0)
    with pytest.raises(ConnectionError):
        adapter.complete("anything")


# ---------------------------------------------------------------------------
