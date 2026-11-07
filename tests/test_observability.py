"""Observability hooks: structured events + token usage metadata."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from longloom import InMemoryContextStore, LLMAdapter, MapReduceProcessor
from longloom.llm import LLMAdapter as Adapter
from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc


class UsageReportingFake(PhaseAwareFakeChatModel):
    """Reports usage_metadata the way real langchain models do."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)
        msg = result.generations[0].message
        result.generations[0].message = AIMessage(
            content=msg.content,
            usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
        )
        return result


def test_event_sequence_and_counts():
    events: list[dict] = []
    processor = MapReduceProcessor(
        llm=PhaseAwareFakeChatModel(), store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
        on_event=events.append,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Summarize", session_id="obs")

    names = [e["event"] for e in events]
    assert names[0] == "run_started"
    assert names[-1] == "run_completed"
    assert names.count("chunk_stored") == result["num_chunks"]
    assert names.count("map_completed") == result["num_chunks"]
    assert names.count("merge_completed") == 1
    assert "run_started" in names[0] and events[0]["session_id"] == "obs"
    assert all("ts" in e for e in events)


def test_events_carry_token_usage():
    events: list[dict] = []
    processor = MapReduceProcessor(
        llm=UsageReportingFake(), store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
        on_event=events.append,
    )
    processor.process(make_doc(KEY_CONCEPTS, repeats=30), task="Summarize", session_id="usage")
    map_events = [e for e in events if e["event"] == "map_completed"]
    assert map_events and all(
        e.get("usage", {}) and e["usage"].get("total_tokens") == 18 for e in map_events)


def test_on_event_failure_never_kills_the_run():
    def bad_event(e):
        raise RuntimeError("telemetry is down")
    processor = MapReduceProcessor(
        llm=PhaseAwareFakeChatModel(), store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
        on_event=bad_event,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Summarize", session_id="bad-obs")
    assert result["final_result"].strip()


def test_adapter_extracts_usage_from_both_shapes():
    from langchain_core.messages import AIMessage

    class ShapeFake(PhaseAwareFakeChatModel):
        shape: str = "openai"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            result = super()._generate(messages, stop, run_manager, **kwargs)
            msg = result.generations[0].message
            if self.shape == "openai":
                result.generations[0].message = AIMessage(
                    content=msg.content,
                    response_metadata={"token_usage": {"prompt_tokens": 3,
                                                       "completion_tokens": 4,
                                                       "total_tokens": 7}})
            else:
                result.generations[0].message = AIMessage(
                    content=msg.content,
                    usage_metadata={"input_tokens": 3, "output_tokens": 4,
                                    "total_tokens": 7})
            return result

    for shape in ("openai", "langchain"):
        adapter = LLMAdapter(ShapeFake(shape=shape))
        text, meta = adapter.complete_with_meta("CHUNK 1/1: x")
        assert meta["usage"] and meta["usage"]["total_tokens"] == 7, shape


def test_process_async_matches_process():
    import asyncio

    processor = MapReduceProcessor(
        llm=PhaseAwareFakeChatModel(), store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
    )
    doc = make_doc(KEY_CONCEPTS, repeats=30)
    sync_result = processor.process(doc, task="Summarize", session_id="a1")
    async_result = asyncio.run(processor.process_async(
        doc, task="Summarize", session_id="a2"))
    assert async_result["num_chunks"] == sync_result["num_chunks"]
    assert async_result["final_result"] == sync_result["final_result"]