#!/usr/bin/env python3
"""01 — Quickstart: minimal map/merge/repair, zero external services.

Runs anywhere: no API keys, no database servers, no tiktoken.
"""
from _common import DemoLLM, SAMPLE_DOC, show

from longloom import InMemoryContextStore, MapReduceProcessor


def main() -> int:
    processor = MapReduceProcessor(
        llm=DemoLLM(),                  # any BaseChatModel / callable works here
        store=InMemoryContextStore(),   # zero-dependency default store
        max_chunk_tokens=600,
        max_context_tokens=8192,
        max_repair_attempts=3,
    )
    result = processor.process(SAMPLE_DOC, task="Summarize key themes", session_id="quickstart")
    show("Quickstart (demo LLM + in-memory store)", result, processor)
    print("\nSession stored — retrieve anytime:")
    info = processor.get_session_info("quickstart")
    print(f"  {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())