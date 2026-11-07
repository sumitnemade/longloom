#!/usr/bin/env python3
"""05 — Anthropic Claude as the LLM (same workflow, different vendor).

Requires:  export ANTHROPIC_API_KEY=...   and   pip install langchain-anthropic
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SAMPLE_DOC, guard, show

from longloom import InMemoryContextStore, MapReduceProcessor


def main() -> int:
    guard("ANTHROPIC_API_KEY env var", bool(os.environ.get("ANTHROPIC_API_KEY")))
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        guard("langchain-anthropic package:  pip install langchain-anthropic", False)
        return 1

    llm = ChatAnthropic(model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                        api_key=os.environ["ANTHROPIC_API_KEY"], max_tokens=1024)
    processor = MapReduceProcessor(
        llm=llm,
        store=InMemoryContextStore(),
        max_chunk_tokens=3000,
        max_context_tokens=200_000,
    )
    result = processor.process(SAMPLE_DOC * 4, task="Summarize key themes",
                               session_id="anthropic-demo")
    show("Anthropic ChatAnthropic", result, processor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())