#!/usr/bin/env python3
"""04 — OpenAI as the LLM (any model, same workflow).

Requires:  export OPENAI_API_KEY=...   and   pip install langchain-openai
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SAMPLE_DOC, guard, show

from longloom import InMemoryContextStore, MapReduceProcessor


def main() -> int:
    guard("OPENAI_API_KEY env var", bool(os.environ.get("OPENAI_API_KEY")))
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        guard("langchain-openai package:  pip install 'longloom[openai]'", False)
        return 1

    llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                     api_key=os.environ["OPENAI_API_KEY"],
                     temperature=0.2, max_tokens=1024)
    processor = MapReduceProcessor(
        llm=llm,
        store=InMemoryContextStore(),
        max_chunk_tokens=3000,
        max_context_tokens=128_000,  # gpt-4o-mini window
    )
    result = processor.process(SAMPLE_DOC * 4, task="Summarize key themes",
                               session_id="openai-demo")
    show("OpenAI ChatOpenAI", result, processor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())