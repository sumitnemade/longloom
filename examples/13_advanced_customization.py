#!/usr/bin/env python3
"""13 — Advanced: custom prompts, task focus, token counters, prompt safety.

Shows the remaining extension points:
* custom map/merge/repair prompt templates (brace-safe — content with
  JSON/braces never breaks rendering),
* task_focus steering,
* pluggable token counters (EstimateCounter vs TiktokenCounter),
* per-run configuration (chunk size, repair budget).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import DemoLLM, SAMPLE_DOC, show

from longloom import (
    EstimateCounter,
    InMemoryContextStore,
    MapReduceProcessor,
    PromptTemplates,
    TiktokenCounter,
)


def main() -> int:
    # 1) Custom prompts — an extraction-flavoured pipeline.
    prompts = PromptTemplates(
        map=(
            "Extract a bullet list of FACTS from the text below. "
            "Text:\n{chunk_content}\n"
            "Only facts about: {task_focus}. Prefix with [chunk {chunk_index}/{total_chunks}]."
        ),
        merge=(
            "Deduplicate and organize these extracted facts into a single report:\n"
            "{chunk_results}\nGoal: {task_focus}"
        ),
        repair=(
            "This report is {current_tokens} tokens, limit is {max_tokens}. "
            "Compress it to under {max_tokens} tokens, keeping every unique fact:\n"
            "{merged_result}"
        ),
    )

    # 2) Pluggable token counters — exact (tiktoken) vs zero-dep estimate.
    counter = TiktokenCounter()          # swap for EstimateCounter() if tiktoken absent
    print(f"counters: tiktoken(SAMPLE_DOC)={counter(SAMPLE_DOC)}, "
          f"estimate={EstimateCounter()(SAMPLE_DOC)}")

    processor = MapReduceProcessor(
        llm=DemoLLM(),
        store=InMemoryContextStore(),
        prompts=prompts,
        token_counter=counter,
        max_chunk_tokens=500,
        max_context_tokens=8192,
        task_focus="healthcare AI: imaging, drug discovery, regulation, operations",
    )
    result = processor.process(
        SAMPLE_DOC,
        task="Extract all key facts",
        task_focus="healthcare AI: imaging, drug discovery, regulation, operations",
        session_id="advanced-demo",
    )
    show("Custom prompts + pluggable counter", result, processor)

    # 3) Prompt rendering is brace-safe: JSON-in-content cannot break it.
    from longloom import render_prompt
    rendered = render_prompt("Text: {chunk_content}", {
        "chunk_content": '{"model": "gpt", "notes": "{not_a_variable}"}'})
    assert '{"model": "gpt", "notes": "{not_a_variable}"}' in rendered
    print("  brace-safety: JSON content rendered verbatim ✓")

    # 4) Chunk-level retrieval for grounding / citations.
    chunks = processor.retrieve_chunks("advanced-demo", chunk_indices=[0])
    print(f"  first chunk: {chunks[0].token_count} tokens, "
          f"hash={chunks[0].metadata['content_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())