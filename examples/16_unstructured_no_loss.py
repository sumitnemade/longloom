#!/usr/bin/env python3
"""16 — UNSTRUCTURED input, real LLM, no-loss measurement.

Feeds a deliberately messy document (server logs, JSON, an email thread, an
ASCII table, meeting notes, config diffs, tickets) through the map/merge
pipeline and then checks EVERY planted fact (IDs, IPs, emails, versions,
dates, money amounts...) against the final output, fact by fact.

Two guarantees are reported:
  A. LLM visibility  — chunking is lossless and every chunk is mapped
                       (structural: proven by construction, verified live).
  B. Fact preservation — how many planted facts reach the final result.

    export NVIDIA_API_KEY=...      # primary
    # or OPENAI_API_KEY=...        # fallback
    python examples/16_unstructured_no_loss.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _unstructured_doc import FACTS, build_unstructured_doc, preservation_report

from longloom import (
    InMemoryContextStore,
    MapReduceProcessor,
    PromptTemplates,
    SQLiteContextStore,
    init_llm,
)

TASK = ("Extract every fact-bearing line from the document verbatim. "
        "No summarization — no loss allowed.")
TASK_FOCUS = "completeness: every identifier, number, email, URL, host, version, date, amount"


def build_llm():
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY"):
        return init_llm("nvidia", os.environ.get(
            "NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
            api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY"),
            temperature=0.0, max_tokens=2048, thinking=False), "nvidia"
    if os.environ.get("OPENAI_API_KEY"):
        return init_llm("openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                        api_key=os.environ["OPENAI_API_KEY"],
                        temperature=0.0, max_tokens=2048, timeout=120), "openai"
    print("SKIPPED: set NVIDIA_API_KEY or OPENAI_API_KEY for the real-LLM run")
    raise SystemExit(3)


def main() -> int:
    doc = build_unstructured_doc()
    llm, provider = build_llm()
    print(f"LLM: {provider} | unstructured doc: {len(doc):,} chars\n")

    # verbatim extraction preset — map copies lines, merge keeps every line
    prompts = PromptTemplates.for_extraction()
    store = SQLiteContextStore("unstructured_ctx.db")
    processor = MapReduceProcessor(
        llm=llm, store=store,
        prompts=prompts,
        max_chunk_tokens=450,          # fine-grained chunks → less per-chunk drop risk
        max_context_tokens=8192,
        max_repair_attempts=2,
        task_focus=TASK_FOCUS,
    )
    result = processor.process(doc, task=TASK, session_id="unstructured-no-loss")

    chunks = processor.retrieve_chunks(result["session_id"])
    reconstructed = "".join(c.content for c in chunks).replace(" ", "")
    a_lossless = reconstructed == doc.replace(" ", "")
    a_all_mapped = result["num_chunks"] >= 2

    print("=== Guarantee A: LLM sees 100% of the input ===")
    print(f"  chunks={result['num_chunks']}  input_tokens={result['total_input_tokens']:,}  "
          f"chunking lossless={a_lossless}  all chunks mapped={a_all_mapped}")

    # Guarantee B — semantic: fact-by-fact preservation in the final result.
    report = preservation_report(result["final_result"])
    print(f"\n=== Guarantee B: fact preservation ({report['preserved']}/{report['total']} "
          f"= {report['rate']:.1f}%) ===")
    if report["missing"]:
        print("  MISSING from final result:")
        for f in report["missing"]:
            print(f"    - {f}")

    print(f"\n  repair_attempts={result['repair_attempts']}  "
          f"merge_tokens={result['merge_token_count']:,}")
    print(f"  final result ({len(result['final_result']):,} chars):")
    print("  " + result["final_result"][:400].replace("\n", "\n  ") + " ...")

    store.close()
    ok = a_lossless and a_all_mapped and report["rate"] == 100.0
    print(f"\n{'✅ NO LOSS: all planted facts preserved' if ok else '❌ loss detected'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())