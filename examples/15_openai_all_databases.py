#!/usr/bin/env python3
"""15 — OpenAI (lower-context model) × EVERY storage backend.

Same production matrix as example 14, with the OPENAI_API_KEY from the global
env. The lowest-context chat model available on this key is gpt-4o-mini; the
workflow is configured as a deliberately low-context deployment
(max_context_tokens=8192) so chunk/merge/repair budgets genuinely bind.

    export OPENAI_API_KEY=...
    OPENAI_MODEL=gpt-4o-mini python examples/15_openai_all_databases.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import SAMPLE_DOC, guard
from _db_matrix import run_matrix

from longloom import init_llm

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")  # lowest-context chat model on this key
MAX_CONTEXT = int(os.environ.get("OPENAI_MAX_CONTEXT", "8192"))  # deliberately low


def main() -> int:
    guard("OPENAI_API_KEY env var", bool(os.environ.get("OPENAI_API_KEY")))
    from longloom import LLMAdapter
    llm = LLMAdapter(
        init_llm("openai", MODEL, temperature=0.2, max_tokens=1024, timeout=120),
        min_interval=1.0,      # pace calls across the 8-backend matrix
        retry_attempts=5, retry_backoff=4.0,   # survive provider 429 windows
    )
    print(f"LLM: {MODEL} (real API) — configured as low-context deployment "
          f"(max_context_tokens={MAX_CONTEXT})\n")

    ran, skipped, failed = run_matrix(
        llm, SAMPLE_DOC, "Summarize key themes",
        prefix="openai-mx", max_context_tokens=MAX_CONTEXT,
    )
    print(f"\nReal OpenAI ({MODEL}) verified against {ran} storage backend(s); "
          f"{skipped} skipped, {failed} failed.")
    return 1 if (failed or ran == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())