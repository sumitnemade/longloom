#!/usr/bin/env python3
"""14 — Real NVIDIA Nemotron × EVERY storage backend.

The full production matrix: one real LLM, every database. Backends that
aren't configured are skipped with the exact reason; everything reachable
runs the complete chunk → map → merge workflow and verifies persistence.

    export NVIDIA_API_KEY=...                    # required
    export LONGCTX_PG_DSN=postgresql://...       # optional (or docker PG on :5433)
    export LONGCTX_MYSQL_DSN=mysql://...         # optional (or local :3307)
    export LONGCTX_MONGO_URI=mongodb://...       # optional (default :27017)
    python examples/14_nvidia_all_databases.py
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

MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")


def main() -> int:
    guard("NVIDIA_API_KEY env var", bool(
        os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")))

    from longloom import LLMAdapter
    llm = LLMAdapter(
        init_llm("nvidia", MODEL,
                 api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY"),
                 temperature=0.2, max_tokens=1024, thinking=False),
        min_interval=1.0,      # pace calls across the 8-backend matrix
        retry_attempts=5, retry_backoff=4.0,   # survive provider 429 windows
    )
    print(f"LLM: {MODEL} (real API, thinking disabled)\n")

    ran, skipped, failed = run_matrix(
        llm, SAMPLE_DOC, "Summarize key themes", prefix="nvidia-mx",
    )
    print(f"\nReal NVIDIA verified against {ran} storage backend(s); "
          f"{skipped} skipped, {failed} failed.")
    return 1 if (failed or ran == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())