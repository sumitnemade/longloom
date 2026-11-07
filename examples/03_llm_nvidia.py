#!/usr/bin/env python3
"""03 — Real NVIDIA Nemotron end-to-end with persistent SQLite storage.

Requires:  export NVIDIA_API_KEY=...   (https://integrate.api.nvidia.com/)

Uses the real model ID (nvidia/nemotron-3-ultra-550b-a55b) with thinking
disabled so the full max_tokens budget produces output, and stores every
chunk in SQLite so sessions survive restarts.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SAMPLE_DOC, guard, show

from longloom import MapReduceProcessor, SQLiteContextStore, init_llm


def main() -> int:
    guard("NVIDIA_API_KEY env var (get one at https://integrate.api.nvidia.com/)",
          bool(os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")
               or os.environ.get("NEMOTRON_API_KEY")))

    llm = init_llm("nvidia", os.environ.get(
        "NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        temperature=0.2, max_tokens=1024, thinking=False,
    )
    processor = MapReduceProcessor(
        llm=llm,
        store=SQLiteContextStore("nvidia_demo.db"),
        max_chunk_tokens=1500,
        max_context_tokens=8192,
        max_repair_attempts=2,
    )
    result = processor.process(
        SAMPLE_DOC * 4, task="Summarize key themes, challenges and future directions",
        session_id="nvidia-demo",
    )
    show("Real NVIDIA + SQLiteContextStore", result, processor)
    info = processor.get_session_info("nvidia-demo")
    print(f"  persisted in SQLite: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())