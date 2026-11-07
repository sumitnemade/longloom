#!/usr/bin/env python3
"""06 — Fully local LLM via Ollama (no cloud, no keys — plain HTTP callable).

Requires:  Ollama running on localhost:11434 with a model pulled, e.g.:
    ollama pull llama3.1
    OLLAMA_MODEL=llama3.1 python examples/06_llm_ollama_local.py

Demonstrates wrapping ANY HTTP API as an LLM with a plain callable.
"""
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import SAMPLE_DOC, guard, show

from longloom import InMemoryContextStore, MapReduceProcessor

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")


def ollama_llm(model: str):
    def call(prompt: str) -> str:
        response = requests.post(
            f"{OLLAMA}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["response"]
    return call


def main() -> int:
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    try:
        requests.get(f"{OLLAMA}/api/tags", timeout=3).raise_for_status()
    except Exception as exc:
        guard(f"Ollama reachable at {OLLAMA} (start it: 'ollama serve'; "
              f"pull a model: 'ollama pull {model}') — {type(exc).__name__}", False)

    processor = MapReduceProcessor(
        llm=ollama_llm(model),          # plain callable — no LangChain wrapper needed
        store=InMemoryContextStore(),
        max_chunk_tokens=1200,
        max_context_tokens=8192,
    )
    result = processor.process(SAMPLE_DOC * 3, task="Summarize key themes",
                               session_id="ollama-demo")
    show(f"Local Ollama ({model})", result, processor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())