#!/usr/bin/env python3
"""18 — Enterprise integration: corporate LLM gateway + existing database.

The two questions every enterprise asks first:

1. "Our LLM access goes through a gateway (LiteLLM / Azure APIM / Portkey),
   not vendor SDKs."  → LongLoom's minimum LLM contract is a plain
   ``str -> str`` callable — point it at your gateway. Done.
2. "We're not provisioning a database; chunks must live in OUR Postgres
   table with OUR columns, next to application data."  → table=, columns=,
   namespace=, init_schema=False.

    export NVIDIA_API_KEY=...        (or OPENAI_API_KEY=...)
    python examples/18_enterprise_gateway.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from longloom import LLMAdapter, MapReduceProcessor


def build_llm():
    """The corporate gateway — the ONLY integration surface for LLM access."""
    if os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY"):
        key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY")

        def gateway(prompt: str) -> str:
            import requests

            r = requests.post(
                "https://integrate.api.nvidia.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "nvidia/nemotron-3-ultra-550b-a55b",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 1024, "temperature": 0.0,
                      "chat_template_kwargs": {"thinking": False}},
                timeout=120)
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content") or ""

        return LLMAdapter(gateway, min_interval=1.0), "corporate gateway -> NVIDIA"
    if os.environ.get("OPENAI_API_KEY"):
        from longloom import init_llm

        return LLMAdapter(init_llm("openai", "gpt-4o-mini",
                                   api_key=os.environ["OPENAI_API_KEY"],
                                   timeout=120), min_interval=1.0), "corporate gateway -> OpenAI"
    print("SKIPPED: set NVIDIA_API_KEY or OPENAI_API_KEY")
    raise SystemExit(3)


def build_store():
    """The company's EXISTING database — discovered, not provisioned."""
    dsn = os.environ.get("LONGCTX_PG_DSN")
    if not dsn:
        try:
            import subprocess

            env = subprocess.run(
                ["docker", "inspect", "ei-postgres", "--format",
                 "{{range .Config.Env}}{{println .}}{{end}}"],
                capture_output=True, text=True, timeout=10).stdout
            cfg = dict(ln.split("=", 1) for ln in env.strip().splitlines() if "=" in ln)
            dsn = (f"postgresql://{cfg['POSTGRES_USER']}:{cfg['POSTGRES_PASSWORD']}"
                   f"@127.0.0.1:5433/{cfg['POSTGRES_DB']}")
        except Exception:
            return None, "postgres (not reachable)"
    try:
        import psycopg2

        from longloom import POSTGRES_DIALECT, SqlContextStore

        store = SqlContextStore(
            connect=lambda: psycopg2.connect(dsn),       # raw corporate connection
            dialect=POSTGRES_DIALECT,
            table="enterprise_doc_chunks",               # DBA-approved table
            columns={"id": "chunk_pk", "content": "body_text",
                     "session_id": "sess"},              # company schema
            namespace="claims-pipeline",                 # multi-tenant table
            namespace_field="tenant",
            init_schema=False,                           # DBA owns DDL
        )
        store.init_schema()
        return store, f"existing Postgres ({dsn.split('@')[-1]})"
    except Exception:
        return None, "postgres (not reachable)"


def main() -> int:
    llm, llm_label = build_llm()
    store, store_label = build_store()
    if store is None:
        print(f"SKIPPED: no reachable database for the demo ({store_label}) — "
              "the gateway seam itself needs no database; see example 01.")
        raise SystemExit(3)

    print(f"LLM: {llm_label}\nDB: {store_label}\n")
    processor = MapReduceProcessor(llm=llm, store=store,
                                   max_chunk_tokens=900, max_context_tokens=8192)
    doc = ("Claim CLM-88231 filed 2026-09-14, amount $12,480, policy PN-77123. " * 120)
    result = processor.process(doc, task="List every claim, date and amount verbatim",
                               session_id="claims-88231")

    print(f"chunks={result['num_chunks']}  tokens={result['total_input_tokens']:,}  "
          f"repairs={result['repair_attempts']}")
    final = result["final_result"]
    for fact in ("CLM-88231", "2026-09-14", "PN-77123"):
        assert fact in final, f"compliance failure: {fact} lost"
    assert "12,480" in final or "12480" in final
    print("compliance spot-check: claim id, date, policy and amount all preserved ✓")
    processor.delete_session("claims-88231")
    store.close()
    print("\n✅ enterprise pattern verified: gateway callable + existing table + tenant namespace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
