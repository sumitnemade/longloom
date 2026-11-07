#!/usr/bin/env python3
"""11 — Supabase storage backend (REST via supabase-py).

Requires a Supabase project and supabase-py:
    pip install 'longloom[supabase]'

    export SUPABASE_URL="https://<project>.supabase.co"
    export SUPABASE_KEY="<service-or-anon-key>"
    python examples/11_store_supabase.py

One-time setup — run the DDL from the Supabase SQL editor:
    python -c "from longloom.stores.supabase import table_sql; print(table_sql())"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

from _common import DemoLLM, SAMPLE_DOC, guard, show

from longloom import MapReduceProcessor, create_store
from longloom.stores.supabase import table_sql


def main() -> int:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    guard("SUPABASE_URL and SUPABASE_KEY env vars pointing at a test project",
          bool(url and key))

    processor = MapReduceProcessor(
        llm=DemoLLM(), store=create_store(url, key=key),
        max_chunk_tokens=600, max_context_tokens=8192,
    )
    result = processor.process(SAMPLE_DOC, task="Summarize key themes",
                               session_id="supabase-demo")
    show(f"Supabase ({url})", result, processor)
    processor.delete_session("supabase-demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())