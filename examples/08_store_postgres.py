#!/usr/bin/env python3
"""08 — PostgreSQL storage backend (real server).

Requires a PostgreSQL server and either SQLAlchemy or psycopg2 installed:
    pip install 'longloom[postgres]'

    export LONGCTX_PG_DSN="postgresql://user:password@host:5432/longloom"
    python examples/08_store_postgres.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

from _common import DemoLLM, SAMPLE_DOC, guard, show

from longloom import MapReduceProcessor, create_store


def main() -> int:
    candidates = []
    if os.environ.get("LONGCTX_PG_DSN"):
        candidates.append(os.environ["LONGCTX_PG_DSN"])
    if Path("/tmp/longloom_pg_dsn").exists():  # stashed by example 14
        candidates.append(Path("/tmp/longloom_pg_dsn").read_text().strip())
    try:  # local docker Postgres (e.g. ei-postgres on :5433)
        import subprocess

        env = subprocess.run(
            ["docker", "inspect", "ei-postgres", "--format",
             "{{range .Config.Env}}{{println .}}{{end}}"],
            capture_output=True, text=True, timeout=10).stdout
        cfg = dict(line.split("=", 1) for line in env.strip().splitlines() if "=" in line)
        candidates.append(f"postgresql://{cfg['POSTGRES_USER']}:{cfg['POSTGRES_PASSWORD']}"
                          f"@127.0.0.1:5433/{cfg['POSTGRES_DB']}")
    except Exception:
        pass

    store, dsn = None, None
    for candidate in candidates:
        try:
            store = create_store(candidate)
            store.get_session_info("probe")
            dsn = candidate
            break
        except Exception:
            continue
    if store is None:
        guard("PostgreSQL reachable (set LONGCTX_PG_DSN or run a local docker PG; "
              "install driver: pip install 'longloom[postgres]')", False)

    processor = MapReduceProcessor(
        llm=DemoLLM(), store=store,
        max_chunk_tokens=600, max_context_tokens=8192,
    )
    result = processor.process(SAMPLE_DOC, task="Summarize key themes",
                               session_id="pg-demo")
    show(f"PostgreSQL ({dsn.split('@')[-1]})", result, processor)
    processor.delete_session("pg-demo")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())