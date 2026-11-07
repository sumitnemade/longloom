"""Shared storage-backend matrix used by the real-LLM runner examples (14, 15).

Yields every backend with a lazy factory; factories raise when the backend is
unavailable so the caller can skip it with a reason. `run_matrix` executes the
full chunk→map→merge workflow per backend and verifies persistence.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
# The PostgREST stub lives in tests/ — make it importable from any entrypoint.
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _redis_store():
    """Reuse the custom Redis backend from example 12 (import by path)."""
    spec = importlib.util.spec_from_file_location(
        "redis_example", Path(__file__).parent / "12_store_redis_custom.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # noqa: defines RedisContextStore
    return mod.RedisContextStore()


def candidate_stores():
    """Yield (label, factory, cleanup) for every configured backend."""
    pg_dsn = os.environ.get("LONGCTX_PG_DSN")
    if not pg_dsn and Path("/tmp/longloom_pg_dsn").exists():
        pg_dsn = Path("/tmp/longloom_pg_dsn").read_text().strip()
    if not pg_dsn:
        try:  # local docker Postgres convention (e.g. ei-postgres on :5433)
            import subprocess

            env = subprocess.run(
                ["docker", "inspect", "ei-postgres", "--format",
                 "{{range .Config.Env}}{{println .}}{{end}}"],
                capture_output=True, text=True, timeout=10).stdout
            cfg = dict(ln.split("=", 1) for ln in env.strip().splitlines() if "=" in ln)
            pg_dsn = (f"postgresql://{cfg['POSTGRES_USER']}:{cfg['POSTGRES_PASSWORD']}"
                      f"@127.0.0.1:5433/{cfg['POSTGRES_DB']}")
        except Exception:
            pass

    yield "memory", lambda: _make(lambda: __import__("longloom").InMemoryContextStore()), None
    yield "sqlite", lambda: _make(lambda: __import__("longloom").SQLiteContextStore(
        str(ROOT / "matrix_ctx.db"))), None
    yield "jsonl", lambda: _make(lambda: __import__("longloom").JSONFileContextStore(
        str(ROOT / "matrix_ctx_data"))), None

    def pg():
        store = _make(lambda: _create_store(pg_dsn, table=f"longloom_mx_{uuid.uuid4().hex[:6]}"))
        store.get_session_info("probe")
        return store

    if pg_dsn:
        yield "postgres", pg, None

    def mysql():
        dsn = os.environ.get(
            "LONGCTX_MYSQL_DSN",
            "mysql://longloom:longloom_test@127.0.0.1:3307/longloom",
        )
        store = _make(lambda: _create_store(dsn, table=f"longloom_mx_{uuid.uuid4().hex[:6]}"))
        store.get_session_info("probe")
        return store

    yield "mysql", mysql, None

    mongo_uri = os.environ.get("LONGCTX_MONGO_URI", "mongodb://127.0.0.1:27017")
    db_name = f"longloom_mx_{uuid.uuid4().hex[:6]}"

    def mongo():
        store = _make(lambda: _create_store(f"{mongo_uri}/{db_name}"))
        store.get_session_info("probe")
        return store

    def mongo_cleanup(store):
        store.client.drop_database(store.collection.database.name)

    yield "mongodb", mongo, mongo_cleanup

    def supabase():
        from postgrest_stub import StubPostgREST  # in tests/

        stub = StubPostgREST(port=8905).start()
        store = _make(lambda: _create_store(
            "supabase://127.0.0.1:8905", table=f"longloom_mx_{uuid.uuid4().hex[:6]}",
            key="local-stub-key"))
        store.get_session_info("probe")
        store._stub = stub
        return store

    def supabase_cleanup(store):
        store._stub.tables.clear()
        store._stub.stop()

    yield "supabase (local stub)", supabase, supabase_cleanup

    def redis():
        store = _redis_store()
        store.r.ping()
        return store

    yield "redis (custom backend)", redis, None


# -- helpers ------------------------------------------------------------

def _create_store(url: str, **kw):
    from longloom import create_store

    return create_store(url, **kw)


def _make(fn: Callable[[], Any]) -> Any:
    return fn()


def run_matrix(llm, sample_doc: str, task: str, *, prefix: str = "matrix",
               max_chunk_tokens: int = 800, max_context_tokens: int = 8192,
               label_width: int = 24) -> Tuple[int, int, int]:
    """Run the workflow against every reachable backend. Returns (ran, skipped, failed)."""
    from longloom import MapReduceProcessor

    ran = skipped = failed = 0
    for label, factory, cleanup in candidate_stores():
        try:
            store = factory()
        except Exception as exc:
            print(f"[{label:>{label_width}}] SKIP — {type(exc).__name__}: {str(exc).splitlines()[0][:70]}")
            skipped += 1
            continue

        session_id = f"{prefix}_{label.split()[0]}_{uuid.uuid4().hex[:6]}"
        processor = MapReduceProcessor(
            llm=llm, store=store,
            max_chunk_tokens=max_chunk_tokens, max_context_tokens=max_context_tokens,
        )
        try:
            result = processor.process(sample_doc, task=task, session_id=session_id)
            info = processor.get_session_info(session_id)
            assert result["final_result"].strip(), "empty result"
            assert info["chunk_count"] == result["num_chunks"], "storage mismatch"
            assert processor.search_session(session_id, "AlphaFold"), "search failed"
            preview = result["final_result"].replace("\n", " ")[:80]
            print(f"[{label:>{label_width}}] OK   chunks={result['num_chunks']} "
                  f"tokens={result['total_input_tokens']:,} | {preview}...")
            ran += 1
        except Exception as exc:
            print(f"[{label:>{label_width}}] FAIL — {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
            failed += 1
        finally:
            try:
                processor.delete_session(session_id)
                if cleanup:
                    cleanup(store)
                if hasattr(store, "close"):
                    store.close()
            except Exception:
                pass

    return ran, skipped, failed