"""REAL database integration tests — PostgreSQL, MySQL, MongoDB.

Each backend is probed at session start: if the server is reachable and a
connection succeeds, the full storage contract runs against the real database;
otherwise the test skips honestly. Override endpoints via env:

    LONGCTX_PG_DSN=postgresql://user@/db pytest tests/test_db_integration.py -v
    LONGCTX_MYSQL_DSN=mysql://user:pw@host/db ...
    LONGCTX_MONGO_URI=mongodb://host:27017 ...
"""
from __future__ import annotations

import uuid

import pytest

from longloom import create_store
from store_contract import run_store_contract


def _unique_dsn_suffix() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def _pg_store():
    import os
    from pathlib import Path

    dsn = os.environ.get("LONGCTX_PG_DSN")
    candidates = [dsn] if dsn else []
    # Local docker Postgres convention (e.g. ei-postgres on :5433) — DSN stashed
    # by examples/14 or read straight from the container config.
    if Path("/tmp/longloom_pg_dsn").exists():
        candidates.append(Path("/tmp/longloom_pg_dsn").read_text().strip())
    try:
        import subprocess

        env = subprocess.run(
            ["docker", "inspect", "ei-postgres", "--format",
             "{{range .Config.Env}}{{println .}}{{end}}"],
            capture_output=True, text=True, timeout=10).stdout
        cfg = dict(line.split("=", 1) for line in env.strip().splitlines() if "=" in line)
        candidates.append(
            f"postgresql://{cfg['POSTGRES_USER']}:{cfg['POSTGRES_PASSWORD']}"
            f"@127.0.0.1:5433/{cfg['POSTGRES_DB']}"
        )
    except Exception:
        pass
    candidates += ["postgresql://@/longloom_test", "postgresql://@/postgres"]

    for candidate in candidates:
        if not candidate:
            continue
        try:
            store = create_store(candidate, table=f"longloom_{_unique_dsn_suffix()}")
            store.get_session_info("probe")  # touch the DB
            return store, candidate
        except Exception:
            continue
    return None, None


def test_postgres_contract():
    store, dsn = _pg_store()
    if store is None:
        pytest.skip("PostgreSQL not reachable (set LONGCTX_PG_DSN to override)")
    run_store_contract(store, f"postgres ({dsn})")
    store.close()


# ---------------------------------------------------------------------------
# MySQL / MariaDB
# ---------------------------------------------------------------------------

def _mysql_store():
    import os

    dsn = os.environ.get("LONGCTX_MYSQL_DSN")
    candidates = [dsn] if dsn else [
        "mysql://longloom:longloom_test@127.0.0.1:3307/longloom",  # user-space instance
        "mysql://root@127.0.0.1/mysql",                          # dev boxes
        "mysql://root@localhost/mysql?unix_socket=/var/run/mysqld/mysqld.sock",
    ]
    for candidate in candidates:
        try:
            store = create_store(candidate, table=f"longloom_{_unique_dsn_suffix()}")
            store.get_session_info("probe")
            return store, candidate
        except Exception:
            continue
    return None, None


def test_mysql_contract():
    store, dsn = _mysql_store()
    if store is None:
        pytest.skip("MySQL/MariaDB not reachable (set LONGCTX_MYSQL_DSN to override)")
    run_store_contract(store, f"mysql ({dsn})")
    store.close()


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------

def _mongo_store():
    import os

    uri = os.environ.get("LONGCTX_MONGO_URI", "mongodb://127.0.0.1:27017")
    try:
        from pymongo import MongoClient

        client = MongoClient(uri, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")  # force connection attempt
    except Exception:
        return None, None
    from longloom import MongoContextStore

    db_name = f"longloom_test_{_unique_dsn_suffix()}"
    return MongoContextStore(client_or_uri=uri, db_name=db_name), db_name


def test_mongodb_contract():
    store, db_name = _mongo_store()
    if store is None:
        pytest.skip("MongoDB not reachable (set LONGCTX_MONGO_URI to override)")
    try:
        run_store_contract(store, f"mongodb ({db_name})")
    finally:
        store.client.drop_database(db_name)
        store.close()


# ---------------------------------------------------------------------------
# Supabase — remote service, only runs with explicit credentials
# ---------------------------------------------------------------------------

def test_supabase_contract():
    import os

    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
        pytest.skip(
            "SUPABASE_URL / SUPABASE_KEY not set — Supabase is a remote "
            "service; point these at a test project to run for real"
        )
    try:
        store = create_store(os.environ["SUPABASE_URL"], table=f"longloom_{_unique_dsn_suffix()}")
    except ImportError:
        pytest.skip("supabase-py not installed: pip install 'longloom[supabase]'")
    run_store_contract(store, "supabase")


# ---------------------------------------------------------------------------
# End-to-end workflow against a REAL database
# ---------------------------------------------------------------------------

def test_e2e_workflow_on_real_databases(tmp_path):
    """Map/merge/repair over a real DB when one is available (first match wins)."""
    from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc
    from longloom import MapReduceProcessor

    store, label = _mongo_store()
    kind = "mongodb"
    if store is None:
        store, _ = _pg_store()
        kind = "postgres"
    if store is None:
        store, _ = _mysql_store()
        kind = "mysql"
    if store is None:
        pytest.skip("no local database server reachable")

    try:
        processor = MapReduceProcessor(
            llm=PhaseAwareFakeChatModel(
                merge_response="[MERGE RESULT] " + "; ".join(KEY_CONCEPTS) * 30
            ),
            store=store,
            max_chunk_tokens=400,
            max_context_tokens=300,  # force repair path too
        )
        result = processor.process(
            make_doc(KEY_CONCEPTS, repeats=30),
            task="Summarize key findings",
            session_id=f"e2e-db_{uuid.uuid4().hex[:8]}",
        )
        assert result["num_chunks"] >= 2, kind
        assert result["repair_attempts"] >= 1, kind
        final = result["final_result"].lower()
        missing = [c for c in KEY_CONCEPTS if c.lower() not in final]
        assert not missing, f"{kind}: missing concepts {missing}"
        assert processor.search_session(result["session_id"], "AlphaFold"), kind
    finally:
        if hasattr(store, "client"):
            store.client.drop_database(store.collection.database.name)
            store.close()
        else:
            store.delete_session("probe")