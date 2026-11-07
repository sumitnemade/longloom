"""create_store(url) — URL factory for every backend (no external servers)."""
from __future__ import annotations

import pytest

from longloom import (
    InMemoryContextStore,
    JSONFileContextStore,
    SQLiteContextStore,
    create_store,
)
from store_contract import run_store_contract


def test_memory_url():
    store = create_store("memory://")
    assert isinstance(store, InMemoryContextStore)
    run_store_contract(store, "memory://")


def test_json_url(tmp_path):
    store = create_store(f"json://{tmp_path / 'j'}")
    assert isinstance(store, JSONFileContextStore)
    run_store_contract(store, "json://")


def test_sqlite_file_url(tmp_path):
    db = tmp_path / "f.db"
    store = create_store(f"sqlite:///{db}")
    assert isinstance(store, SQLiteContextStore)
    assert db.exists()
    run_store_contract(store, "sqlite-file")


def test_sqlite_memory_url():
    store = create_store("sqlite://")
    assert isinstance(store, SQLiteContextStore)
    run_store_contract(store, "sqlite-memory")


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="Unsupported store URL"):
        create_store("oracle://thin:something")


def test_empty_url_raises():
    with pytest.raises(ValueError):
        create_store("")


def test_sqlite_roundtrip_e2e_with_processor(tmp_path):
    """Factory-created store works transparently inside MapReduceProcessor."""
    from longloom import MapReduceProcessor
    from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc

    processor = MapReduceProcessor(
        llm=PhaseAwareFakeChatModel(),
        store=create_store(f"sqlite:///{tmp_path / 'p.db'}"),
        max_chunk_tokens=400,
        max_context_tokens=8192,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30), task="Summarize")
    assert result["num_chunks"] >= 2
    assert processor.get_session_info(result["session_id"])["total_tokens"] == \
        result["total_input_tokens"]


def test_generic_sql_store_via_sqlite_dialect():
    """The generic SqlContextStore (Postgres/MySQL code path) fully exercised.

    Runs the shared contract through the generic DB-API backend using the
    SQLite dialect — same code that serves PostgreSQL/MySQL, different dialect.
    """
    import sqlite3

    from longloom import SQLITE_DIALECT, SqlContextStore
    from store_contract import run_store_contract

    store = SqlContextStore(
        connect=lambda: sqlite3.connect(":memory:"),
        dialect=SQLITE_DIALECT,
        table="longloom_generic_test",
    )
    run_store_contract(store, "sql-generic/sqlite-dialect")
    store.close()


def test_sql_dialect_generation():
    """Postgres/MySQL dialects generate correct upsert + DDL SQL."""
    from longloom import MYSQL_DIALECT, POSTGRES_DIALECT

    cols = ["id", "content_hash", "content"]
    pg = POSTGRES_DIALECT.upsert("t", cols)
    assert pg == (
        "INSERT INTO t (id, content_hash, content) VALUES (%s, %s, %s) "
        "ON CONFLICT (id) DO UPDATE SET content_hash = EXCLUDED.content_hash, "
        "content = EXCLUDED.content"
    )
    my = MYSQL_DIALECT.upsert("t", cols)
    assert "ON DUPLICATE KEY UPDATE" in my and "VALUES(content)" in my
    assert "%s" in my
    assert POSTGRES_DIALECT.placeholder == "%s"
    assert any("CREATE TABLE IF NOT EXISTS t" in stmt for stmt in POSTGRES_DIALECT.ddl("t"))
    assert any("MEDIUMTEXT" in stmt for stmt in MYSQL_DIALECT.ddl("t"))


def test_invalid_table_name_rejected():
    import pytest

    import sqlite3
    from longloom import SQLITE_DIALECT, SqlContextStore

    with pytest.raises(ValueError, match="invalid table name"):
        SqlContextStore(
            connect=lambda: sqlite3.connect(":memory:"),
            dialect=SQLITE_DIALECT,
            table="chunks; DROP TABLE users",  # injection attempt
        )