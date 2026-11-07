"""create_store(url) — URL factory for every backend (no external servers)."""
from __future__ import annotations

import sys
import types
import uuid

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
        create_store("crystaldb://magic")


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

    from longloom.stores.naming import FieldMap

    cols = FieldMap(table="t").all_columns          # full logical → physical map
    custom = FieldMap(table="t", columns={"id": "chunk_id", "content": "body"},
                      namespace="app-1", namespace_field="app_name").all_columns

    pg = POSTGRES_DIALECT.upsert("t", cols, None)
    assert pg.startswith("INSERT INTO t (id, content_hash") and "ON CONFLICT (id) DO UPDATE" in pg
    assert "EXCLUDED.content_hash" in pg and pg.count("%s") == len(cols)

    pg_custom = POSTGRES_DIALECT.upsert("t", custom, "app_name")
    assert pg_custom.startswith("INSERT INTO t (chunk_id") and "app_name" in pg_custom
    assert "ON CONFLICT (chunk_id)" in pg_custom

    my = MYSQL_DIALECT.upsert("t", cols, None)
    assert "ON DUPLICATE KEY UPDATE" in my and "VALUES(content)" in my and "%s" in my
    assert POSTGRES_DIALECT.placeholder == "%s"
    ddl_pg = "\n".join(POSTGRES_DIALECT.ddl("t", custom, "app_name"))
    assert "chunk_id TEXT PRIMARY KEY" in ddl_pg and "body TEXT" in ddl_pg
    assert "app_name TEXT NOT NULL DEFAULT ''" in ddl_pg
    assert any("MEDIUMTEXT" in stmt for stmt in MYSQL_DIALECT.ddl("t", cols, None))


class _FakeConn:
    """Minimal conn with .execute — passes _as_executable through unchanged."""

    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return []

    def exec_driver_sql(self, sql, params=None):
        return self.execute(sql, params)

    def commit(self):
        pass

    def close(self):
        pass


def test_factory_driver_import_errors(monkeypatch):
    """With sqlalchemy AND every native driver missing, each engine's
    fallback raises a helpful ImportError naming the extra to install."""
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)

    monkeypatch.setitem(sys.modules, "psycopg2", None)
    monkeypatch.setitem(sys.modules, "psycopg", None)
    with pytest.raises(ImportError, match="psycopg2-binary"):
        create_store("postgresql://u:p@h/db")

    monkeypatch.setitem(sys.modules, "pymysql", None)
    monkeypatch.setitem(sys.modules, "MySQLdb", None)
    with pytest.raises(ImportError, match="pymysql or mysqlclient"):
        create_store("mysql://u:p@h/db")

    monkeypatch.setitem(sys.modules, "pymssql", None)
    with pytest.raises(ImportError, match="pymssql"):
        create_store("mssql://u:p@h/db")

    monkeypatch.setitem(sys.modules, "oracledb", None)
    with pytest.raises(ImportError, match="oracledb"):
        create_store("oracle://u:p@h/svc")


def test_factory_sa_adapter_commit_rollback_close(monkeypatch):
    """Store operations run through the SQLAlchemy adapter: exec_driver_sql on
    persist, commit via _persist, rollback/close reachable on the adapter."""
    import sqlalchemy as sa

    from longloom import create_store

    class FakeSaConn:
        def __init__(self):
            self.executed = []
            self.rolled = False
            self.committed = False
            self.closed = False

        def exec_driver_sql(self, sql, params=None):
            self.executed.append((sql, params))

        def cursor(self):
            return types.SimpleNamespace(execute=lambda sql, params=None: None,
                                         close=lambda: None)

        def commit(self):
            self.committed = True

        def rollback(self):
            self.rolled = True

        def close(self):
            self.closed = True

    conns = []

    def fake_create_engine(url, **kw):
        class E:
            def connect(self_inner):
                c = FakeSaConn()
                conns.append(c)
                return c
        return E()

    monkeypatch.setattr(sa, "create_engine", fake_create_engine)
    store = create_store(f"postgresql://u:p@h/db_{uuid.uuid4().hex[:6]}")
    store.store_context("sa adapter ops 123", session_id="sa", max_chunk_tokens=100)
    assert conns and conns[-1].committed
    store.conn.rollback()          # adapter rollback reachable
    assert conns[-1].rolled
    store.close()
    assert conns[-1].closed


def test_factory_mysql_normalization_without_any_driver(monkeypatch):
    """No MySQLdb AND no pymysql: URL stays unnormalized, SA reports it."""
    import sqlalchemy as sa

    seen = {}

    def fake_create_engine(url, **kw):
        seen["url"] = url
        return types.SimpleNamespace(connect=lambda: _FakeConn())

    monkeypatch.setattr(sa, "create_engine", fake_create_engine)
    monkeypatch.setitem(sys.modules, "MySQLdb", None)
    monkeypatch.setitem(sys.modules, "pymysql", None)
    from longloom.factory import _sql_connect_factory

    _sql_connect_factory("mysql://u:p@h/db", "mysql")
    assert seen["url"] == "mysql://u:p@h/db"      # unchanged — SA reports the gap


def test_invalid_table_name_rejected():
    import pytest

    import sqlite3
    from longloom import SQLITE_DIALECT, SqlContextStore

    with pytest.raises(ValueError, match="invalid table/collection name"):
        SqlContextStore(
            connect=lambda: sqlite3.connect(":memory:"),
            dialect=SQLITE_DIALECT,
            table="chunks; DROP TABLE users",  # injection attempt
        )