"""MSSQL + Oracle dialects: exact SQL generation (live-server verification
pending by design — see CHANGELOG)."""
from __future__ import annotations

import pytest

from longloom import MSSQL_DIALECT, ORACLE_DIALECT
from longloom.stores.naming import FieldMap


COLS = FieldMap(table="app_chunks").all_columns
CUSTOM = FieldMap(table="app_chunks",
                  columns={"id": "chunk_pk", "content": "body"},
                  namespace="tenant-1", namespace_field="app_name").all_columns


def test_mssql_ddl_uses_sql_server_idioms():
    stmts = MSSQL_DIALECT.ddl("app_chunks", COLS, None)
    assert len(stmts) == 2  # table + index, both existence-guarded
    table_stmt = stmts[0]
    assert table_stmt.startswith("IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'app_chunks')")
    assert "CREATE TABLE app_chunks" in table_stmt
    assert "NVARCHAR(255) PRIMARY KEY" in table_stmt and "NVARCHAR(MAX)" in table_stmt
    assert "CREATE INDEX idx_app_chunks_session" in stmts[1]


def test_mssql_upsert_is_a_merge():
    sql = MSSQL_DIALECT.upsert("app_chunks", COLS, None)
    assert sql.startswith("MERGE INTO app_chunks WITH (HOLDLOCK) AS t")
    assert "USING (SELECT %s AS id" in sql
    assert "ON t.id = s.id" in sql
    assert "WHEN MATCHED THEN UPDATE SET t.content_hash = s.content_hash" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql
    assert sql.count("%s") == len(COLS)
    assert sql.rstrip().endswith(";")


def test_mssql_custom_columns_and_namespace():
    stmts = MSSQL_DIALECT.ddl("app_chunks", CUSTOM, "app_name")
    assert "chunk_pk NVARCHAR(255) PRIMARY KEY" in stmts[0]
    assert "body NVARCHAR(MAX)" in stmts[0]
    assert "app_name NVARCHAR(128) NOT NULL DEFAULT ''" in stmts[0]
    sql = MSSQL_DIALECT.upsert("app_chunks", CUSTOM, "app_name")
    assert "ON t.chunk_pk = s.chunk_pk" in sql and "app_name" in sql


def test_oracle_ddl_is_a_plsql_block_with_guards():
    stmts = ORACLE_DIALECT.ddl("app_chunks", COLS, None)
    assert len(stmts) == 2
    for stmt in stmts:
        assert stmt.startswith("BEGIN\n  EXECUTE IMMEDIATE 'CREATE ")
        assert "WHEN OTHERS THEN" in stmt and "-955" in stmt
    assert "NVARCHAR2(255) PRIMARY KEY" in stmts[0] and "CLOB" in stmts[0]
    assert "NUMBER(10)" in stmts[0]


def test_oracle_upsert_uses_numbered_binds_and_dual():
    sql = ORACLE_DIALECT.upsert("app_chunks", COLS, None)
    assert sql.startswith("MERGE INTO app_chunks t USING (SELECT :1 AS id")
    assert "FROM dual" in sql
    assert "ON (t.id = s.id)" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql
    for i in range(1, len(COLS) + 1):
        assert f":{i}" in sql
    assert ORACLE_DIALECT.bind(2) == ":2"


def test_oracle_custom_columns_and_namespace():
    sql = ORACLE_DIALECT.upsert("app_chunks", CUSTOM, "app_name")
    assert ":1 AS chunk_pk" in sql and "ON (t.chunk_pk = s.chunk_pk)" in sql
    ddl = "\n".join(ORACLE_DIALECT.ddl("app_chunks", CUSTOM, "app_name"))
    assert "chunk_pk NVARCHAR2(255)" in ddl and "body CLOB" in ddl


class RecordingConnection:
    """Captures SQL + params — proves bind generation without a real Oracle."""

    def __init__(self):
        self.statements: list[tuple[str, list]] = []
        self.canned_row: tuple = ("nb_0000_h", "h", '{}', 5, "2026-01-01",
                                  0, 1, "nb", "numbered bind check 123")

    def execute(self, sql, params=None):
        self.statements.append((sql, list(params or [])))
        return self

    def fetchall(self):
        return [self.canned_row]

    def fetchone(self):
        return (0, 0, None, None)

    def commit(self):
        pass

    def close(self):
        pass


def test_numbered_binds_flow_into_sql_store_queries():
    """Oracle numbered binds must produce :1/:2/:3 sequences inside
    SqlContextStore queries, with params positional and ordered."""
    from longloom import SqlContextStore
    from longloom.stores.sql import ORACLE_DIALECT

    conn = RecordingConnection()
    store = SqlContextStore(
        connect=lambda: conn, dialect=ORACLE_DIALECT,
        table="app_chunks", columns={"id": "chunk_pk", "content": "body"},
        namespace="tenant-1", namespace_field="app_name", init_schema=False,
    )
    chunks = store.store_context("numbered bind check 123", session_id="nb",
                                 max_chunk_tokens=100)
    assert len(chunks) == 1
    loaded = store.retrieve_context("nb", chunk_indices=[0])
    assert loaded and loaded[0].content == "numbered bind check 123"

    sqls = [sql for sql, _ in conn.statements]
    load_sql, load_params = next((sql, p) for sql, p in conn.statements
                                 if sql.startswith("SELECT"))
    # numbered, gap-free, in order
    import re
    binds = re.findall(r":(\d+)", load_sql)
    assert binds == [str(i) for i in range(1, len(binds) + 1)]
    assert load_params[0] == "nb" and "tenant-1" in load_params
    assert "app_chunks" in load_sql and "chunk_pk" in load_sql


def test_create_store_routes_mssql_oracle_schemes():
    """Factory routes the schemes to the right dialects; drivers are eagerly
    required at construction (this machine has none — the ImportError proves
    the routing reached the engine-specific branch)."""
    from longloom import create_store

    with pytest.raises(ImportError, match="pymssql"):
        create_store("mssql://sa:pw@localhost:1433/longloom")
    with pytest.raises(ImportError, match="oracledb"):
        create_store("oracle://system:pw@host:1521/ORCLPDB1")
