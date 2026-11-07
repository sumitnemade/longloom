"""Generic SQL storage backend — one implementation for any SQL database.

Works with any DB-API 2.0 connection (PEP 249). Dialects provided for
PostgreSQL, MySQL/MariaDB, and SQLite; any other DB-API database can be used
by supplying a custom :class:`SqlDialect`.

Embedding in an existing database::

    store = SqlContextStore(
        connect=lambda: psycopg2.connect(app_dsn),
        dialect=POSTGRES_DIALECT,
        table="app_chunks",                        # your table
        columns={"id": "chunk_id", "content": "body"},  # your columns
        namespace="support-agent",                 # share the table across apps
        init_schema=False,                         # your migrations own the DDL
    )

Or let :func:`longloom.create_store` build it from a URL (uses SQLAlchemy when
installed, else tries known drivers directly).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .base import BaseContextStore
from .naming import LOGICAL_FIELDS, FieldMap
from ..chunks import ContextChunk

__all__ = [
    "SqlDialect", "SqlContextStore",
    "POSTGRES_DIALECT", "MYSQL_DIALECT", "SQLITE_DIALECT",
]

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class SqlDialect:
    name: str
    placeholder: str                                   # "?" (sqlite) / "%s" (pg, mysql)
    ddl: Callable[[str, Dict[str, str], Optional[str]], List[str]]
    upsert: Callable[[str, Dict[str, str], Optional[str]], str]


def _ddl_postgres(table: str, c: Dict[str, str], ns: Optional[str]) -> List[str]:
    defs = [
        f"{c['id']} TEXT PRIMARY KEY",
        f"{c['content_hash']} TEXT NOT NULL",
        f"{c['metadata']} TEXT NOT NULL",
        f"{c['token_count']} INTEGER NOT NULL",
        f"{c['created_at']} TEXT NOT NULL",
        f"{c['chunk_index']} INTEGER NOT NULL",
        f"{c['total_chunks']} INTEGER NOT NULL",
        f"{c['session_id']} TEXT NOT NULL",
        f"{c['content']} TEXT",
    ]
    if ns:
        defs.append(f"{ns} TEXT NOT NULL DEFAULT ''")
    return [
        f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(defs)})",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_session ON {table} ({c['session_id']}, {c['chunk_index']})",
    ]


def _upsert_postgres(table: str, c: Dict[str, str], ns: Optional[str]) -> str:
    cols = list(c.values())
    updates = ", ".join(f"{col} = EXCLUDED.{col}" for col in cols if col != c["id"])
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('%s' for _ in cols)}) "
        f"ON CONFLICT ({c['id']}) DO UPDATE SET {updates}"
    )


def _ddl_mysql(table: str, c: Dict[str, str], ns: Optional[str]) -> List[str]:
    defs = [
        f"{c['id']} VARCHAR(255) PRIMARY KEY",
        f"{c['content_hash']} VARCHAR(64) NOT NULL",
        f"{c['metadata']} TEXT NOT NULL",
        f"{c['token_count']} INTEGER NOT NULL",
        f"{c['created_at']} VARCHAR(64) NOT NULL",
        f"{c['chunk_index']} INTEGER NOT NULL",
        f"{c['total_chunks']} INTEGER NOT NULL",
        f"{c['session_id']} VARCHAR(255) NOT NULL",
        f"{c['content']} MEDIUMTEXT",
    ]
    if ns:
        defs.append(f"{ns} VARCHAR(128) NOT NULL DEFAULT ''")
    return [
        f"""CREATE TABLE IF NOT EXISTS {table} ({', '.join(defs)},
            INDEX idx_{table}_session ({c['session_id']}, {c['chunk_index']}))""",
    ]


def _upsert_mysql(table: str, c: Dict[str, str], ns: Optional[str]) -> str:
    cols = list(c.values())
    updates = ", ".join(f"{col} = VALUES({col})" for col in cols if col != c["id"])
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('%s' for _ in cols)}) "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )


def _ddl_sqlite(table: str, c: Dict[str, str], ns: Optional[str]) -> List[str]:
    defs = [
        f"{c['id']} TEXT PRIMARY KEY",
        f"{c['content_hash']} TEXT NOT NULL",
        f"{c['metadata']} TEXT NOT NULL",
        f"{c['token_count']} INTEGER NOT NULL",
        f"{c['created_at']} TEXT NOT NULL",
        f"{c['chunk_index']} INTEGER NOT NULL",
        f"{c['total_chunks']} INTEGER NOT NULL",
        f"{c['session_id']} TEXT NOT NULL",
        f"{c['content']} TEXT",
    ]
    if ns:
        defs.append(f"{ns} TEXT NOT NULL DEFAULT ''")
    return [
        f"CREATE TABLE IF NOT EXISTS {table} ({', '.join(defs)})",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_session ON {table} ({c['session_id']}, {c['chunk_index']})",
    ]


def _upsert_sqlite(table: str, c: Dict[str, str], ns: Optional[str]) -> str:
    cols = list(c.values())
    return (
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )


POSTGRES_DIALECT = SqlDialect("postgres", "%s", _ddl_postgres, _upsert_postgres)
MYSQL_DIALECT = SqlDialect("mysql", "%s", _ddl_mysql, _upsert_mysql)
SQLITE_DIALECT = SqlDialect("sqlite", "?", _ddl_sqlite, _upsert_sqlite)


def _as_executable(conn: Any) -> Any:
    """Adapt a raw DB-API connection so ``.execute()`` is available on it.

    sqlite3 and psycopg(v3) connections expose ``execute``; psycopg2, pymysql
    and MySQLdb only expose cursors (and pymysql's cursor.execute returns an
    int, not the cursor) — normalize all of them here so direct construction
    with a raw corporate-DB connection just works.
    """
    if hasattr(conn, "execute"):
        return conn

    class _Executable:
        def __init__(self, raw: Any):
            self._raw = raw
            self._cursor = raw.cursor()

        def execute(self, sql: str, params: Optional[list] = None) -> Any:
            self._cursor.execute(sql, params or [])
            return self._cursor  # pymysql returns an int; normalize

        def commit(self) -> None:
            self._raw.commit()

        def rollback(self) -> None:
            self._raw.rollback()

        def close(self) -> None:
            try:
                self._cursor.close()
            finally:
                self._raw.close()

    return _Executable(conn)


class SqlContextStore(BaseContextStore):
    """Portable SQL backend over any DB-API 2.0 connection.

    ``content_dir=None`` keeps chunk text inline in the table (default, fully
    self-contained). Pass ``content_dir=...`` to store text in hash-addressed
    files instead.

    Embedding in an existing database: pass ``table``, ``columns`` (logical →
    physical names) and ``namespace`` (a discriminator so several apps share
    one table). See :mod:`longloom.stores.naming`.
    """

    def __init__(
        self,
        connect: Callable[[], Any],
        dialect: SqlDialect,
        table: str = "context_chunks",
        columns: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
        namespace_field: str = "namespace",
        content_dir: Optional[str] = None,
        init_schema: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.names = FieldMap(table=table, columns=columns or {}, namespace=namespace,
                              namespace_field=namespace_field)
        self._connect = connect
        self.dialect = dialect
        self.table = self.names.table
        self.content_dir = content_dir
        if content_dir:
            os.makedirs(content_dir, exist_ok=True)
        self.conn = _as_executable(connect())
        if init_schema:
            self.init_schema()

    # -- schema -----------------------------------------------------------

    def ddl(self) -> List[str]:
        """CREATE statements for the (possibly custom-named) table — for your
        own migration tooling."""
        return self.dialect.ddl(self.table, self.names.all_columns, self.names.namespace_field)

    def init_schema(self) -> None:
        for statement in self.ddl():
            self.conn.execute(statement)
        self.conn.commit()

    # -- row (de)serialization -------------------------------------------

    def _to_row(self, chunk: ContextChunk) -> List[Any]:
        content = chunk.content
        if self.content_dir:
            path = self._content_path(chunk.metadata["content_hash"])
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(chunk.content)
            content = None
        session_id = chunk.id.rsplit("_", 2)[0]
        row: List[Any] = [
            chunk.id,
            chunk.metadata["content_hash"],
            json.dumps(chunk.metadata),
            chunk.token_count,
            chunk.created_at,
            chunk.chunk_index,
            chunk.total_chunks,
            session_id,
            content,
        ]
        if self.names.namespace is not None:
            row.append(self.names.namespace)
        return row

    def _from_row(self, row: Any) -> ContextChunk:
        (
            cid, content_hash, metadata_json, token_count, created_at,
            chunk_index, total_chunks, _session_id, content,
        ) = row[:9]
        if content is None and self.content_dir:
            path = self._content_path(content_hash)
            content = ""
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    content = fh.read()
        return ContextChunk(
            id=cid,
            content=content or "",
            metadata=json.loads(metadata_json),
            token_count=token_count,
            created_at=created_at,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
        )

    # -- backend primitives ------------------------------------------------

    def _persist(self, chunks: List[ContextChunk]) -> None:
        sql = self.dialect.upsert(self.table, self.names.all_columns, self.names.namespace_field)
        for chunk in chunks:
            self.conn.execute(sql, self._to_row(chunk))
        self.conn.commit()

    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        c = self.names.all_columns
        select_cols = [c[name] for name in LOGICAL_FIELDS]
        sql = f"SELECT {', '.join(select_cols)} FROM {self.table} WHERE {c['session_id']} = {self.dialect.placeholder}"
        params: List[Any] = [session_id]
        if self.names.namespace is not None:
            sql += f" AND {c['namespace']} = {self.dialect.placeholder}"
            params.append(self.names.namespace)
        if chunk_indices is not None:
            if not chunk_indices:
                return []
            marks = ", ".join(self.dialect.placeholder for _ in chunk_indices)
            sql += f" AND {c['chunk_index']} IN ({marks})"
            params += list(chunk_indices)
        sql += f" ORDER BY {c['chunk_index']}"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def _delete_session(self, session_id: str) -> None:
        ph = self.dialect.placeholder
        c = self.names.all_columns
        if self.content_dir:
            sql = f"SELECT {c['content_hash']} FROM {self.table} WHERE {c['session_id']} = {ph}"
            params: List[Any] = [session_id]
            if self.names.namespace is not None:
                sql += f" AND {c['namespace']} = {ph}"
                params.append(self.names.namespace)
            for (content_hash,) in self.conn.execute(sql, params).fetchall():
                path = self._content_path(content_hash)
                if os.path.exists(path):
                    os.remove(path)
        sql = f"DELETE FROM {self.table} WHERE {c['session_id']} = {ph}"
        params = [session_id]
        if self.names.namespace is not None:
            sql += f" AND {c['namespace']} = {ph}"
            params.append(self.names.namespace)
        self.conn.execute(sql, params)
        self.conn.commit()

    # -- search / info --------------------------------------------------------

    def search_context(
        self,
        session_id: str,
        query: str,
        limit: int = 10,
    ) -> List[ContextChunk]:
        q = query.lower()
        hits: List[ContextChunk] = []
        for chunk in self._load_chunks(session_id):
            if q in (chunk.content or "").lower():
                hits.append(chunk)
                if len(hits) >= limit:
                    break
        return hits

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        ph = self.dialect.placeholder
        c = self.names.all_columns
        sql = (f"SELECT COUNT(*), COALESCE(SUM({c['token_count']}), 0), "
               f"MIN({c['created_at']}), MAX({c['created_at']}) "
               f"FROM {self.table} WHERE {c['session_id']} = {ph}")
        params: List[Any] = [session_id]
        if self.names.namespace is not None:
            sql += f" AND {c['namespace']} = {ph}"
            params.append(self.names.namespace)
        count, tokens, first, last = self.conn.execute(sql, params).fetchone()
        if not count:
            return {"session_id": session_id, "chunk_count": 0, "total_tokens": 0}
        return {
            "session_id": session_id,
            "chunk_count": count,
            "total_tokens": tokens,
            "first_chunk_at": first,
            "last_chunk_at": last,
        }

    def close(self) -> None:
        try:
            self.conn.commit()
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    def _content_path(self, content_hash: str) -> str:
        return os.path.join(self.content_dir or "", f"{content_hash}.txt")