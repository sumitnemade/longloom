"""Generic SQL storage backend — one implementation for any SQL database.

Works with any DB-API 2.0 connection (PEP 249). Dialects provided for
PostgreSQL, MySQL/MariaDB, and SQLite; any other DB-API database can be used
by supplying a custom :class:`SqlDialect`.

    from longloom.stores.sql import SqlContextStore, POSTGRES_DIALECT
    import psycopg2
    store = SqlContextStore(
        connect=lambda: psycopg2.connect("postgresql://user@/mydb"),
        dialect=POSTGRES_DIALECT,
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
from ..chunks import ContextChunk

__all__ = ["SqlDialect", "SqlContextStore", "POSTGRES_DIALECT", "MYSQL_DIALECT", "SQLITE_DIALECT"]

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_COLS = (
    "id", "content_hash", "metadata", "token_count", "created_at",
    "chunk_index", "total_chunks", "session_id", "content",
)


@dataclass(frozen=True)
class SqlDialect:
    name: str
    placeholder: str                      # "?" (sqlite) / "%s" (pg, mysql)
    ddl: Callable[[str], List[str]]       # table -> CREATE statements
    upsert: Callable[[str, List[str]], str]  # (table, cols) -> upsert SQL


def _ddl_postgres(table: str) -> List[str]:
    return [
        f"""CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            metadata TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            total_chunks INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            content TEXT
        )""",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_session ON {table} (session_id, chunk_index)",
    ]


def _upsert_postgres(table: str, cols: List[str]) -> str:
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id")
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('%s' for _ in cols)}) "
        f"ON CONFLICT (id) DO UPDATE SET {updates}"
    )


def _ddl_mysql(table: str) -> List[str]:
    return [
        f"""CREATE TABLE IF NOT EXISTS {table} (
            id VARCHAR(255) PRIMARY KEY,
            content_hash VARCHAR(64) NOT NULL,
            metadata TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at VARCHAR(64) NOT NULL,
            chunk_index INTEGER NOT NULL,
            total_chunks INTEGER NOT NULL,
            session_id VARCHAR(255) NOT NULL,
            content MEDIUMTEXT,
            INDEX idx_{table}_session (session_id, chunk_index)
        )""",
    ]


def _upsert_mysql(table: str, cols: List[str]) -> str:
    updates = ", ".join(f"{c} = VALUES({c})" for c in cols if c != "id")
    return (
        f"INSERT INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('%s' for _ in cols)}) "
        f"ON DUPLICATE KEY UPDATE {updates}"
    )


def _ddl_sqlite(table: str) -> List[str]:
    return [
        f"""CREATE TABLE IF NOT EXISTS {table} (
            id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            metadata TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            total_chunks INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            content TEXT
        )""",
        f"CREATE INDEX IF NOT EXISTS idx_{table}_session ON {table} (session_id, chunk_index)",
    ]


def _upsert_sqlite(table: str, cols: List[str]) -> str:
    return (
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )


POSTGRES_DIALECT = SqlDialect("postgres", "%s", _ddl_postgres, _upsert_postgres)
MYSQL_DIALECT = SqlDialect("mysql", "%s", _ddl_mysql, _upsert_mysql)
SQLITE_DIALECT = SqlDialect("sqlite", "?", _ddl_sqlite, _upsert_sqlite)


class SqlContextStore(BaseContextStore):
    """Portable SQL backend over any DB-API 2.0 connection.

    ``content_dir=None`` keeps chunk text inline in the table (default, fully
    self-contained). Pass ``content_dir=...`` to store text in hash-addressed
    files instead — the same layout the legacy SQLite backend used.
    """

    def __init__(
        self,
        connect: Callable[[], Any],
        dialect: SqlDialect,
        table: str = "context_chunks",
        content_dir: Optional[str] = None,
        init_schema: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not _NAME_RE.match(table):
            raise ValueError(f"invalid table name: {table!r}")
        self._connect = connect
        self.dialect = dialect
        self.table = table
        self.content_dir = content_dir
        if content_dir:
            os.makedirs(content_dir, exist_ok=True)
        self.conn = connect()
        if init_schema:
            self.init_schema()

    def init_schema(self) -> None:
        for statement in self.dialect.ddl(self.table):
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
        return [
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

    def _from_row(self, row: Any) -> ContextChunk:
        (
            cid, content_hash, metadata_json, token_count, created_at,
            chunk_index, total_chunks, _session_id, content,
        ) = row
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
        sql = self.dialect.upsert(self.table, list(_COLS))
        ph = self.dialect.placeholder
        for chunk in chunks:
            self.conn.execute(sql, self._to_row(chunk))
        self.conn.commit()

    def _load_chunks(
        self,
        session_id: str,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        ph = self.dialect.placeholder
        sql = f"SELECT {', '.join(_COLS)} FROM {self.table} WHERE session_id = {ph}"
        params: List[Any] = [session_id]
        if chunk_indices is not None:
            if not chunk_indices:
                return []
            marks = ", ".join(ph for _ in chunk_indices)
            sql += f" AND chunk_index IN ({marks})"
            params += list(chunk_indices)
        sql += " ORDER BY chunk_index"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def _delete_session(self, session_id: str) -> None:
        ph = self.dialect.placeholder
        if self.content_dir:
            rows = self.conn.execute(
                f"SELECT content_hash FROM {self.table} WHERE session_id = {ph}",
                [session_id],
            ).fetchall()
            for (content_hash,) in rows:
                path = self._content_path(content_hash)
                if os.path.exists(path):
                    os.remove(path)
        self.conn.execute(
            f"DELETE FROM {self.table} WHERE session_id = {ph}", [session_id]
        )
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
        row = self.conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(token_count), 0), "
            f"MIN(created_at), MAX(created_at) "
            f"FROM {self.table} WHERE session_id = {ph}",
            [session_id],
        ).fetchone()
        count, tokens, first, last = row
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

    def _content_path(self, content_hash: str) -> str:
        return os.path.join(self.content_dir or "", f"{content_hash}.txt")