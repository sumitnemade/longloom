"""SQLite storage backend — single-file persistence.

A thin specialization of the generic :class:`SqlContextStore` (same dialect
machinery), so it supports the full embedding configuration: custom table
name, custom column names, and a ``namespace`` discriminator for sharing one
database file across applications.

Two content modes:
* ``content_dir=None`` (default): chunk text is stored inline in the DB —
  fully portable single file, no sidecar files.
* ``content_dir=<path>``: chunk text is written to hash-addressed files and
  the DB keeps only metadata.
"""
from __future__ import annotations

from typing import Dict, Optional

from .sql import SQLITE_DIALECT, SqlContextStore

__all__ = ["SQLiteContextStore"]


class SQLiteContextStore(SqlContextStore):
    def __init__(
        self,
        db_path: str = "longloom.db",
        content_dir: Optional[str] = None,
        table: str = "context_chunks",
        columns: Optional[Dict[str, str]] = None,
        namespace: Optional[str] = None,
        namespace_field: str = "namespace",
        init_schema: bool = True,
        **kwargs,
    ):
        import sqlite3

        super().__init__(
            connect=lambda: sqlite3.connect(db_path),
            dialect=SQLITE_DIALECT,
            table=table,
            columns=columns,
            namespace=namespace,
            namespace_field=namespace_field,
            content_dir=content_dir,
            init_schema=init_schema,
            **kwargs,
        )
