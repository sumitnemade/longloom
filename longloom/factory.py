"""``create_store(url)`` — build any supported backend from a connection string.

Supported URL schemes:

==========================  ==============================================
``memory://``               :class:`InMemoryContextStore`
``json://<dir>``            :class:`JSONFileContextStore`
``sqlite:///<path>``        :class:`SQLiteContextStore` (``sqlite://`` = in-memory)
``postgresql://<dsn>``      :class:`SqlContextStore` (PostgreSQL dialect)
``mysql://<dsn>``           :class:`SqlContextStore` (MySQL/MariaDB dialect)
``mongodb://<uri>``         :class:`MongoContextStore` (path = database name)
``https://<proj>.supabase.co``  :class:`SupabaseContextStore` (key via SUPABASE_KEY)
==========================  ==============================================

SQL URLs prefer SQLAlchemy when installed (pooled, works with every driver);
otherwise known drivers (psycopg2/psycopg, pymysql/MySQLdb) are tried directly.
"""
from __future__ import annotations

import os
import re
from typing import Any, Callable, Optional
from urllib.parse import unquote, urlparse

from .stores.base import BaseContextStore
from .stores.memory import InMemoryContextStore

__all__ = ["create_store", "SUPPORTED_SCHEMES"]

SUPPORTED_SCHEMES = (
    "memory", "json", "sqlite", "postgres", "postgresql", "mysql", "mariadb",
    "mongodb", "mongodb+srv", "supabase", "https",
)


def create_store(
    url: str,
    table: str = "context_chunks",
    columns: Optional[Dict[str, str]] = None,
    namespace: Optional[str] = None,
    **kwargs,
) -> BaseContextStore:
    """Create a storage backend from a connection string (see module docstring).

    ``table``, ``columns`` (logical → physical names) and ``namespace`` (a
    discriminator letting several apps share one table/collection) embed
    LongLoom into existing databases — supported by the SQL, MongoDB and
    Supabase backends; ``memory`` and ``json`` ignore them.
    """
    if not url:
        raise ValueError("url must be a non-empty connection string")
    import re as _re

    if table and not _re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise ValueError(f"invalid table/collection name: {table!r}")
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()

    if scheme == "memory":
        return InMemoryContextStore(**kwargs)

    if scheme == "json":
        root = parsed.netloc + parsed.path if parsed.netloc else parsed.path
        from .stores.jsonfile import JSONFileContextStore

        return JSONFileContextStore(root_dir=unquote(root) or "longloom_data", **kwargs)

    if scheme == "sqlite":
        path = parsed.path or ""
        if parsed.netloc and parsed.netloc != ":memory:":
            path = parsed.netloc + path
        db_path = unquote(path) if path and path != ":memory:" else ":memory:"
        from .stores.sqlite import SQLiteContextStore

        return SQLiteContextStore(db_path=db_path, table=table, columns=columns,
                                  namespace=namespace, **kwargs)

    if scheme in ("postgres", "postgresql"):
        from .stores.sql import POSTGRES_DIALECT, SqlContextStore

        return SqlContextStore(
            connect=_sql_connect_factory(url, "postgres"),
            dialect=POSTGRES_DIALECT,
            table=table, columns=columns, namespace=namespace,
            **kwargs,
        )

    if scheme in ("mysql", "mariadb"):
        from .stores.sql import MYSQL_DIALECT, SqlContextStore

        return SqlContextStore(
            connect=_sql_connect_factory(url, "mysql"),
            dialect=MYSQL_DIALECT,
            table=table, columns=columns, namespace=namespace,
            **kwargs,
        )

    if scheme in ("mongodb", "mongodb+srv"):
        from .stores.mongo import MongoContextStore

        db_name = unquote(parsed.path.lstrip("/")) or kwargs.pop("db_name", "longloom")
        return MongoContextStore(client_or_uri=url, db_name=db_name,
                                 collection=kwargs.pop("collection", table),
                                 columns=columns, namespace=namespace, **kwargs)

    if scheme == "supabase":
        from .stores.supabase import SupabaseContextStore

        host = parsed.netloc
        # supabase://x.supabase.co → https://x.supabase.co (cloud);
        # supabase://127.0.0.1:8905 → http://… (local PostgREST stub).
        project_url = f"https://{host}" if host.endswith(".supabase.co") else f"http://{host}"
        return SupabaseContextStore(url=project_url, table=table, columns=columns,
                                    namespace=namespace, **kwargs)

    if scheme == "https" and re.search(r"\.supabase\.", parsed.netloc or ""):
        from .stores.supabase import SupabaseContextStore

        return SupabaseContextStore(url=url, table=table, **kwargs)

    raise ValueError(
        f"Unsupported store URL scheme: {scheme!r}. Supported: memory://, json://, "
        "sqlite://, postgresql://, mysql://, mongodb://, supabase://, https://<proj>.supabase.co"
    )


# ----------------------------------------------------------------------
# SQL connection helpers
# ----------------------------------------------------------------------

def _sql_connect_factory(url: str, kind: str) -> Callable[[], Any]:
    """Return a zero-arg callable yielding a DB-API-style connection."""

    # Preferred: SQLAlchemy — uniform support for every driver and pooling.
    try:
        import sqlalchemy  # noqa: F401
        from sqlalchemy import create_engine

        norm = url
        if norm.startswith("postgres://"):
            norm = "postgresql://" + norm[len("postgres://"):]
        if kind == "mysql" and not norm.startswith("mysql+"):
            try:
                import MySQLdb  # type: ignore  # noqa: F401
            except ImportError:
                try:
                    import pymysql  # type: ignore  # noqa: F401

                    norm = "mysql+pymysql://" + norm.split("://", 1)[1]
                except ImportError:
                    pass  # let SQLAlchemy report the missing driver

        engine = create_engine(norm, pool_pre_ping=True)

        class _SaConn:
            """DB-API-style adapter over a SQLAlchemy Connection.

            SQLAlchemy 2.0 ``Connection.execute`` is statement-oriented, not
            cursor-oriented (returns the statement result only via
            ``exec_driver_sql``), so expose exactly the interface
            SqlContextStore needs.
            """

            def __init__(self, sa_conn: Any):
                self._c = sa_conn

            def execute(self, sql: str, params: Optional[list] = None) -> Any:
                # exec_driver_sql requires a tuple for positional params
                # (a bare list fails SQLAlchemy's param validation).
                return self._c.exec_driver_sql(sql, tuple(params or []))

            def commit(self) -> None:
                self._c.commit()

            def rollback(self) -> None:
                self._c.rollback()

            def close(self) -> None:
                self._c.close()

        def connect_sa() -> Any:
            return _SaConn(engine.connect())

        # Validate eagerly so misconfigurations surface at construction time.
        connect_sa().execute("SELECT 1")
        return connect_sa
    except ImportError:
        pass

    # Fallback: known drivers directly.
    if kind == "postgres":
        def connect_pg() -> Any:
            try:
                import psycopg2

                return _as_executable(psycopg2.connect(url))
            except ImportError:
                pass
            try:
                import psycopg

                return _as_executable(psycopg.connect(url))
            except ImportError as exc:
                raise ImportError(
                    "No PostgreSQL driver found. Install one of: "
                    "psycopg2-binary, psycopg, or sqlalchemy. "
                    "pip install 'longloom[postgres]'"
                ) from exc

        connect_pg()  # validate eagerly
        return connect_pg

    def connect_mysql() -> Any:
        parsed = urlparse(url)
        conn_kwargs = {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 3306,
            "user": unquote(parsed.username or "root"),
            "password": unquote(parsed.password or ""),
            "database": unquote(parsed.path.lstrip("/")) or None,
        }
        conn_kwargs = {k: v for k, v in conn_kwargs.items() if v is not None}
        try:
            import pymysql

            return _as_executable(pymysql.connect(**conn_kwargs))
        except ImportError:
            pass
        try:
            import MySQLdb

            return _as_executable(MySQLdb.connect(**conn_kwargs))
        except ImportError as exc:
            raise ImportError(
                "No MySQL driver found. Install pymysql or mysqlclient, "
                "or sqlalchemy. pip install 'longloom[mysql]'"
            ) from exc

    connect_mysql()  # validate eagerly
    return connect_mysql


def _as_executable(conn: Any) -> Any:
    """Adapt a raw DB-API connection so ``.execute()`` is available on it.

    sqlite3 and psycopg(v3) connections expose ``execute``; psycopg2, pymysql
    and SQLAlchemy raw connections only expose cursors — wrap those uniformly.
    """
    if hasattr(conn, "execute"):
        return conn

    class _Executable:
        def __init__(self, raw: Any):
            self._raw = raw
            self._cursor = raw.cursor()

        def execute(self, sql: str, params: Optional[list] = None) -> Any:
            self._cursor.execute(sql, params or [])
            return self._cursor  # pymysql returns an int; normalize to cursor

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