"""Embedding LongLoom inside EXISTING databases.

Real deployments share tables/collections with application data: custom
table names, custom column/field names, a namespace discriminator, and
user-owned migrations (init_schema=False). Every claim here is verified
against a table the "application" created with its own schema.
"""
from __future__ import annotations

import sqlite3
import uuid

import pytest

from longloom import (
    SQLITE_DIALECT,
    SqlContextStore,
    SQLiteContextStore,
    create_store,
)
from store_contract import run_store_contract


# ---------------------------------------------------------------------------
# 1) custom column names — LongLoom adapts to YOUR schema
# ---------------------------------------------------------------------------

def test_custom_column_names_on_sqlite(tmp_path):
    store = SQLiteContextStore(
        db_path=str(tmp_path / "app.db"),
        table="app_chunks",
        columns={
            "id": "chunk_id",
            "content": "body_text",
            "metadata": "meta_json",
            "token_count": "n_tokens",
            "session_id": "doc_session",
            "chunk_index": "ord",
        },
    )
    run_store_contract(store, "custom-columns")

    # physical schema really uses the mapped names
    conn = sqlite3.connect(str(tmp_path / "app.db"))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(app_chunks)")}
    conn.close()
    assert {"chunk_id", "body_text", "meta_json", "n_tokens", "doc_session"} <= cols
    assert "content" not in cols and "metadata" not in cols


def test_pre_existing_table_with_user_schema(tmp_path):
    """The app's migration created the table; LongLoom must not touch DDL."""
    db = str(tmp_path / "app.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE doc_chunks (
            chunk_id   TEXT PRIMARY KEY,
            blob       TEXT NOT NULL,
            hash       TEXT NOT NULL,
            meta       TEXT NOT NULL,
            tokens     INTEGER NOT NULL,
            stamp      TEXT NOT NULL,
            ord        INTEGER NOT NULL,
            total      INTEGER NOT NULL,
            sess       TEXT NOT NULL
        )""")
    conn.commit()
    conn.close()

    store = SQLiteContextStore(
        db_path=db,
        table="doc_chunks",
        init_schema=False,     # user's migration owns the DDL
        columns={
            "id": "chunk_id", "content": "blob", "content_hash": "hash",
            "metadata": "meta", "token_count": "tokens", "created_at": "stamp",
            "chunk_index": "ord", "total_chunks": "total", "session_id": "sess",
        },
    )
    run_store_contract(store, "pre-existing-table")


def test_ddl_respects_custom_names(tmp_path):
    store = SqlContextStore(
        connect=lambda: sqlite3.connect(":memory:"),
        dialect=SQLITE_DIALECT,
        table="app_chunks",
        columns={"id": "chunk_id", "content": "body"},
    )
    ddl = "\n".join(store.ddl())
    assert "chunk_id TEXT PRIMARY KEY" in ddl and "body TEXT" in ddl
    assert " id " not in ddl.replace("chunk_id", "")


# ---------------------------------------------------------------------------
# 2) namespace — several apps share ONE table without seeing each other
# ---------------------------------------------------------------------------

def test_namespace_isolation_sqlite(tmp_path):
    db = str(tmp_path / "shared.db")
    store_a = SQLiteContextStore(db_path=db, namespace="app-a")
    store_b = SQLiteContextStore(db_path=db, namespace="app-b")

    doc_a = "App A document about AlphaFold. " * 60
    doc_b = "App B document about sepsis triage. " * 60
    a = store_a.store_context(doc_a, session_id="shared-sess")
    b = store_b.store_context(doc_b, session_id="shared-sess")  # SAME session id!

    # same session id, same table — namespaces keep them apart
    assert len(store_a.retrieve_context("shared-sess")) == len(a)
    assert len(store_b.retrieve_context("shared-sess")) == len(b)
    assert "AlphaFold" in a[0].content and "sepsis" in b[0].content
    assert store_a.get_session_info("shared-sess")["total_tokens"] == \
        sum(c.token_count for c in a)
    assert store_b.get_session_info("shared-sess")["total_tokens"] == \
        sum(c.token_count for c in b)

    # deleting in one namespace leaves the other untouched
    store_a.delete_session("shared-sess")
    assert store_a.get_session_info("shared-sess")["chunk_count"] == 0
    assert len(store_b.retrieve_context("shared-sess")) == len(b)


def test_namespace_full_contract(tmp_path):
    store = SQLiteContextStore(
        db_path=str(tmp_path / "ns.db"), namespace="tenant-42")
    run_store_contract(store, "namespace-tenant")


# ---------------------------------------------------------------------------
# 3) create_store passthrough
# ---------------------------------------------------------------------------

def test_create_store_embedding_kwargs(tmp_path):
    store = create_store(
        f"sqlite:///{tmp_path / 'e.db'}",
        table="app_chunks",
        columns={"id": "chunk_id", "content": "body"},
        namespace="ns-create",
    )
    run_store_contract(store, "create_store-embedding")


def test_invalid_names_rejected():
    with pytest.raises(ValueError, match="invalid table"):
        create_store("memory://", table="chunks; DROP TABLE users")
    with pytest.raises(ValueError, match="invalid physical name"):
        SqlContextStore(
            connect=lambda: sqlite3.connect(":memory:"),
            dialect=SQLITE_DIALECT,
            columns={"content": 'body"; DROP TABLE users'},
        )


# ---------------------------------------------------------------------------
# 4) MongoDB: custom collection + field names + namespace (live server)
# ---------------------------------------------------------------------------

def _mongo_available():
    try:
        from pymongo import MongoClient

        client = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=1500)
        client.admin.command("ping")
        return client
    except Exception:
        return None


def test_mongodb_custom_fields_and_namespace():
    from longloom import MongoContextStore

    client = _mongo_available()
    if client is None:
        pytest.skip("MongoDB not reachable")
    db_name = f"longloom_embed_{uuid.uuid4().hex[:8]}"
    store = MongoContextStore(
        client, db_name=db_name,
        collection="app_chunks",
        columns={"content": "body", "session_id": "sess"},
        namespace="app-x",
    )
    try:
        run_store_contract(store, "mongo-custom-fields")
        # physical field names really are mapped (fresh doc, contract deletes its own)
        store.store_context("field mapping check", session_id="fieldmap-sess",
                            max_chunk_tokens=100)
        doc = store.collection.find_one({"sess": "fieldmap-sess"})
        assert doc is not None and "body" in doc and "content" not in doc
        assert doc["namespace"] == "app-x"
    finally:
        client.drop_database(db_name)
        store.close()


# ---------------------------------------------------------------------------
# 5) e2e: full workflow inside an app-owned table
# ---------------------------------------------------------------------------

def test_workflow_embedded_in_existing_database(tmp_path):
    from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc
    from longloom import MapReduceProcessor

    db = str(tmp_path / "app.db")
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE ai_pipeline_chunks (
            chunk_pk TEXT PRIMARY KEY, text_body TEXT NOT NULL,
            hash_val TEXT NOT NULL, meta_json TEXT NOT NULL,
            token_n INTEGER NOT NULL, created TEXT NOT NULL,
            ord_n INTEGER NOT NULL, total_n INTEGER NOT NULL, sess TEXT NOT NULL,
            app_name TEXT NOT NULL DEFAULT 'legacy'
        )""")
    conn.execute("INSERT INTO ai_pipeline_chunks VALUES ('legacy-1', 'app row', "
                 "'h', '{}', 0, 'now', 0, 0, 'legacy-sess', 'legacy')")
    conn.commit()
    conn.close()

    store = SqlContextStore(
        connect=lambda: sqlite3.connect(db),
        dialect=SQLITE_DIALECT,
        table="ai_pipeline_chunks",
        init_schema=False,
        columns={"id": "chunk_pk", "content": "text_body", "content_hash": "hash_val",
                 "metadata": "meta_json", "token_count": "token_n",
                 "created_at": "created", "chunk_index": "ord_n",
                 "total_chunks": "total_n", "session_id": "sess"},
        namespace="longloom", namespace_field="app_name",
    )
    processor = MapReduceProcessor(
        llm=PhaseAwareFakeChatModel(), store=store,
        max_chunk_tokens=400, max_context_tokens=8192,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Summarize", session_id="embedded-run")

    assert result["num_chunks"] >= 2
    # the app's own (non-LongLoom) row is untouched next to our chunks
    conn = sqlite3.connect(db)
    legacy = conn.execute(
        "SELECT chunk_pk FROM ai_pipeline_chunks WHERE app_name = 'legacy'"
    ).fetchall()
    ours = conn.execute(
        "SELECT COUNT(*) FROM ai_pipeline_chunks WHERE app_name = 'longloom'"
    ).fetchone()[0]
    conn.close()
    assert len(legacy) == 1 and ours == result["num_chunks"]

# ---------------------------------------------------------------------------
# 6) chunking invariant: no fact token is ever split across chunks
#    (regression guard for line-aware splitting + whitespace-snapped hard split)
# ---------------------------------------------------------------------------

def test_no_fact_token_ever_split_across_chunks():
    import importlib.util
    from pathlib import Path

    from longloom.chunks import split_text
    from longloom.tokens import EstimateCounter, TiktokenCounter

    spec = importlib.util.spec_from_file_location(
        "udoc", Path(__file__).resolve().parents[1] / "examples" / "_unstructured_doc.py")
    udoc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(udoc)
    doc = udoc.build_unstructured_doc()

    for counter, label in ((TiktokenCounter(), "tiktoken"), (EstimateCounter(), "estimate")):
        for max_tokens in (125, 250, 350, 800):
            chunks = split_text(doc, max_tokens, counter)
            for fact in udoc.FACTS:
                assert any(fact in c for c in chunks), (
                    f"{label}@{max_tokens}: fact {fact!r} split across chunks")
