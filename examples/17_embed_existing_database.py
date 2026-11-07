#!/usr/bin/env python3
"""17 — Embedding LongLoom inside an EXISTING database (no dedicated DB needed).

Real deployments don't provision a separate database for chunk storage.
This example plays the realistic scenario:

* your app already owns `app.db` with a table `app_document_chunks`
  created by YOUR migration, with YOUR column names,
* other app features already write rows into that table,
* LongLoom adapts to it: custom table + custom columns + a namespace
  discriminator so several apps can share the table safely.

Runs fully offline (SQLite). MongoDB variant at the bottom when reachable.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import DemoLLM, show

from longloom import MapReduceProcessor, SqlContextStore, SQLITE_DIALECT

APP_DB = "embedded_app.db"

# ── 1. "Your" app's own migration creates the table with your names ──────
conn = sqlite3.connect(APP_DB)
conn.execute("""
    CREATE TABLE IF NOT EXISTS app_document_chunks (
        chunk_pk     TEXT PRIMARY KEY,
        text_body    TEXT NOT NULL,
        hash_val     TEXT NOT NULL,
        meta_json    TEXT NOT NULL,
        token_n      INTEGER NOT NULL,
        created      TEXT NOT NULL,
        ord_n        INTEGER NOT NULL,
        total_n      INTEGER NOT NULL,
        sess         TEXT NOT NULL,
        app_name     TEXT NOT NULL DEFAULT 'legacy'
    )""")
conn.execute("INSERT OR IGNORE INTO app_document_chunks VALUES "
             "('legacy-row-1', 'row written by another app feature', "
             "'h', '{}', 0, '2026-11-07', 0, 0, 'legacy', 'legacy')")
conn.commit()
conn.close()

# ── 2. LongLoom adapts to YOUR schema (init_schema=False: migrations own DDL)
store = SqlContextStore(
    connect=lambda: sqlite3.connect(APP_DB),
    dialect=SQLITE_DIALECT,
    table="app_document_chunks",
    init_schema=False,
    columns={
        "id": "chunk_pk", "content": "text_body", "content_hash": "hash_val",
        "metadata": "meta_json", "token_count": "token_n", "created_at": "created",
        "chunk_index": "ord_n", "total_chunks": "total_n", "session_id": "sess",
    },
    namespace="support-agent", namespace_field="app_name",
)

# ── 3. The whole map/merge pipeline runs inside your table ───────────────
DOC = ("Support playbook: escalate sepsis alerts within 5 minutes. "
       "Ticket TKT-2048 tracks the FAQ bot. AlphaFold references belong in "
       "research docs, not support. ") * 20
processor = MapReduceProcessor(
    llm=DemoLLM(), store=store,
    max_chunk_tokens=300, max_context_tokens=8192,
)
result = processor.process(DOC, task="Extract playbook facts",
                           session_id="playbook-2026")

show("LongLoom inside YOUR table (namespace 'support-agent')", result, processor)

# ── 4. Coexistence + isolation ────────────────────────────────────────────
conn = sqlite3.connect(APP_DB)
by_app = dict(conn.execute(
    "SELECT app_name, COUNT(*) FROM app_document_chunks GROUP BY app_name"))
conn.close()
print(f"\nrows per app in the shared table: {by_app}")
assert by_app.get("legacy") == 1, "the other app's row must be untouched"
assert by_app.get("support-agent") == result["num_chunks"]

# ── 5. A second app shares the SAME table via its own namespace ──────────
store_b = SqlContextStore(
    connect=lambda: sqlite3.connect(APP_DB),
    dialect=SQLITE_DIALECT, table="app_document_chunks", init_schema=False,
    columns={
        "id": "chunk_pk", "content": "text_body", "content_hash": "hash_val",
        "metadata": "meta_json", "token_count": "token_n", "created_at": "created",
        "chunk_index": "ord_n", "total_chunks": "total_n", "session_id": "sess",
    },
    namespace="research-agent", namespace_field="app_name",
)
b = store_b.store_context("Research-only doc about AlphaFold folding. " * 10,
                          session_id="playbook-2026")  # same session id!
assert len(store.retrieve_context("playbook-2026")) == result["num_chunks"]
assert len(store_b.retrieve_context("playbook-2026")) == len(b)  # isolated
print(f"namespace isolation verified: support-agent and research-agent "
      f"coexist in one table ({dict(by_app, **{'research-agent': len(b)})})")

store.delete_session("playbook-2026")
store_b.delete_session("playbook-2026")
store_b.close(); store.close()
import os
os.remove(APP_DB)
print("\n✅ embedding verified: your table, your columns, shared safely")
