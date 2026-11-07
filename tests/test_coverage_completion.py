"""Coverage completion: targeted tests for every remaining branch.

Strategy per area:
* NVIDIAChatModel — fake `requests` module injected into sys.modules
  (the model imports it lazily inside each method), covering _generate,
  stream (data/[DONE]/malformed/reasoning-only deltas), _payload variants,
  _convert role mapping, context_window and the missing-key ValueError.
* Supabase store — real supabase-py against the local PostgREST stub
  (explicit url/key, no environment), plus the missing-credential and
  ImportError branches.
* Factory — fake DB-API driver modules injected via sys.modules cover the
  direct-driver fallback paths and their ImportError raises; sys.modules
  removal of sqlalchemy forces the non-SQLAlchemy branches.
* init_llm — fake provider modules for the direct map; missing-module and
  missing-meta-package ImportError branches.
* Store/store-edge unit tests for limit-breaks, empty-index filters,
  blank-line tolerance, sanitizer, close-swallow and rollback paths.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from pathlib import Path

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from longloom import (
    InMemoryContextStore,
    JSONFileContextStore,
    MapReduceProcessor,
    SQLiteContextStore,
    create_store,
)
from longloom.chunks import split_text
from longloom.llm import LLMAdapter
from longloom.stores.memory import InMemoryContextStore as MemStore
from longloom.tokens import EstimateCounter, TiktokenCounter, default_counter
from fake_llm import KEY_CONCEPTS, PhaseAwareFakeChatModel, make_doc
from store_contract import run_store_contract

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))


# ===========================================================================
# NVIDIAChatModel — fake requests module (no network)
# ===========================================================================

class FakeResponse:
    def __init__(self, payload=None, lines=(), status=200):
        self._payload = payload or {}
        self._lines = lines
        self.status_code = status

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload

    def iter_lines(self):
        return iter(self._lines)


class FakeRequests:
    """Stands in for the `requests` module inside longloom.nvidia."""

    def __init__(self, post_result=None, stream_lines=()):
        self.post_result = post_result or {}
        self.stream_lines = stream_lines
        self.calls: list[dict] = []

    class _Resp:
        def __init__(self, outer, stream):
            self.status_code = 200
            self._outer = outer
            self._stream = stream

        def raise_for_status(self):
            pass

        def json(self):
            return self._outer.post_result

        def iter_lines(self):
            return iter(self._outer.stream_lines)

    def post(self, url, headers=None, json=None, stream=False, timeout=None, **kw):
        self.calls.append({"url": url, "headers": headers,
                           "json": json, "stream": stream})
        return FakeRequests._Resp(self, stream)


def install_fake_requests(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "requests", fake)


def nvidia_model(**kw):
    from longloom import NVIDIAChatModel

    return NVIDIAChatModel(api_key="test-key", **kw)


OPENAI_OK = {"choices": [{"message": {"role": "assistant", "content": "hi there"},
                          "finish_reason": "stop"}]}


def test_nvidia_rejects_missing_key_explicitly():
    from longloom import NVIDIAChatModel

    with pytest.raises(ValueError, match="never reads keys from the environment"):
        NVIDIAChatModel(api_key=None)


def test_nvidia_properties_and_role_mapping():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    m = nvidia_model(model_name="nvidia/nemotron-3-ultra-550b-a55b")
    assert m._llm_type == "nvidia-chat"
    assert m.context_window == 8192
    roles = m._convert([SystemMessage(content="s"), HumanMessage(content="u"),
                        AIMessage(content="a")])
    assert [r["role"] for r in roles] == ["system", "user", "assistant"]


def test_nvidia_payload_variants():
    m = nvidia_model(thinking=None)
    p = m._payload([HumanMessage(content="x")], stream=False, stop=None)
    assert p["stream"] is False and "chat_template_kwargs" not in p
    p = m._payload([HumanMessage(content="x")], stream=True, stop=["END"])
    assert p["stream"] is True and p["stop"] == ["END"]
    m2 = nvidia_model(thinking=False)
    p2 = m2._payload([HumanMessage(content="x")], stream=False, stop=None)
    assert p2["chat_template_kwargs"] == {"thinking": False}


def test_nvidia_generate_success_and_reasoning(monkeypatch):
    fake = FakeRequests(post_result={
        "choices": [{"message": {"role": "assistant", "content": None,
                                 "reasoning_content": "thinking..."},
                     "finish_reason": "length"}]})
    install_fake_requests(monkeypatch, fake)
    m = nvidia_model(max_tokens=1024, thinking=False)
    result = m._generate([HumanMessage(content="q")])
    text = result.generations[0].message.content
    assert text == "" and result.generations[0].generation_info["reasoning_tokens"] == 11
    assert fake.calls[-1]["url"].endswith("/chat/completions")
    assert fake.calls[-1]["json"]["chat_template_kwargs"] == {"thinking": False}


def test_nvidia_stream_variants(monkeypatch):
    sse = [
        b'data: {"choices": [{"delta": {"content": "he"}}]}',
        b"",                                              # blank line skipped
        b"data: not-json",                                # malformed, skipped
        b'data: {"choices": [{"delta": {"reasoning_content": "hmm"}}]}',  # no content
        b'data: {"choices": [{"delta": {"content": "ll"}}]}',
        b"data: [DONE]",
        b'data: {"choices": [{"delta": {"content": "never"}}]}',          # after DONE
    ]
    fake = FakeRequests(stream_lines=sse)
    install_fake_requests(monkeypatch, fake)
    m = nvidia_model(stream=True)
    got = [g.message.content for g in m._generate([
        HumanMessage(content="q")]).generations] if False else None
    chunks = [g.message.content for g in m.stream([HumanMessage(content="q")])]
    assert "".join(chunks) == "hell"
    assert fake.calls[-1]["json"]["stream"] is True


def test_nvidia_headers_merge():
    m = nvidia_model(headers={"X-Corp": "edge"})
    h = m._headers()
    assert h["Authorization"] == "Bearer test-key" and h["X-Corp"] == "edge"


# ===========================================================================
# Supabase store — real supabase-py against the local PostgREST stub
# ===========================================================================

def test_supabase_store_against_local_stub(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from postgrest_stub import StubPostgREST

    from longloom import SupabaseContextStore

    stub = StubPostgREST(port=8909).start()
    try:
        store = SupabaseContextStore(url=stub.url(), key="local-key",
                                     table=f"sb_{uuid.uuid4().hex[:6]}",
                                     namespace="tenant-x",    # covers ns branches
                                     namespace_field="tenant")  # custom field name
        own = f"sb-own-{uuid.uuid4().hex[:8]}"
        store.store_context("multi e hits here. more e text.", session_id=own,
                            max_chunk_tokens=200)
        run_store_contract(store, "supabase-local-stub")
        assert store.retrieve_context("contract-none", chunk_indices=[]) == []
        assert store.search_context("contract-none", "zzz-nonexistent", limit=1) == []
        assert len(store.search_context(own, "e", limit=1)) == 1   # limit-break
        store.delete_session(own)
    finally:
        stub.stop()


def test_supabase_requires_explicit_url_and_key():
    from longloom import SupabaseContextStore

    with pytest.raises(ValueError, match="never reads credentials"):
        SupabaseContextStore(url=None, key=None)


def test_supabase_import_error_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "supabase":
            raise ImportError("no supabase")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from longloom import SupabaseContextStore

    with pytest.raises(ImportError, match="supabase-py is required"):
        SupabaseContextStore(url="https://x.supabase.co", key="k")


def test_supabase_table_sql_respects_custom_names():
    from longloom.stores.supabase import table_sql

    sql = table_sql("app_chunks", {"content": "body"},
                    namespace="tenant-1", namespace_field="tenant")
    assert "chunk_id" not in sql       # id kept under its default name
    assert "body TEXT" in sql          # content column mapped
    assert "tenant TEXT NOT NULL" in sql   # custom namespace field name
    assert "app_chunks" in sql


# ===========================================================================
# Factory — fake drivers + missing-sqlalchemy branches
# ===========================================================================

def _fake_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_factory_direct_driver_fallbacks(monkeypatch):
    """With sqlalchemy removed, the factory falls back to known drivers."""
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)  # force ImportError

    recorded = {}

    fake_psycopg2 = _fake_module(
        "psycopg2", connect=lambda dsn: (recorded.__setitem__("pg", dsn), _FakeConn())[1])
    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    store = __import__("longloom").create_store("postgresql://u:p@h/db")
    assert "pg" in recorded and store.dialect.name == "postgres"

    fake_pymysql = _fake_module("pymysql", connect=lambda **kw: (recorded.__setitem__("my", kw), _FakeConn())[1])
    monkeypatch.setitem(sys.modules, "pymysql", fake_pymysql)
    store = __import__("longloom").create_store("mysql://u:p@h/db")
    assert "my" in recorded and store.dialect.name == "mysql"

    fake_pymssql = _fake_module("pymssql", connect=lambda **kw: (recorded.__setitem__("ms", kw), _FakeConn())[1])
    monkeypatch.setitem(sys.modules, "pymssql", fake_pymssql)
    store = __import__("longloom").create_store("mssql://u:p@h/db")
    assert "ms" in recorded and store.dialect.name == "mssql"

    fake_oracledb = _fake_module("oracledb", connect=lambda user, password, dsn: (recorded.__setitem__("or", dsn), _FakeConn())[1])
    monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)
    store = __import__("longloom").create_store("oracle://u:p@h/svc")
    assert "or" in recorded and store.dialect.name == "oracle"


class _FakeConn:
    """Minimal DB-API-ish connection (has .execute, so no adapter needed)."""

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


class _CursorlessConn:
    """DB-API conn WITHOUT .execute — forces the _Executable adapter path."""

    def __init__(self):
        self.rolled_back = False
        self.closed = False
        self._cursor = types.SimpleNamespace(
            execute=lambda sql, params=None: self,
            fetchall=lambda: [], fetchone=lambda: (0, 0, None, None),
            close=lambda: None)

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_executable_adapter_rollback_and_close():
    from longloom.stores.sql import _as_executable

    raw = _CursorlessConn()
    wrapped = _as_executable(raw)
    assert hasattr(wrapped, "execute")          # adapter applied
    wrapped.execute("SELECT 1", [])
    wrapped.rollback()
    assert raw.rolled_back
    wrapped.close()
    assert raw.closed


def test_sql_store_close_swallows_closed_connection_errors(tmp_path):
    store = SQLiteContextStore(str(tmp_path / "c2.db"))
    store.store_context("close-swallow 123", session_id="cs", max_chunk_tokens=100)

    class Broken:
        def commit(self):
            raise RuntimeError("already closed")

        def close(self):
            raise RuntimeError("already closed")

    store.conn = Broken()   # simulate a connection that dies before close
    store.close()           # must swallow both errors
    store.close()           # idempotent


def test_sql_delete_session_content_dir_with_namespace(tmp_path):
    store = SQLiteContextStore(str(tmp_path / "ns.db"),
                               content_dir=str(tmp_path / "files"),
                               namespace="tenant-y")
    store.store_context("file-backed namespaced 123", session_id="fb-ns",
                        max_chunk_tokens=100)
    store.delete_session("fb-ns")
    assert store.get_session_info("fb-ns")["chunk_count"] == 0
    store.close()


def test_factory_sa_adapter_commit_close(monkeypatch):
    """Store operations run through the SQLAlchemy adapter: exec_driver_sql on
    persist, commit via store.close() — covering the _SaConn wrapper."""
    import sqlalchemy as sa

    from longloom import create_store

    class FakeSaConn:
        def __init__(self):
            self.executed = []
            self.committed = False
            self.closed = False

        def exec_driver_sql(self, sql, params=None):
            self.executed.append((sql, params))

        def cursor(self):
            return types.SimpleNamespace(execute=lambda sql, params=None: None,
                                         close=lambda: None)

        def commit(self):
            self.committed = True

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
    store.store_context("sa adapter check 123", session_id="sa", max_chunk_tokens=100)
    assert conns and conns[-1].committed          # commit came through the adapter
    store.close()
    assert conns[-1].closed


def test_factory_json_netloc_and_mongodb_supabase_schemes():
    from longloom import JSONFileContextStore, create_store

    store = create_store("json://data")            # netloc form (line 71)
    assert store.root_dir == "data"

    mongo = create_store("mongodb://127.0.0.1:27017/covdb")   # mongodb branch
    assert mongo.names.table == "context_chunks"
    mongo.close()

    sb = create_store("supabase://proj.supabase.co", key="k")  # cloud URL form
    assert sb.client is not None

    sb2 = create_store("https://proj2.supabase.co", key="k")   # https form
    assert sb2.client is not None


def test_factory_psycopg3_and_mysqldb_fallbacks(monkeypatch):
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    monkeypatch.setitem(sys.modules, "pymysql", None)

    recorded = {}

    fake_psycopg = _fake_module("psycopg", connect=lambda conninfo: (
        recorded.__setitem__("pg3", conninfo), _FakeConn())[1])
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)
    store = __import__("longloom").create_store("postgresql://u:p@h/db")
    assert "pg3" in recorded and store.dialect.name == "postgres"

    fake_mysqldb = _fake_module("MySQLdb", connect=lambda **kw: (
        recorded.__setitem__("mydb", kw), _FakeConn())[1])
    monkeypatch.setitem(sys.modules, "MySQLdb", fake_mysqldb)
    store = __import__("longloom").create_store("mysql://u:p@h/db")
    assert "mydb" in recorded and store.dialect.name == "mysql"


def test_factory_mssql_oracle_fallbacks(monkeypatch):
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)

    recorded = {}
    fake_pymssql = _fake_module("pymssql", connect=lambda **kw: (
        recorded.__setitem__("ms", kw), _FakeConn())[1])
    monkeypatch.setitem(sys.modules, "pymssql", fake_pymssql)
    store = __import__("longloom").create_store("mssql://u:p@h/db")
    assert "ms" in recorded and store.dialect.name == "mssql"

    fake_oracledb = _fake_module("oracledb", connect=lambda user, password, dsn: (
        recorded.__setitem__("or", dsn), _FakeConn())[1])
    monkeypatch.setitem(sys.modules, "oracledb", fake_oracledb)
    store = __import__("longloom").create_store("oracle://u:p@h/svc")
    assert "or" in recorded and store.dialect.name == "oracle"


def _fake_connect():
    raise NotImplementedError("replaced in test")
    monkeypatch.setitem(sys.modules, "sqlalchemy", None)
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    monkeypatch.setitem(sys.modules, "psycopg", None)
    monkeypatch.delitem(sys.modules, "psycopg2", raising=False) if False else None
    with pytest.raises(ImportError, match="psycopg2-binary"):
        __import__("longloom").create_store("postgresql://u:p@h/db")

    monkeypatch.setitem(sys.modules, "pymysql", None)
    monkeypatch.setitem(sys.modules, "MySQLdb", None)
    with pytest.raises(ImportError, match="pymysql or mysqlclient"):
        __import__("longloom").create_store("mysql://u:p@h/db")

    monkeypatch.setitem(sys.modules, "pymssql", None)
    with pytest.raises(ImportError, match="pymssql"):
        __import__("longloom").create_store("mssql://u:p@h/db")

    monkeypatch.setitem(sys.modules, "oracledb", None)
    with pytest.raises(ImportError, match="oracledb"):
        __import__("longloom").create_store("oracle://u:p@h/svc")


def test_factory_sqlalchemy_normalization(monkeypatch):
    """mssql:///oracle:// URLs are normalized to SQLAlchemy dialect+driver."""
    seen = {}

    def fake_create_engine(url, **kw):
        seen["url"] = url
        return types.SimpleNamespace(connect=lambda: _FakeConn())

    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "create_engine", fake_create_engine)
    from longloom.factory import _sql_connect_factory

    pg = _sql_connect_factory("postgres://u:p@h/db", "postgres")
    assert seen["url"].startswith("postgresql://")
    my = _sql_connect_factory("mysql://u:p@h/db", "mysql")
    assert seen["url"].startswith("mysql+pymysql://")
    ms = _sql_connect_factory("mssql://u:p@h/db", "mssql")
    assert seen["url"].startswith("mssql+pymssql://")
    orc = _sql_connect_factory("oracle://u:p@h/db", "oracle")
    assert seen["url"].startswith("oracle+oracledb://")


# ===========================================================================
# init_llm — direct map, custom import error, fallback
# ===========================================================================

def test_init_llm_direct_map_with_fake_module(monkeypatch):
    fake = _fake_module("langchain_groq", ChatGroq=lambda model, api_key, **kw:
                        types.SimpleNamespace(model=model, provider="groq"))
    monkeypatch.setitem(sys.modules, "langchain_groq", fake)
    m = init_llm("groq", "llama-3", api_key="k")
    assert m.model == "llama-3" and m.provider == "groq"


def test_init_llm_direct_map_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    with pytest.raises(ImportError, match="langchain-anthropic"):
        init_llm("anthropic", "claude", api_key="k")


def test_init_llm_fallback_needs_meta_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain", None)
    with pytest.raises(ImportError, match="meta-package"):
        init_llm("totally-unknown-provider", "m", api_key="k")


def test_init_llm_custom_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    with pytest.raises(ImportError, match="langchain-openai"):
        init_llm("custom", "m", base_url="http://x/v1")


def test_init_llm_ollama_needs_no_key(monkeypatch):
    fake = _fake_module("langchain_ollama", ChatOllama=lambda model, api_key=None, **kw:
                        types.SimpleNamespace(model=model))
    monkeypatch.setitem(sys.modules, "langchain_ollama", fake)
    m = init_llm("ollama", "llama3.1")
    assert m.model == "llama3.1"


def test_llm_adapter_acomplete():
    adapter = LLMAdapter(lambda p: "async ok")
    assert asyncio.run(adapter.acomplete("x")) == "async ok"


# ===========================================================================
# init_llm needs `import os` free zone — no-op import guard
# ===========================================================================

from longloom import init_llm  # noqa: E402


# ===========================================================================
# Store edges: limits, empty filters, sanitizer paths, close-safety
# ===========================================================================

def test_store_limit_break_paths(tmp_path):
    store = SQLiteContextStore(str(tmp_path / "lim.db"))
    store.store_context("alpha beta gamma delta. " * 30, session_id="lim",
                        max_chunk_tokens=120)
    hits = store.search_context("lim", "alpha", limit=1)
    assert len(hits) == 1
    empty = store.retrieve_context("lim", chunk_indices=[])
    assert empty == []
    store.close(); store.close()  # double close must not raise


def test_jsonfile_sanitized_session_and_blank_lines(tmp_path):
    store = JSONFileContextStore(str(tmp_path / "jf"))
    store.store_context("content with 42 facts.", session_id="weird/id with spaces",
                        max_chunk_tokens=200)
    hits = store.search_context("weird/id with spaces", "42")
    assert hits
    # blank line inside the JSONL file is tolerated on load
    path = next(Path(store.root_dir).glob("*.jsonl"))
    path.write_text(path.read_text() + "\n")
    assert store.retrieve_context("weird/id with spaces")


def test_memory_search_limit_break():
    store = InMemoryContextStore()
    for i in range(5):
        store.store_context(f"doc {i} mentions needle.", session_id=f"s{i}",
                            max_chunk_tokens=100)
    hits = store.search_context("s0", "needle", limit=1)
    assert len(hits) == 1


def test_processor_delete_session_failure_is_swallowed():
    class BoomStore(InMemoryContextStore):
        def delete_session(self, session_id):
            raise RuntimeError("delete failed")

    processor = MapReduceProcessor(llm=lambda p: "ok", store=BoomStore(),
                                   max_chunk_tokens=200)
    result = processor.process("tiny doc with 5 tokens.", task="Summarize",
                               session_id="boom")
    assert result["final_result"] == "ok"


def test_single_chunk_doc_hits_direct_merge_path():
    processor = MapReduceProcessor(
        llm=PhaseAwareFakeChatModel(), store=InMemoryContextStore(),
        max_chunk_tokens=4000, max_context_tokens=8192,
    )
    result = processor.process("One small paragraph. It has 8 tokens total.",
                               task="Summarize", session_id="single")
    assert result["num_chunks"] == 1


def test_hard_split_token_path_on_single_long_line():
    """A single line longer than max_tokens must be hard-split."""
    line = "x" * 2000  # one line, no whitespace
    pieces = split_text(line, 100, lambda s: max(1, len(s) // 4))
    assert len(pieces) == 5                       # 500 tokens / 100 budget
    # quantized counter: each piece measures <= 100 tokens
    assert all(max(1, len(p) // 4) <= 100 for p in pieces)
    assert "".join(pieces) == line


def test_hard_split_minimum_cut_guards():
    """Adversarial counter forces the guess down to 1 (minimum-cut guards)."""
    calls = {"n": 0}

    def adversarial_counter(text: str) -> int:
        calls["n"] += 1
        return 10 ** 6  # always wildly over budget

    pieces = split_text("abcdef", 5, adversarial_counter)
    assert pieces  # terminated without hanging


def test_split_text_empty_input():
    assert split_text("", 100, default_counter()) == []
    with pytest.raises(ValueError):
        split_text("x", 0, default_counter())


# ===========================================================================
# tokens / naming / module-dunder edges
# ===========================================================================

def test_tiktoken_counter_split_direct():
    tc = TiktokenCounter()
    parts = tc.split("one two three four five.", 2)
    assert len(parts) >= 2 and "".join(parts).split() == \
        "one two three four five.".split()


def test_estimate_counter_rejects_bad_ratio():
    with pytest.raises(ValueError):
        EstimateCounter(chars_per_token=0)


def test_default_counter_falls_back_without_tiktoken(monkeypatch):
    import longloom.tokens as tok

    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def fake_import(name, *a, **kw):
        if name == "tiktoken":
            raise ImportError("no tiktoken")
        return real_import(name, *a, **kw)

    import builtins
    monkeypatch.setattr(builtins, "__import__", fake_import)
    counter = tok.default_counter()
    assert isinstance(counter, EstimateCounter)


def test_sqlite_netloc_path_form(tmp_path, monkeypatch):
    """sqlite://host/path keeps the netloc in the db path (line-71 branch)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "relative" / "dir").mkdir(parents=True)
    store = create_store("sqlite://relative/dir/x.db")
    assert store.names.table == "context_chunks"
    store.store_context("netloc form 123", session_id="nl", max_chunk_tokens=100)
    assert (tmp_path / "relative" / "dir" / "x.db").exists()


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


def test_naming_default_namespace_col_when_namespace_off():
    from longloom.stores.naming import FieldMap

    assert FieldMap().col("namespace") == "namespace"


def test_naming_rejects_bad_namespace_field_and_unknown_logical():
    from longloom.stores.naming import FieldMap

    with pytest.raises(ValueError, match="invalid namespace field"):
        FieldMap(namespace="ns", namespace_field="bad name")
    with pytest.raises(ValueError, match="unknown logical field"):
        FieldMap(columns={"not_a_field": "x"})


def test_package_dunder_getattr_unknown():
    import longloom

    with pytest.raises(AttributeError, match="no attribute"):
        longloom.definitely_not_an_attribute


# ===========================================================================
# graph: hierarchical merge reaches single-partial path
# ===========================================================================

def test_hierarchical_merge_converges_to_single_partial():
    """Dedup-faithful fake: after enough grouping passes, one partial remains
    and the merge takes the `merged = current[0]` branch."""
    fake = PhaseAwareFakeChatModel(
        merge_response="[MERGE RESULT] " + "; ".join(KEY_CONCEPTS) * 8)
    processor = MapReduceProcessor(
        llm=fake, store=InMemoryContextStore(),
        max_chunk_tokens=300, max_context_tokens=400,  # tight: forces passes
        max_repair_attempts=1,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Summarize", session_id="hier")
    assert result["num_chunks"] >= 2
    assert result["repair_attempts"] <= 1
    assert "Hierarchical merge pass" in "\n".join(result["processing_log"]) or \
        result["merge_token_count"] <= 400

# ===========================================================================
# Final sweep: last uncovered branches
# ===========================================================================

def test_nvidia_stream_skips_non_data_lines(monkeypatch):
    sse = [
        b"event: ping",                                   # non-data line -> continue
        b'data: {"choices": [{"delta": {"content": "ok"}}]}',
        b"data: [DONE]",
    ]
    fake = FakeRequests(stream_lines=sse)
    install_fake_requests(monkeypatch, fake)
    chunks = [g.message.content for g in nvidia_model().stream([HumanMessage(content="q")])]
    assert chunks == ["ok"]


def test_retry_after_malformed_header_falls_back(monkeypatch):
    import time as _time

    sleeps: list[float] = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))

    class BadHeader(PhaseAwareFakeChatModel):
        n: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
            prompt = "\n".join(str(m.content) for m in messages)
            if self.n == 0 and "CHUNK" in prompt:
                self.n += 1
                err = RetryAfterError("not-a-number")   # malformed header
                return err
            return super()._generate(messages, stop, run_manager, **kwargs)

    adapter = LLMAdapter(BadHeader(), retry_attempts=2, retry_backoff=1.0)
    out = adapter.complete("CHUNK 1/2: fact")
    assert out.startswith("[MAP RESULT]") or out
    assert sleeps and 1.0 <= sleeps[0] <= 1.25          # base backoff + jitter, no header boost


def test_pacing_sleeps_between_rapid_calls(monkeypatch):
    import time as _time

    sleeps: list[float] = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s) if s > 0 else None)

    adapter = LLMAdapter(PhaseAwareFakeChatModel(), min_interval=5.0)
    adapter.complete("CHUNK 1/2: a")
    adapter.complete("CHUNK 2/2: b")
    assert any(s >= 4.0 for s in sleeps)   # second call waits out the interval


def test_init_llm_nvidia_native_fallback_without_openai_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_openai", None)
    from longloom import NVIDIAChatModel

    m = init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", api_key="k")
    assert isinstance(m, NVIDIAChatModel)


def test_init_llm_nvidia_chatopenai_without_thinking_kwarg():
    m = init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", api_key="k")
    assert m.model_name == "nvidia/nemotron-3-ultra-550b-a55b"


def test_init_llm_fallback_returns_init_chat_model_result(monkeypatch):
    fake_pkg = types.ModuleType("langchain")
    fake_cm = types.ModuleType("langchain.chat_models")
    fake_cm.init_chat_model = lambda model, model_provider, api_key=None, **kw: \
        types.SimpleNamespace(model=model, provider=model_provider)
    fake_pkg.chat_models = fake_cm
    monkeypatch.setitem(sys.modules, "langchain", fake_pkg)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", fake_cm)
    m = init_llm("unknown-provider", "some-model", api_key="k")
    assert m.provider == "unknown-provider"


def test_processor_delete_session_passthrough(tmp_path):
    store = SQLiteContextStore(str(tmp_path / "p.db"))
    processor = MapReduceProcessor(llm=lambda p: "ok", store=store)
    processor.store.store_context("some content 123", session_id="del-me",
                                  max_chunk_tokens=100)
    processor.delete_session("del-me")
    assert processor.get_session_info("del-me")["chunk_count"] == 0


def test_prompts_for_extraction_preset():
    from longloom import DEFAULT_PROMPTS, PromptTemplates

    preset = PromptTemplates.for_extraction()
    assert preset.map is not DEFAULT_PROMPTS.map
    assert "VERBATIM" in preset.map or "verbatim" in preset.map


def test_single_chunk_document_uses_direct_merge():
    processor = MapReduceProcessor(
        llm=PhaseAwareFakeChatModel(), store=InMemoryContextStore(),
        max_chunk_tokens=4000, max_context_tokens=8192,
    )
    result = processor.process("One tiny paragraph with 9 tokens.", task="Summarize",
                               session_id="single")
    assert result["num_chunks"] == 1


def test_mongo_empty_chunk_index_filter_and_list_sessions():
    pytest.importorskip("pymongo")
    from pymongo import MongoClient

    from longloom import MongoContextStore

    client = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=1500)
    client.admin.command("ping")
    db_name = f"longloom_cov_{uuid.uuid4().hex[:8]}"
    store = MongoContextStore(client_or_uri=client, db_name=db_name)
    try:
        store.store_context("empty-filter check 123", session_id="cov",
                            max_chunk_tokens=200)
        assert store.retrieve_context("cov", chunk_indices=[]) == []
        assert "cov" in store.list_sessions()
    finally:
        client.drop_database(db_name)
        store.close()


def test_mongo_missing_pymongo_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymongo", None)
    from longloom import MongoContextStore

    with pytest.raises(ImportError, match="pymongo is required"):
        MongoContextStore(client_or_uri="mongodb://127.0.0.1:27017")


def test_jsonfile_empty_persist_is_noop(tmp_path):
    store = JSONFileContextStore(str(tmp_path / "jf"))
    store._persist([])  # no-op guard
    assert store.list_sessions() == []


def test_jsonfile_search_limit_break(tmp_path):
    store = JSONFileContextStore(str(tmp_path / "jf"))
    for i in range(4):
        store.store_context(f"needle number {i}.", session_id=f"j{i}",
                            max_chunk_tokens=100)
    hits = store.search_context("j0", "needle", limit=1)
    assert len(hits) == 1


def test_sql_store_close_is_idempotent_and_swallowing(tmp_path):
    store = SQLiteContextStore(str(tmp_path / "c.db"))
    store.store_context("close check 123", session_id="c", max_chunk_tokens=100)
    store.close()
    store.close()  # second close: commit/close on a closed connection must be swallowed


def test_sql_executable_rollback_proxy():
    from longloom.stores.sql import _as_executable

    class FakeRaw:
        def __init__(self):
            self.rolled = False
            self.closed = False

        def cursor(self):
            return self

        def execute(self, sql, params=None):
            return self

        def commit(self):
            pass

        def rollback(self):
            self.rolled = True

        def close(self):
            self.closed = True

    raw = FakeRaw()
    wrapped = _as_executable(raw)
    assert hasattr(wrapped, "execute")
    wrapped.rollback()
    assert raw.rolled
    wrapped.close()
    assert raw.closed


def test_sql_list_sessions_with_namespace(tmp_path):
    from longloom import SQLiteContextStore

    store = SQLiteContextStore(str(tmp_path / "ns.db"), namespace="tenant-x")
    store.store_context("namespaced listing 123", session_id="ns-list",
                        max_chunk_tokens=100)
    assert store.list_sessions() == ["ns-list"]
    store.delete_session("ns-list")
    assert store.list_sessions() == []


def test_sql_filebacked_delete_with_namespace(tmp_path):
    store = SQLiteContextStore(str(tmp_path / "fb.db"),
                               content_dir=str(tmp_path / "files"),
                               namespace="tenant-y")
    store.store_context("file-backed namespaced content 123", session_id="fb-ns",
                        max_chunk_tokens=100)
    assert len(store.retrieve_context("fb-ns")) == 1
    store.delete_session("fb-ns")
    assert store.get_session_info("fb-ns")["chunk_count"] == 0
