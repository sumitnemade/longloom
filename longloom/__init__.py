"""longloom — long-context map/merge/repair for any LLM and any storage.

Process documents far larger than a model's context window using a LangGraph
map/merge/repair workflow, with dependency injection for both the LLM and the
storage backend::

    import os
    from longloom import MapReduceProcessor, SQLiteContextStore, init_llm

    processor = MapReduceProcessor(
        llm=init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b",
                     api_key=os.environ["NVIDIA_API_KEY"]),
        store=SQLiteContextStore("ctx.db"),
    )
    result = processor.process(long_document, task="Summarize")
"""

from .chunks import ContextChunk, content_hash, split_text
from .factory import SUPPORTED_SCHEMES, create_store
from .stores.naming import FieldMap
from .graph import MapReduceState, build_graph
from .llm import LLMAdapter, LLMLike, init_llm
from .processor import MapReduceProcessor
from .prompts import DEFAULT_PROMPTS, PromptTemplates, render_prompt
from .stores import (
    MSSQL_DIALECT,
    MYSQL_DIALECT,
    ORACLE_DIALECT,
    POSTGRES_DIALECT,
    SQLITE_DIALECT,
    BaseContextStore,
    InMemoryContextStore,
    JSONFileContextStore,
    MongoContextStore,
    SQLiteContextStore,
    SqlContextStore,
    SupabaseContextStore,
)
from .tokens import EstimateCounter, TiktokenCounter, TokenCounter, default_counter

__version__ = "1.1.0"

__all__ = [
    # processor / graph
    "MapReduceProcessor",
    "MapReduceState",
    "build_graph",
    # llm
    "LLMAdapter",
    "LLMLike",
    "init_llm",
    "NVIDIAChatModel",
    # stores
    "BaseContextStore",
    "InMemoryContextStore",
    "SQLiteContextStore",
    "JSONFileContextStore",
    "SqlContextStore",
    "POSTGRES_DIALECT",
    "MYSQL_DIALECT",
    "SQLITE_DIALECT",
    "MSSQL_DIALECT",
    "ORACLE_DIALECT",
    "MongoContextStore",
    "SupabaseContextStore",
    "create_store",
    "SUPPORTED_SCHEMES",
    "FieldMap",
    # tokens
    "TokenCounter",
    "TiktokenCounter",
    "EstimateCounter",
    "default_counter",
    # chunks
    "ContextChunk",
    "split_text",
    "content_hash",
    # prompts
    "PromptTemplates",
    "DEFAULT_PROMPTS",
    "render_prompt",
]


def __getattr__(name: str):
    # Optional import: NVIDIAChatModel only pulls `requests` when used.
    if name == "NVIDIAChatModel":
        from .nvidia import NVIDIAChatModel

        return NVIDIAChatModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")