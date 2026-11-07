"""High-level API: MapReduceProcessor.

Usage with any LLM and any storage::

    import os
    from longloom import MapReduceProcessor, SQLiteContextStore, init_llm

    processor = MapReduceProcessor(
        llm=init_llm("openai", "gpt-4o-mini",       # or any BaseChatModel / callable
                     api_key=os.environ["OPENAI_API_KEY"]),
        store=SQLiteContextStore("ctx.db"),          # or InMemory / JSONFile / your own
    )
    result = processor.process(content, task="Summarize", session_id="doc-1")
"""
from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional

from .chunks import ContextChunk
from .graph import MapReduceState, build_graph
from .llm import LLMAdapter, LLMLike
from .prompts import DEFAULT_PROMPTS, PromptTemplates
from .stores.base import BaseContextStore
from .stores.memory import InMemoryContextStore
from .tokens import TokenCounter, default_counter

__all__ = ["MapReduceProcessor"]


class MapReduceProcessor:
    """Long-context map/merge/repair processing with dependency injection."""

    def __init__(
        self,
        llm: LLMLike,
        store: Optional[BaseContextStore] = None,
        token_counter: Optional[TokenCounter] = None,
        prompts: Optional[PromptTemplates] = None,
        max_chunk_tokens: int = 3000,
        max_context_tokens: int = 8192,
        max_repair_attempts: int = 3,
        task_focus: str = "key information and insights",
    ):
        self.llm = llm if isinstance(llm, LLMAdapter) else LLMAdapter(llm)
        self.store = store or InMemoryContextStore()
        self.prompts = prompts or DEFAULT_PROMPTS
        # A map result larger than the merge budget can never be combined —
        # keep chunk inputs at most half the context window.
        self.max_chunk_tokens = min(max_chunk_tokens, max(1, max_context_tokens // 2))
        self.max_context_tokens = max_context_tokens
        self.max_repair_attempts = max_repair_attempts
        self.task_focus = task_focus
        self._counter: TokenCounter = token_counter or self.store.counter
        self.graph = build_graph(
            llm=self.llm,
            store=self.store,
            token_counter=self._counter,
            prompts=self.prompts,
        )

    # ------------------------------------------------------------------

    def process(
        self,
        content: str,
        task: str,
        session_id: Optional[str] = None,
        task_focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run chunk → map → merge → repair over *content*."""
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

        initial: Dict[str, Any] = {
            "input_content": content,
            "session_id": session_id,
            "task": task,
            "task_focus": task_focus or self.task_focus,
            "max_chunk_tokens": self.max_chunk_tokens,
            "max_context_tokens": self.max_context_tokens,
            "max_repair_attempts": self.max_repair_attempts,
            "processing_log": [],
        }
        result: MapReduceState = self.graph.invoke(initial)

        return {
            "session_id": session_id,
            "task": task,
            "final_result": result.get("final_result", ""),
            "total_input_tokens": result.get("total_input_tokens", 0),
            "num_chunks": len(result.get("chunks", [])),
            "merge_token_count": result.get("merge_token_count", 0),
            "repair_attempts": result.get("repair_attempts", 0),
            "processing_log": result.get("processing_log", []),
        }

    # -- store passthroughs --------------------------------------------

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        return self.store.get_session_info(session_id)

    def retrieve_chunks(
        self,
        session_id: str,
        max_tokens: Optional[int] = None,
        chunk_indices: Optional[List[int]] = None,
    ) -> List[ContextChunk]:
        return self.store.retrieve_context(session_id, max_tokens=max_tokens, chunk_indices=chunk_indices)

    def search_session(self, session_id: str, query: str, limit: int = 10) -> List[ContextChunk]:
        return self.store.search_context(session_id, query, limit=limit)

    def delete_session(self, session_id: str) -> None:
        self.store.delete_session(session_id)
