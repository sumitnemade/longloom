"""The LangGraph map → merge → repair workflow, with all dependencies injected.

Every node is a closure over an :class:`LLMAdapter` and a
:class:`BaseContextStore` supplied by the caller — the graph itself has zero
knowledge of any vendor or storage engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .chunks import ContextChunk
from .llm import LLMAdapter
from .prompts import DEFAULT_PROMPTS, PromptTemplates, render_prompt
from .stores.base import BaseContextStore
from .tokens import TokenCounter, default_counter

__all__ = ["MapReduceState", "build_graph"]


class MapReduceState(TypedDict, total=False):
    # Input
    input_content: str
    session_id: str
    task: str
    task_focus: str
    # Config
    max_chunk_tokens: int
    max_context_tokens: int
    max_repair_attempts: int
    # Map phase
    chunks: List[ContextChunk]
    map_results: List[str]
    # Merge phase
    merged_result: str
    merge_token_count: int
    # Repair phase
    needs_repair: bool
    repair_attempts: int
    final_result: str
    # Metadata
    total_input_tokens: int
    processing_log: List[str]


def build_graph(
    llm: LLMAdapter,
    store: BaseContextStore,
    token_counter: Optional[TokenCounter] = None,
    prompts: Optional[PromptTemplates] = None,
) -> Any:
    """Compile the map/merge/repair graph with injected dependencies."""
    prompts = prompts or DEFAULT_PROMPTS
    counter: TokenCounter = token_counter or store.counter

    def _combine(parts: List[str], state: MapReduceState, label: str) -> str:
        """One combine call over a group of results (fits the merge budget)."""
        chunk_results_text = "\n\n---\n\n".join(
            f"Part {i + 1}:\n{part}" for i, part in enumerate(parts)
        )
        prompt = render_prompt(
            prompts.merge,
            {
                "task": state["task"],
                "task_focus": state.get("task_focus", "key information and insights"),
                "chunk_results": chunk_results_text,
            },
        )
        return llm.complete(prompt)

    # ------------------------------------------------------------------
    # Nodes (closures over llm/store/prompts/counter)
    # ------------------------------------------------------------------

    def chunk_input(state: MapReduceState) -> Dict[str, Any]:
        chunks = store.store_context(
            content=state["input_content"],
            session_id=state["session_id"],
            metadata={"task": state["task"]},
            max_chunk_tokens=state.get("max_chunk_tokens", 3000),
        )
        total_tokens = sum(c.token_count for c in chunks)
        return {
            "chunks": chunks,
            "total_input_tokens": total_tokens,
            "repair_attempts": 0,
            "processing_log": [
                f"Chunked input into {len(chunks)} chunks ({total_tokens} total tokens)"
            ],
        }

    def map_chunks(state: MapReduceState) -> Dict[str, Any]:
        chunks = state["chunks"]
        total = len(chunks)
        log = list(state.get("processing_log", []))
        results: List[str] = []
        for chunk in chunks:
            prompt = render_prompt(
                prompts.map,
                {
                    "task": state["task"],
                    "task_focus": state.get("task_focus", "key information and insights"),
                    "chunk_index": chunk.chunk_index + 1,
                    "total_chunks": total,
                    "chunk_content": chunk.content,
                },
            )
            results.append(llm.complete(prompt))
            log.append(f"Mapped chunk {chunk.chunk_index + 1}/{total}: {len(results[-1])} chars")
        return {"map_results": results, "processing_log": log}

    def merge_results(state: MapReduceState) -> Dict[str, Any]:
        results = state["map_results"]
        log = list(state.get("processing_log", []))
        max_context = state.get("max_context_tokens", 8192)

        # Merge-input budget: keep every combine call comfortably inside the
        # model window (60% head-room for instructions + output).
        budget = max(1000, int(max_context * 0.6))

        # HIERARCHICAL MERGE: with verbatim/exhaustive map phases the combined
        # map results can exceed any model's input window. Merge in groups
        # that fit the budget, recursively, until one result remains.
        current = list(results)
        passes = 0
        while True:
            total = sum(counter(r) for r in current) + 30 * len(current)
            if total <= budget or len(current) <= 1:
                break
            avg = max(1, total // len(current))
            # group size from CURRENT average result size: k * avg <= budget
            k = max(2, budget // avg)
            if k >= len(current):
                break  # cannot split further; final merge below takes the risk
            nxt: List[str] = []
            for i in range(0, len(current), k):
                group = current[i:i + k]
                nxt.append(_combine(group, state, f"pass{passes + 1}-g{i // k + 1}"))
            current = nxt
            passes += 1
            log.append(f"Hierarchical merge pass {passes}: "
                       f"{len(results)} map results -> {len(current)} partials")

        if len(current) == 1:
            merged = current[0]
        else:
            merged = _combine(current, state, "final")

        token_count = counter(merged)
        log_msg = (f"Merged {len(results)} results into {token_count} tokens"
                   + (f" ({passes} hierarchical pass(es))" if passes else ""))
        log.append(log_msg)
        return {
            "merged_result": merged,
            "merge_token_count": token_count,
            "needs_repair": token_count > max_context,
            "processing_log": log,
        }

    def repair_result(state: MapReduceState) -> Dict[str, Any]:
        current_tokens = state["merge_token_count"]
        max_context = state.get("max_context_tokens", 8192)
        attempts = state.get("repair_attempts", 0)
        prompt = render_prompt(
            prompts.repair,
            {
                "task": state["task"],
                "task_focus": state.get("task_focus", "key information and insights"),
                "current_tokens": current_tokens,
                "max_tokens": max_context,
                "merged_result": state["merged_result"],
            },
        )
        repaired = llm.complete(prompt)
        repaired_tokens = counter(repaired)
        attempts += 1
        log = list(state.get("processing_log", []))
        log.append(f"Repair attempt {attempts}: {current_tokens} -> {repaired_tokens} tokens")

        still_too_long = repaired_tokens > max_context
        can_retry = attempts < state.get("max_repair_attempts", 3)
        update: Dict[str, Any] = {
            "merged_result": repaired,
            "merge_token_count": repaired_tokens,
            "needs_repair": still_too_long and can_retry,
            "repair_attempts": attempts,
            "processing_log": log,
        }
        if not (still_too_long and can_retry):
            update["final_result"] = repaired
        return update

    def finalize(state: MapReduceState) -> Dict[str, Any]:
        final = state.get("final_result") or state.get("merged_result", "")
        log = list(state.get("processing_log", [])) + ["Processing complete"]
        return {"final_result": final, "processing_log": log}

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _route(state: MapReduceState) -> str:
        return "repair_result" if state.get("needs_repair", False) else "finalize"

    workflow = StateGraph(MapReduceState)
    workflow.add_node("chunk_input", chunk_input)
    workflow.add_node("map_chunks", map_chunks)
    workflow.add_node("merge_results", merge_results)
    workflow.add_node("repair_result", repair_result)
    workflow.add_node("finalize", finalize)

    workflow.set_entry_point("chunk_input")
    workflow.add_edge("chunk_input", "map_chunks")
    workflow.add_edge("map_chunks", "merge_results")
    workflow.add_conditional_edges("merge_results", _route)
    workflow.add_conditional_edges("repair_result", _route)
    workflow.add_edge("finalize", END)

    return workflow.compile()