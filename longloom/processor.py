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

Parallelism, resume, observability and hygiene::

    processor = MapReduceProcessor(
        llm=llm, store=store,
        max_workers=8,                        # parallel map phase
        on_event=lambda e: print(e),          # structured progress events
        scrubber=my_pii_scrubber,             # rewrite content before anything is stored
    )
    result = processor.process(doc, task=..., resume=True)   # resume a crashed run
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import re
import uuid
from typing import Any, Callable, Dict, List, Optional

from .chunks import ContextChunk
from .graph import MapReduceState, build_graph
from .llm import LLMAdapter, LLMLike, RateLimitError
from .prompts import (DEFAULT_PROMPTS, VERBATIM_MAP_PROMPT, PromptTemplates,
                      render_prompt)

logger = logging.getLogger("longloom.processor")
from .stores.base import BaseContextStore
from .stores.memory import InMemoryContextStore
from .tokens import TokenCounter

__all__ = ["MapReduceProcessor", "RateLimitError"]

# A source line is treated as fact-bearing when it carries a digit or a
# currency/percent symbol. This mirrors what the verbatim map prompt asks
# the model to copy ("numbers, ... money amounts, ... percentages"), so the
# source-anchored audit can restore exactly what the prompt demands.
_SOURCE_FACT_RE = re.compile(r"[\d\u20b9\u20ac\u00a3%]")

# Numeric figures for the summarization audit: comma-grouped numbers,
# decimals, or runs of 4+ digits (amounts, totals, years). Short numbers
# (page numbers, small counts) are too noisy to be useful signals.
_FIGURE_RE = re.compile(r"\d+\.\d+|\d[\d,]{2,}|\d{4,}")

# Prefixes stripped before two lines are compared for identity. Real merge
# models do not always preserve the bullet: table rows may come back raw,
# numbered items may lose their index — the comparison must be prefix-blind
# or those lines are invisible to coverage counting (measured in the
# unstructured-BRD validation: 15 lines present in the merge output were
# missed by bullet-only matching).
_ITEM_PREFIXES = ("- ", "* ", "\u2022 ", "> ", "| ")


def _now_iso() -> str:
    """Event timestamp, same format as graph.emit."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _normalize_item(line: str) -> str:
    """Canonical form of an extracted item: strip list/table/quote prefixes
    and collapse whitespace, so '- FACT-1', '| FACT-1 |', 'FACT-1' match."""
    ln = line.strip()
    changed = True
    while changed and ln:
        changed = False
        for p in _ITEM_PREFIXES:
            if ln.startswith(p):
                ln = ln[len(p):].strip()
                changed = True
        m = re.match(r"^\d{1,3}[.)]\s+", ln)   # '1. ', '12) '
        if m:
            ln = ln[m.end():].strip()
            changed = True
    return re.sub(r"\s+", " ", ln)


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
        max_workers: int = 1,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
        scrubber: Optional[Callable[[str], str]] = None,
        detect_conflicts: bool = False,
        verify_coverage: Optional[bool] = None,
        coverage_keys: Optional[Callable[[str], List[str]]] = None,
        source_audit: Optional[bool] = None,
        numeric_audit: Optional[bool] = None,
    ):
        """Create a processor.

        ``verify_coverage``: when enabled, the final result is checked against
        the union of all map outputs, and ``result["coverage"]`` reports how
        many map-extracted items are missing. With the verbatim extraction
        preset (default when ``None``), missing lines are restored verbatim —
        deterministically, with no extra LLM calls — because a real merge
        model can silently drop lines even when the prompt forbids it. With
        custom prompts, pass ``coverage_keys`` (a callable extracting stable
        identifiers from any text) for report-only checking.

        ``source_audit``: the map phase itself can under-capture on
        unstructured input (measured: table rows 36%, action-log lines 48%
        dropped vs 2% for bullets) — and the map-union diff above cannot see
        what map never extracted. When enabled (default for the verbatim
        preset), the final result is additionally diffed against the SOURCE
        document's fact-bearing lines (lines containing a digit or currency
        symbol); missing lines are appended verbatim, deterministically, at
        zero LLM cost. ``result["source_audit"]`` reports the accounting.
        Pass ``False`` to disable, or ``True`` to force report-only checking
        for non-verbatim prompts. PII scrubbers are honored (the source is
        scrubbed before diffing).

        ``numeric_audit``: report-only check that every multi-digit figure
        cited in the final result exists in the source. Figures in the final
        result that the source does not contain are flagged in
        ``result["numeric_audit"]`` and ``result["warnings"]`` — measured
        failure mode: a real model reported a document total 15-38% off with
        no signal. Aggregates computed by the model (legitimate sums) will
        also be flagged, so treat this as a verification hint, not an error.
        On by default (zero cost, regex only); pass ``False`` to disable.
        """
        self.llm = llm if isinstance(llm, LLMAdapter) else LLMAdapter(llm)
        self.store = store or InMemoryContextStore()
        self.prompts = prompts or DEFAULT_PROMPTS
        # A map result larger than the merge budget can never be combined —
        # keep chunk inputs at most half the context window. Surface the
        # clamp: silent halving made users configure 2x more (and pricier)
        # LLM calls than they intended.
        effective_chunk = min(max_chunk_tokens, max(1, max_context_tokens // 2))
        if effective_chunk != max_chunk_tokens:
            logger.warning(
                "max_chunk_tokens=%d clamped to %d (half of max_context_tokens=%d): "
                "chunks larger than half the merge budget can never be combined. "
                "Raise max_context_tokens if you intended larger chunks.",
                max_chunk_tokens, effective_chunk, max_context_tokens)
        self.max_chunk_tokens = effective_chunk
        self.max_context_tokens = max_context_tokens
        self.max_repair_attempts = max_repair_attempts
        self.task_focus = task_focus
        self.max_workers = max(1, max_workers)
        self.on_event = on_event
        self.scrubber = scrubber
        self.detect_conflicts = detect_conflicts
        self.verify_coverage = verify_coverage
        self.coverage_keys = coverage_keys
        self.source_audit = source_audit
        self.numeric_audit = numeric_audit
        self._counter: TokenCounter = token_counter or self.store.counter
        self.graph = build_graph(
            llm=self.llm,
            store=self.store,
            token_counter=self._counter,
            prompts=self.prompts,
            max_workers=self.max_workers,
            on_event=self.on_event,
            scrubber=self.scrubber,
        )

    # ------------------------------------------------------------------

    def process(
        self,
        content: str,
        task: str,
        session_id: Optional[str] = None,
        task_focus: Optional[str] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """Run chunk → map → merge → repair over *content*.

        With ``resume=True`` a previously crashed run of the *same* content
        reuses its checkpointed map results (stored under ``{session}::run``)
        and only maps the chunks that never completed. The checkpoint is
        removed on success.
        """
        session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
        run_sid = f"{session_id}::run"

        logger.info("run started (session=%s, workers=%d, resume=%s)",
                    session_id, self.max_workers, resume)
        initial: Dict[str, Any] = {
            "input_content": content,
            "session_id": session_id,
            "task": task,
            "task_focus": task_focus or self.task_focus,
            "max_chunk_tokens": self.max_chunk_tokens,
            "max_context_tokens": self.max_context_tokens,
            "max_repair_attempts": self.max_repair_attempts,
            "max_workers": self.max_workers,
            "resume": resume,
            "processing_log": [],
        }
        result: MapReduceState = self.graph.invoke(initial)
        try:
            self.store.delete_session(run_sid)  # checkpoint no longer needed
        except Exception:
            pass

        logger.info("run completed (session=%s, chunks=%d, input_tokens=%d, repairs=%d)",
                    session_id, len(result.get("chunks", [])),
                    result.get("total_input_tokens", 0),
                    result.get("repair_attempts", 0))
        sources = [
            {"chunk_index": c.chunk_index, "chunk_id": c.id,
             "token_count": c.token_count, "preview": c.content[:120]}
            for c in result.get("chunks", [])
        ]
        out = {
            "session_id": session_id,
            "task": task,
            "final_result": result.get("final_result", ""),
            "total_input_tokens": result.get("total_input_tokens", 0),
            "num_chunks": len(result.get("chunks", [])),
            "merge_token_count": result.get("merge_token_count", 0),
            "repair_attempts": result.get("repair_attempts", 0),
            "processing_log": result.get("processing_log", []),
            "sources": sources,
        }

        # ------------------------------------------------------------------
        # Coverage verification & salvage. The map phase is faithful (verbatim
        # extraction returns every item), but a real merge model can silently
        # DROP lines even when the prompt forbids it — and the loss compounds
        # across hierarchical merge passes. Verify the final result against
        # the union of map outputs; recover missing verbatim lines
        # deterministically (no LLM cost) when the extraction preset produced
        # them; otherwise report the gap in `warnings`.
        warnings: List[str] = []
        map_results = result.get("map_results", [])
        verbatim_preset = self.prompts.map == VERBATIM_MAP_PROMPT
        enabled = (self.verify_coverage if self.verify_coverage is not None
                   else verbatim_preset)
        if enabled and map_results:
            extract = self.coverage_keys or self._default_coverage_keys
            map_items: set = set()
            for mr in map_results:
                map_items.update(extract(mr))
            final_items = set(extract(out["final_result"]))
            missing = sorted(map_items - final_items)
            out["coverage"] = {
                "expected": len(map_items),
                "found": len(map_items & final_items),
                "missing": len(missing),
                "salvaged": 0,
                "method": None,
            }
            if missing and verbatim_preset:
                out["final_result"] = (out["final_result"].rstrip()
                                       + "\n"
                                       + "\n".join(f"- {ln}" for ln in missing))
                out["coverage"]["salvaged"] = len(missing)
                out["coverage"]["method"] = "verbatim_append"
                missing = []
            elif missing:
                out["coverage"]["method"] = "report_only"
                warnings.append(
                    f"coverage check: {len(missing)} map-extracted item(s) "
                    "missing from final_result; no salvage available for "
                    "non-verbatim prompts (pass coverage_keys and inspect "
                    "result['coverage']).")
            if self.on_event:
                try:  # observability must never kill a run
                    self.on_event({"event": "coverage_verified",
                                   "session_id": session_id,
                                   "ts": _now_iso(),
                                   **out["coverage"]})
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Source-anchored audit. Coverage above can only restore lines the
        # MAP phase extracted — but on unstructured input the map model
        # itself under-captures (measured: pipe-table rows 36% and
        # action-log lines 48% dropped vs 2% for bullets). The caller holds
        # the source document, so diff the final result against the source's
        # fact-bearing lines: anything still missing is appended verbatim
        # (deterministic, zero LLM cost) for the verbatim preset, or
        # reported for other prompts. PII scrubbers are honored.
        sa_enabled = (self.source_audit if self.source_audit is not None
                      else verbatim_preset)
        if sa_enabled and content:
            src = self.scrubber(content) if self.scrubber else content
            src_items = {_normalize_item(ln) for ln in src.splitlines()
                         if _SOURCE_FACT_RE.search(ln) and ln.strip()}
            src_items.discard("")
            final_norm = {_normalize_item(ln) for ln in out["final_result"].splitlines()}
            sa_missing = sorted(src_items - final_norm)
            out["source_audit"] = {
                "source_fact_lines": len(src_items),
                "missing": len(sa_missing),
                "appended": 0,
                "method": None,
            }
            if sa_missing and verbatim_preset:
                out["final_result"] = (out["final_result"].rstrip()
                                       + "\n"
                                       + "\n".join(f"- {ln}" for ln in sa_missing))
                out["source_audit"]["appended"] = len(sa_missing)
                out["source_audit"]["method"] = "verbatim_append"
                sa_missing = []
            elif sa_missing:
                out["source_audit"]["method"] = "report_only"
                warnings.append(
                    f"source audit: {len(sa_missing)} source fact-bearing "
                    "line(s) absent from final_result; verbatim append is "
                    "only available for the verbatim extraction preset.")
            if self.on_event:
                try:
                    self.on_event({"event": "source_audit_completed",
                                   "session_id": session_id,
                                   "ts": _now_iso(),
                                   **out["source_audit"]})
                except Exception:
                    pass

        # ------------------------------------------------------------------
        # Numeric-consistency audit (report-only). A real model can report
        # an aggregate 15-38% off the source with no signal; any multi-digit
        # figure in the final result that the source does not contain is a
        # computed aggregate or a hallucination — either way the reader must
        # be told. Regex only, no LLM cost.
        if (self.numeric_audit if self.numeric_audit is not None else True) and content:
            def _fig_norm(s: str) -> str:
                return s.replace(",", "")

            src_figs = {_fig_norm(m) for m in _FIGURE_RE.findall(content)}
            final_figs = {_fig_norm(m) for m in _FIGURE_RE.findall(out["final_result"])}
            src_text_nocomma = content.replace(",", "")
            cited = {f for f in final_figs if f in src_text_nocomma}
            unverifiable = final_figs - cited
            out["numeric_audit"] = {
                "source_figures": len(src_figs),
                "cited": len(cited),
                "unverifiable_count": len(unverifiable),
                "unverifiable": sorted(unverifiable)[:8],
            }
            if unverifiable:
                examples = ", ".join(sorted(unverifiable)[:8])
                warnings.append(
                    f"numeric audit: {len(unverifiable)} figure(s) in "
                    f"final_result do not appear in the source ({examples}) — "
                    "computed aggregates or hallucinations; verify "
                    "independently.")
            if self.on_event:
                try:
                    self.on_event({"event": "numeric_audit_completed",
                                   "session_id": session_id,
                                   "ts": _now_iso(),
                                   **out["numeric_audit"]})
                except Exception:
                    pass

        # Repair is best-effort, not a guarantee: when the result still
        # exceeds the context budget after every attempt, say so explicitly
        # instead of returning an oversized result silently.
        out["merge_token_count"] = self._counter(out["final_result"])
        if out["merge_token_count"] > self.max_context_tokens:
            warnings.append(
                f"final_result is {out['merge_token_count']} tokens, above "
                f"the {self.max_context_tokens}-token context budget after "
                f"{out['repair_attempts']} repair attempt(s).")
        out["warnings"] = warnings

        if self.detect_conflicts:
            prompt = render_prompt(DEFAULT_PROMPTS.conflicts, {
                "final_result": out["final_result"]})
            raw = self.llm.complete(prompt)
            out["conflicts"] = [
                ln.split("CONFLICT:", 1)[1].strip()
                for ln in raw.splitlines()
                if ln.strip().upper().startswith("CONFLICT:")
            ]
        return out

    async def process_async(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Async wrapper — the event loop stays free while the (thread-offloaded)
        pipeline runs. Same signature and result as :meth:`process`."""
        return await asyncio.to_thread(self.process, *args, **kwargs)

    @staticmethod
    def _default_coverage_keys(text: str) -> List[str]:
        """Extract list items from LLM output, prefix-blind: bullets ('- ',
        '* ', '•'), numbered items ('1. '), table pipes and quotes are
        stripped, so '- FACT-1', '| FACT-1 |' and 'FACT-1' are the same item
        (a real merge model does not always preserve the bullet)."""
        out = []
        for ln in text.splitlines():
            item = _normalize_item(ln)
            if item:
                out.append(item)
        return out

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
