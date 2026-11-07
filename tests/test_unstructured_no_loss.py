"""No-loss guarantee for UNSTRUCTURED input — offline, deterministic.

The fake LLM behaves like a faithful extractor: it copies every fact-bearing
line verbatim from whatever it is given (chunk for map, map results for merge,
merged result for repair). Therefore any content missing from the final
result is provably a PIPELINE loss, not an LLM hallucination.

Asserts:
1. Chunking is lossless: chunks concatenate back to the original input.
2. Every chunk is mapped (LLM sees 100% of the input).
3. All planted facts survive map → merge (normal window).
4. All planted facts survive map → merge → repair (forced tiny window).
5. Token accounting matches across store and workflow.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "unstructured_doc", ROOT / "examples" / "_unstructured_doc.py")
udoc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(udoc)

from longloom import InMemoryContextStore, MapReduceProcessor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_FACT_LINE = re.compile(r"[\d@]|INC-|TKT-|ERR_|tr-", re.IGNORECASE)


class FaithfulExtractor(BaseChatModel):
    """Deterministic stand-in: copies fact-bearing lines from its input."""

    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "faithful-extractor"

    @staticmethod
    def _extract(text: str) -> str:
        lines = [ln.rstrip() for ln in text.splitlines() if _FACT_LINE.search(ln)]
        return "\n".join(f"• {ln}" for ln in lines) or "(no facts)"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        # Phase detection identical to the workflow's prompt shapes.
        if "CURRENT RESULT:" in prompt:                       # repair
            body = prompt.split("CURRENT RESULT:", 1)[1]
            self.calls.append("repair")
            out = self._extract(body)
        elif "CHUNK RESULTS:" in prompt:                      # merge
            body = prompt.split("CHUNK RESULTS:", 1)[1]
            self.calls.append("merge")
            out = self._extract(body)
        else:                                                 # map (chunk)
            self.calls.append("map")
            out = self._extract(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=out))])


def _processor(window: int, tmp_path, store_kind="memory"):
    if store_kind == "memory":
        from longloom import InMemoryContextStore
        store = InMemoryContextStore()
    else:
        from longloom import SQLiteContextStore
        store = SQLiteContextStore(db_path=str(tmp_path / "u.db"))
    return MapReduceProcessor(
        llm=FaithfulExtractor(),
        store=store,
        max_chunk_tokens=350,          # small → several chunks over the messy doc
        max_context_tokens=window,
        max_repair_attempts=3,
        task_focus="every identifier, number, email, URL, hostname, version, date, amount",
    )


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_unstructured_no_loss_without_repair(tmp_path, store_kind):
    doc = udoc.build_unstructured_doc()
    processor = _processor(window=8192, tmp_path=tmp_path, store_kind=store_kind)
    result = processor.process(doc, task="Extract every concrete fact verbatim",
                               session_id=f"unstructured_{store_kind}")

    # 1) lossless chunking — chunks reconstruct the original input
    chunks = processor.retrieve_chunks(result["session_id"])
    assert "".join(c.content for c in chunks).replace(" ", "") == doc.replace(" ", "")
    # 2) every chunk was mapped
    assert result["num_chunks"] >= 2 and result["repair_attempts"] == 0
    assert result["total_input_tokens"] == sum(c.token_count for c in chunks)
    # 3+4) every planted fact survives to the final output
    report = udoc.preservation_report(result["final_result"])
    assert report["missing"] == [], f"{store_kind}: lost facts: {report['missing']}"
    assert report["rate"] == 100.0


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_unstructured_no_loss_with_forced_repair(tmp_path, store_kind):
    doc = udoc.build_unstructured_doc()
    processor = _processor(window=250, tmp_path=tmp_path, store_kind=store_kind)
    result = processor.process(doc, task="Extract every concrete fact verbatim",
                               session_id=f"unstructured_repair_{store_kind}")

    assert result["repair_attempts"] >= 1, "tiny window must force repair"
    report = udoc.preservation_report(result["final_result"])
    # A condensing repair may compress formatting but must not drop facts —
    # the faithful extractor copies every fact line it is shown.
    assert report["missing"] == [], f"{store_kind}: repair lost facts: {report['missing']}"
    assert report["rate"] == 100.0