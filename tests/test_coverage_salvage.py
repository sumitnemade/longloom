"""Coverage verification & salvage: a real merge model can silently DROP
lines even when the prompt forbids it (measured: 36% loss at 120 facts,
61% at 200 facts with a live model). The processor must detect the drop and,
for the verbatim extraction preset, restore every map-extracted line
deterministically — with no extra LLM calls.

Also covers the token-budget transparency warning: after all repair attempts
the result may still exceed the context budget; that must surface in
``result["warnings"]`` instead of being returned silently.
"""
from __future__ import annotations

import re

from longloom import MapReduceProcessor, InMemoryContextStore
from longloom.prompts import DEFAULT_PROMPTS, PromptTemplates
from longloom.tokens import EstimateCounter

MAP_CHUNK_RE = re.compile(r"CHUNK \d+/\d+:\n(.*)$", re.S)
FACTS = [f"FACT-{i:03d}|val={i * 7}|host=node{i}.example" for i in range(1, 31)]
DOC = "\n".join(FACTS)


def fact_lines(text: str) -> list:
    """Fact lines, bullet prefix removed (canonical form, like map output)."""
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if "FACT-" in ln:
            out.append(ln[1:].strip() if ln.startswith("- ") else ln)
    return out


class DroppingMergeLLM:
    """Perfect verbatim map; merge model that silently drops every 3rd line."""

    def __init__(self):
        self.map_calls = 0
        self.merge_calls = 0

    def __call__(self, prompt: str) -> str:
        if "Copy VERBATIM every line" in prompt:          # verbatim map
            self.map_calls += 1
            body = MAP_CHUNK_RE.search(prompt).group(1)
            return "\n".join("- " + ln for ln in fact_lines(body))
        if "Remove EXACT duplicate lines only" in prompt:  # verbatim merge
            self.merge_calls += 1
            lines = fact_lines(prompt.split("CHUNK RESULTS:\n", 1)[1])
            kept = [ln for i, ln in enumerate(lines) if i % 3 != 0]
            return "\n".join("- " + ln for ln in dict.fromkeys(kept))
        raise AssertionError("unexpected prompt: " + prompt[:120])


class KeepingLLM:
    """Faithful at every phase (for the token-budget warning test)."""

    def __call__(self, prompt: str) -> str:
        if "Copy VERBATIM every line" in prompt:
            body = MAP_CHUNK_RE.search(prompt).group(1)
            return "\n".join("- " + ln for ln in fact_lines(body))
        if "Remove EXACT duplicate lines only" in prompt:
            lines = fact_lines(prompt.split("CHUNK RESULTS:\n", 1)[1])
            return "\n".join("- " + ln for ln in dict.fromkeys(lines))
        if "Remove only entries that are redundant" in prompt:  # echo repair
            body = prompt.split("CURRENT RESULT:\n", 1)[1]
            return body.split("\n\nReturn", 1)[0].strip()
        raise AssertionError("unexpected prompt: " + prompt[:120])


class GenericDroppingLLM:
    """Default (non-verbatim) prompts: map lists fact IDs, merge drops some."""

    def __call__(self, prompt: str) -> str:
        if "Your task is to" in prompt:
            body = MAP_CHUNK_RE.search(prompt).group(1)
            return "\n".join("ID " + ln.split("|")[0] for ln in fact_lines(body))
        if "Combine these into a single coherent result" in prompt:
            ids = re.findall(r"ID (FACT-\d{3})",
                             prompt.split("CHUNK RESULTS:\n", 1)[1])
            return "\n".join("ID " + i for i in dict.fromkeys(ids)
                             if int(i.split("-")[1]) % 3 != 0)
        raise AssertionError("unexpected prompt: " + prompt[:120])


def run(llm, prompts, **kw):
    kw.setdefault("max_chunk_tokens", 60)   # force multiple chunks → merge runs
    proc = MapReduceProcessor(
        llm=llm, store=InMemoryContextStore(), prompts=prompts,
        token_counter=EstimateCounter(), **kw)
    return proc.process(DOC, task="Extract all facts verbatim",
                        session_id="cov-test")


# ---------------------------------------------------------------- salvage
def test_verbatim_salvage_restores_dropped_lines():
    out = run(DroppingMergeLLM(), PromptTemplates.for_extraction())
    assert out["coverage"]["expected"] == 30
    assert out["coverage"]["missing"] == 10
    assert out["coverage"]["salvaged"] == 10
    assert out["coverage"]["method"] == "verbatim_append"
    final_ids = set(re.findall(r"FACT-\d{3}", out["final_result"]))
    assert final_ids == {f"FACT-{i:03d}" for i in range(1, 31)}
    # every salvaged line stays verbatim
    for line in FACTS:
        assert line in out["final_result"]
    # no warning when the result fits the budget
    assert out["warnings"] == []


def test_coverage_disabled_keeps_previous_behaviour():
    out = run(DroppingMergeLLM(), PromptTemplates.for_extraction(),
              verify_coverage=False)
    assert "coverage" not in out
    assert out["warnings"] == []
    final_ids = set(re.findall(r"FACT-\d{3}", out["final_result"]))
    assert len(final_ids) == 20  # the merge drop is silent, as before


def test_generic_prompts_report_only():
    keys = lambda t: re.findall(r"FACT-\d{3}", t)  # noqa: E731
    out = run(GenericDroppingLLM(), DEFAULT_PROMPTS,
              verify_coverage=True, coverage_keys=keys)
    assert out["coverage"]["expected"] == 30
    assert out["coverage"]["missing"] == 10
    assert out["coverage"]["salvaged"] == 0
    assert out["coverage"]["method"] == "report_only"
    # nothing appended — report-only for non-verbatim prompts
    final_ids = set(re.findall(r"FACT-\d{3}", out["final_result"]))
    assert len(final_ids) == 20
    assert any("missing from final_result" in w for w in out["warnings"])


def test_auto_enabled_for_verbatim_preset_only():
    # verbatim preset + verify_coverage=None → auto-on
    out = run(DroppingMergeLLM(), PromptTemplates.for_extraction())
    assert out["coverage"]["salvaged"] == 10
    # default prompts + verify_coverage=None → auto-off
    out2 = run(GenericDroppingLLM(), DEFAULT_PROMPTS)
    assert "coverage" not in out2


# ------------------------------------------------------- budget warning
def test_oversized_result_warns_instead_of_silence():
    out = run(KeepingLLM(), PromptTemplates.for_extraction(),
              max_chunk_tokens=60, max_context_tokens=200,
              max_repair_attempts=3)
    assert out["repair_attempts"] == 3
    assert out["coverage"]["missing"] == 0  # faithful merge, nothing to salvage
    assert out["merge_token_count"] > 200
    assert any("above the 200-token context budget" in w for w in out["warnings"])
    # every fact is still present (no-loss prioritized over the budget)
    final_ids = set(re.findall(r"FACT-\d{3}", out["final_result"]))
    assert final_ids == {f"FACT-{i:03d}" for i in range(1, 31)}
