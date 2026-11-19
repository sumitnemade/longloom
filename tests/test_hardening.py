"""Hardening fixes from the real-LLM validation campaign (Nov 2026).

Covers: source-anchored audit (map-level under-capture), prefix-blind
coverage keys, RateLimitError with its own backoff budget, the
max_chunk_tokens clamp warning, the numeric-consistency audit, and the
hardened verbatim prompts.
"""
from __future__ import annotations

import logging
import re

import pytest

from longloom import (InMemoryContextStore, LLMAdapter, MapReduceProcessor,
                      RateLimitError)
from longloom.prompts import DEFAULT_PROMPTS, VERBATIM_MAP_PROMPT, VERBATIM_MERGE_PROMPT, PromptTemplates
from longloom.tokens import EstimateCounter

MAP_CHUNK_RE = re.compile(r"CHUNK \d+/\d+:\n(.*)$", re.S)
FACTS = [f"FACT-{i:03d}|val={i * 7}|host=node{i}.example" for i in range(1, 31)]
DOC = "\n".join(FACTS)


def fact_lines(text: str) -> list:
    out = []
    for ln in text.splitlines():
        ln = ln.strip()
        if "FACT-" in ln:
            out.append(ln[1:].strip() if ln.startswith("- ") else ln)
    return out


def run(llm, prompts, **kw):
    kw.setdefault("max_chunk_tokens", 60)   # force multiple chunks -> merge runs
    proc = MapReduceProcessor(
        llm=llm, store=InMemoryContextStore(), prompts=prompts,
        token_counter=EstimateCounter(), **kw)
    return proc.process(DOC, task="Extract all facts verbatim",
                        session_id="hardening-test")


class VerbatimMapMergeLLM:
    """Faithful verbatim map + faithful merge (baseline fake)."""

    def __call__(self, prompt: str) -> str:
        if "Copy VERBATIM every line" in prompt:
            body = MAP_CHUNK_RE.search(prompt).group(1)
            return "\n".join("- " + ln for ln in fact_lines(body))
        if "Remove EXACT duplicate lines only" in prompt:
            lines = fact_lines(prompt.split("CHUNK RESULTS:\n", 1)[1])
            return "\n".join("- " + ln for ln in dict.fromkeys(lines))
        raise AssertionError("unexpected prompt: " + prompt[:120])


class DroppingMapLLM(VerbatimMapMergeLLM):
    """Map silently skips the line matching ``skip`` — the map-level loss the
    coverage (map-union) diff cannot see; only the source audit recovers it."""

    def __init__(self, skip):
        self.skip = skip

    def __call__(self, prompt: str) -> str:
        if "Copy VERBATIM every line" in prompt:
            body = MAP_CHUNK_RE.search(prompt).group(1)
            lines = [ln for ln in fact_lines(body) if not self.skip(ln)]
            return "\n".join("- " + ln for ln in lines)
        return super().__call__(prompt)


# ------------------------------------------------------- source audit
def test_source_audit_recovers_map_level_drops():
    # FACT-007 never enters the map output -> invisible to coverage salvage;
    # the source audit must restore it deterministically.
    out = run(DroppingMapLLM(lambda ln: "FACT-007" in ln),
              PromptTemplates.for_extraction())
    assert out["coverage"]["expected"] == 29        # map union lost one
    sa = out["source_audit"]
    assert sa["source_fact_lines"] == 30
    assert sa["missing"] == 1 and sa["appended"] == 1
    assert sa["method"] == "verbatim_append"
    assert "FACT-007|val=49|host=node7.example" in out["final_result"]
    for line in FACTS:                              # 30/30 byte-exact
        assert line in out["final_result"]
    assert out["warnings"] == []


def test_source_audit_report_only_for_custom_prompts():
    class DefaultLLM:
        def __call__(self, prompt: str) -> str:
            if "Your task is to" in prompt:
                body = MAP_CHUNK_RE.search(prompt).group(1)
                return "\n".join("ID " + ln.split("|")[0]
                                 for ln in fact_lines(body) if "FACT-00" not in ln)
            if "Combine these into a single coherent result" in prompt:
                ids = re.findall(r"ID (FACT-\d{3})",
                                 prompt.split("CHUNK RESULTS:\n", 1)[1])
                return "\n".join("ID " + i for i in dict.fromkeys(ids))
            raise AssertionError("unexpected prompt: " + prompt[:120])

    out = run(DefaultLLM(), DEFAULT_PROMPTS, source_audit=True)
    sa = out["source_audit"]
    assert sa["method"] == "report_only"
    assert sa["appended"] == 0 and sa["missing"] > 0
    assert any("source audit:" in w for w in out["warnings"])
    # nothing appended: summaries must not receive raw source lines
    assert "FACT-001|val=7" not in out["final_result"]


def test_source_audit_can_be_disabled():
    out = run(DroppingMapLLM(lambda ln: "FACT-007" in ln),
              PromptTemplates.for_extraction(), source_audit=False)
    assert "source_audit" not in out
    assert "FACT-007" not in out["final_result"]


def test_source_audit_honors_scrubber():
    scrub = lambda text: text.replace("val=7|", "val=REDACTED|")  # noqa: E731
    out = run(VerbatimMapMergeLLM(), PromptTemplates.for_extraction(),
              scrubber=scrub)
    assert "val=7|" not in out["final_result"]          # no raw PII re-appended
    assert "FACT-001|val=REDACTED|host=node1.example" in out["final_result"]
    assert out["source_audit"]["missing"] == 0


# ------------------------------------ prefix-blind coverage keys
def test_default_coverage_keys_are_prefix_blind():
    keys = MapReduceProcessor._default_coverage_keys(
        "- FACT-001\n| FACT-002 | x |\n1. FACT-003\n* FACT-004\n\nprose line")
    assert keys == ["FACT-001", "FACT-002 | x |", "FACT-003", "FACT-004",
                    "prose line"]


def test_coverage_counts_raw_table_rows_from_merge():
    """The BRD experiment measured 15 merge-output lines invisible to the
    bullet-only extractor. Table rows returned without the '- ' prefix must
    now be counted (no phantom 'missing' -> no spurious salvage)."""
    rows = [f"| FACT-{i:03d} | val={i} |" for i in range(1, 13)]
    doc = "\n".join(rows)

    class RawRowMergeLLM(VerbatimMapMergeLLM):
        def __call__(self, prompt: str) -> str:
            if "Copy VERBATIM every line" in prompt:
                body = MAP_CHUNK_RE.search(prompt).group(1)
                return "\n".join("- " + ln.strip()
                                 for ln in body.splitlines() if "FACT-" in ln)
            if "Remove EXACT duplicate lines only" in prompt:
                lines = [ln.strip() for ln in
                         prompt.split("CHUNK RESULTS:\n", 1)[1].splitlines()
                         if "FACT-" in ln]
                return "\n".join(dict.fromkeys(lines))   # raw rows, no bullet
            raise AssertionError("unexpected prompt: " + prompt[:120])

    proc = MapReduceProcessor(
        llm=RawRowMergeLLM(), store=InMemoryContextStore(),
        prompts=PromptTemplates.for_extraction(),
        token_counter=EstimateCounter(), max_chunk_tokens=40)
    out = proc.process(doc, task="Extract all facts verbatim",
                       session_id="rows-test")
    assert out["coverage"]["expected"] == 12
    assert out["coverage"]["found"] == 12
    assert out["coverage"]["missing"] == 0
    assert out["coverage"]["salvaged"] == 0
    assert out["source_audit"]["missing"] == 0
    assert out["warnings"] == []


# ------------------------------------------------- rate limiting
def test_rate_limit_error_has_its_own_budget(monkeypatch):
    import time as _time
    sleeps: list = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s) if s > 0 else None)

    calls = {"n": 0}

    def always_rl(prompt: str) -> str:
        calls["n"] += 1
        raise RateLimitError("429 quota window")

    adapter = LLMAdapter(always_rl, retry_attempts=2, retry_backoff=0.01,
                         rate_limit_attempts=3, rate_limit_backoff=0.01,
                         rate_limit_max_sleep=0.02)
    with pytest.raises(RateLimitError):
        adapter.complete("p")
    assert calls["n"] == 4              # 1 initial + 3 rate-limit retries
    assert len(sleeps) == 3             # generic budget untouched


def test_rate_limit_recovers_and_budgets_are_independent(monkeypatch):
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] in (1, 2):
            raise RateLimitError("429")
        if calls["n"] == 3:
            raise ConnectionError("timeout")
        return "ok"

    adapter = LLMAdapter(flaky, retry_attempts=1, retry_backoff=0.01,
                         rate_limit_attempts=3, rate_limit_backoff=0.01)
    assert adapter.complete("p") == "ok"
    assert calls["n"] == 4              # 2 rate-limit + 1 generic + success


def test_rate_limit_error_exported_from_package():
    import longloom
    assert longloom.RateLimitError is RateLimitError


# --------------------------------------------- chunk clamp warning
def test_chunk_tokens_clamp_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="longloom.processor"):
        proc = MapReduceProcessor(llm=lambda p: "x", store=InMemoryContextStore(),
                                  max_chunk_tokens=5000, max_context_tokens=2000)
    assert proc.max_chunk_tokens == 1000
    assert any("clamped" in r.message for r in caplog.records)


# ------------------------------------------------- numeric audit
FIG_DOC = ("Total programme budget: 57,176,990 USD across three years.\n"
           "Go-live milestone: 30 Jun 2027.\n"
           "Largest single entry: 994,473 USD.\n"
           "Availability SLA: 99.95 percent.")


class SummarizingLLM:
    """Default prompts: map 'summarizes' each chunk into one line."""

    def __init__(self, hallucinate=True):
        self.hallucinate = hallucinate

    def __call__(self, prompt: str) -> str:
        if "Your task is to" in prompt:
            body = MAP_CHUNK_RE.search(prompt).group(1)
            figs = [ln.split(":")[1].split(" USD")[0].strip()
                    for ln in body.splitlines() if ":" in ln]
            out = "Chunk notes: " + ", ".join(figs)
            if self.hallucinate:
                out += ", grand total 35,610,096 USD"
            return out
        if "Combine these into a single coherent result" in prompt:
            return prompt.split("CHUNK RESULTS:\n", 1)[1].strip()
        raise AssertionError("unexpected prompt: " + prompt[:120])


def _summarize(llm, **kw):
    proc = MapReduceProcessor(llm=llm, store=InMemoryContextStore(),
                              token_counter=EstimateCounter(),
                              max_chunk_tokens=40, max_context_tokens=3000,
                              **kw)
    return proc.process(FIG_DOC, task="Write an executive summary",
                        session_id="numaudit-test")


def test_numeric_audit_flags_figures_absent_from_source():
    out = _summarize(SummarizingLLM(hallucinate=True))
    na = out["numeric_audit"]
    assert na["source_figures"] >= 4
    assert "35610096" in na["unverifiable"]
    assert na["unverifiable_count"] >= 1
    assert any("numeric audit" in w for w in out["warnings"])


def test_numeric_audit_silent_when_every_figure_is_in_source():
    out = _summarize(SummarizingLLM(hallucinate=False))
    assert out["numeric_audit"]["unverifiable_count"] == 0
    assert not any("numeric audit" in w for w in out["warnings"])


def test_numeric_audit_can_be_disabled():
    out = _summarize(SummarizingLLM(hallucinate=True), numeric_audit=False)
    assert "numeric_audit" not in out


# --------------------------------------------------- prompt hardening
def test_verbatim_prompts_cover_non_bullet_fact_shapes():
    assert "pipe-delimited table rows" in VERBATIM_MAP_PROMPT
    assert "action/decision/meeting-log" in VERBATIM_MAP_PROMPT
    assert "Copy VERBATIM every line" in VERBATIM_MAP_PROMPT      # compat
    assert "Remove EXACT duplicate lines only" in VERBATIM_MERGE_PROMPT
    assert "verify that every unique input line" in VERBATIM_MERGE_PROMPT


def test_default_merge_prompt_has_self_check():
    assert "Before returning, verify" in DEFAULT_PROMPTS.merge
