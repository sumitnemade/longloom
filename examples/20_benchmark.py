#!/usr/bin/env python3
"""20 — Scale benchmark: big document, hundreds of facts, measured throughput.

Generates a large synthetic enterprise document (configurable size), plants a
fact every few sentences, runs the full pipeline serially AND in parallel,
verifies no-loss at scale, and prints a benchmark report.

    python examples/20_benchmark.py                # scale run (fake LLM, no API cost)
    NVIDIA_API_KEY=... python examples/20_benchmark.py --real   # real NVIDIA run
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from longloom import InMemoryContextStore, MapReduceProcessor, PromptTemplates


import re as _re

_FACT_RE = _re.compile(r"[\d@]|INC-|INV-|AGR-|POL-|tr-|ERR_", _re.IGNORECASE)
_CHUNK_HEADER = _re.compile(r"^CHUNK \d+/\d+:\s*$", _re.MULTILINE)


def _fact_lines(text: str) -> str:
    lines = [ln for ln in text.splitlines() if _FACT_RE.search(ln)]
    return "\n".join(f"- {ln.strip('- ').strip()}" for ln in lines if ln.strip()) if lines else ""


class FaithfulExtractor:
    """Deterministic LLM that copies every fact-bearing line verbatim.

    Stage-aware: for the map phase it reads ONLY the chunk-content section
    (after the CHUNK i/n header) — never the prompt instructions — so no
    prompt text can leak into results. With this backend, no-loss measures
    the PIPELINE, not an LLM's summarization quality."""

    def __call__(self, prompt: str) -> str:
        if "CHUNK RESULTS:" in prompt:                       # merge (any pass)
            body = prompt.split("CHUNK RESULTS:", 1)[1]
            return _fact_lines(body) or "(none)"
        if "CURRENT RESULT:" in prompt:                      # repair
            body = prompt.split("CURRENT RESULT:", 1)[1]
            return _fact_lines(body) or "(none)"
        m = _CHUNK_HEADER.search(prompt)                     # map
        if m:
            return _fact_lines(prompt[m.end():]) or "(no facts)"
        return "(no facts)"

SECTIONS = [
    ("server logs",   "2026-{m:02d}-{d:02d} {h:02d}:{mi:02d} UTC {level} svc=svc-{n} trace_id=tr-{tid} latency={lat}ms code=ERR_{code}"),
    ("incidents",     '{{"incident_id": "INC-{n}", "severity": "sev{s}", "affected": ["host-{a}", "host-{b}"], "loss_usd": {usd}}}'),
    ("contracts",     "Agreement AGR-{n} executed {date} between Party{p} and Party{q}; value ${val}, term {term} months, renewal auto."),
    ("invoices",      "Invoice INV-{n} dated {date}: ${val} due net-{net}, PO-{po}, cost center CC-{cc}."),
    ("findings",      "Finding F-{n} (priority P{s}): metric {metric} measured {val}{unit} against baseline {base}{unit}; owner team-{t}."),
    ("policies",      "Policy POL-{n} v{v}: retention {r} days, encryption AES-256, region {region}, audit quarterly."),
]

METRICS = ["p99_latency", "error_rate", "cpu_util", "mem_util", "disk_io", "throughput", "mttr", "availability"]
UNITS = ["ms", "%", "GB/s", "min", "hours", "req/s"]


def build_document(num_sections: int, seed: int = 42) -> tuple[str, list[str]]:
    """Large synthetic doc; every section embeds 3-5 unique fact tokens."""
    rng = random.Random(seed)
    facts: list[str] = []
    parts: list[str] = []
    n = 0

    def uniq(prefix: str) -> str:
        nonlocal n
        n += 1
        return f"{prefix}-{n:05d}"

    for i in range(num_sections):
        kind, template = SECTIONS[i % len(SECTIONS)]
        n = rng.randrange(100, 99999)
        vals = {
            "m": rng.randint(1, 12), "d": rng.randint(1, 28), "h": rng.randint(0, 23),
            "mi": rng.randint(0, 59), "level": rng.choice(["ERROR", "WARN", "INFO"]),
            "n": rng.randrange(100, 99999), "tid": rng.randrange(10**6, 10**9),
            "lat": rng.randint(5, 9999), "code": rng.randint(100, 999),
            "s": rng.randint(1, 4), "a": rng.randrange(10, 99), "b": rng.randrange(10, 99),
            "usd": rng.randrange(1000, 999999), "p": rng.randrange(1, 50),
            "q": rng.randrange(1, 50), "date": f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
            "val": rng.randrange(1000, 9999999), "term": rng.randint(3, 60),
            "net": rng.choice([15, 30, 45, 60]), "po": rng.randrange(10000, 99999),
            "cc": rng.randrange(100, 999), "metric": rng.choice(METRICS),
            "unit": rng.choice(UNITS), "base": rng.randint(1, 500),
            "t": rng.randint(1, 20), "v": rng.randint(1, 9), "r": rng.choice([30, 90, 180, 365]),
            "region": rng.choice(["us-east-1", "eu-west-1", "ap-south-1", "us-west-2"]),
        }
        text = template.format(**vals)
        parts.append(f"===== section {i+1}: {kind} =====\n{text}\n")

        # plant fact tokens THAT EXIST in this section's text (per kind)
        section_facts = {
            "server logs": [f"tr-{vals['tid']}", f"{vals['lat']}ms", f"ERR_{vals['code']}"],
            "incidents":  [f"INC-{vals['n']}", f"host-{vals['a']}", f'"loss_usd": {vals["usd"]}'], 
            "contracts":  [f"AGR-{vals['n']}", f"Party{vals['p']}", f"${vals['val']}"],
            "invoices":   [f"INV-{vals['n']}", f"PO-{vals['po']}", f"CC-{vals['cc']}"],
            "findings":   [f"F-{vals['n']}", f"team-{vals['t']}", f"{vals['base']}{vals['unit']}"],
            "policies":   [f"POL-{vals['n']}", vals["region"], f"v{vals['v']}"],
        }[kind]
        facts.extend(section_facts)

    doc = "\n".join(parts)
    seen, dedup = set(), []
    for f in facts:
        if f not in seen:
            seen.add(f)
            dedup.append(f)
    for f in dedup:  # every planted fact MUST exist in the document text
        assert f in doc, f"fixture bug: fact {f!r} not found in document"
    return doc, dedup


def run(label: str, processor, doc: str, task: str, session_id: str) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = processor.process(doc, task=task, session_id=session_id)
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", type=int, default=400,
                        help="number of synthetic sections (each ~3-6 lines)")
    parser.add_argument("--real", action="store_true",
                        help="use the real NVIDIA API instead of the offline demo LLM")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    use_real = args.real and (
        os := __import__("os")).environ.get("NVIDIA_API_KEY")

    doc, facts = build_document(args.sections)
    doc_chars = len(doc)
    print(f"document: {doc_chars:,} chars | planted facts: {len(facts)} | "
          f"sections: {args.sections}")

    # ── serial vs parallel throughput (offline demo LLM) ────────────────
    print("\n── throughput benchmark (deterministic offline LLM) ──")
    results = {}
    for workers in (1, 4, 8):
        processor = MapReduceProcessor(
            llm=FaithfulExtractor(), store=InMemoryContextStore(),
            max_chunk_tokens=600, max_context_tokens=8192,
            prompts=PromptTemplates.for_extraction(),
            max_workers=workers,
        )
        t0 = time.perf_counter()
        r = processor.process(doc, task="Extract every fact verbatim",
                              session_id=f"bench-{workers}")
        dt = time.perf_counter() - t0
        tokens = r["total_input_tokens"]
        throughput = tokens / dt if dt else 0
        results[workers] = (dt, r)
        print(f"  workers={workers}: {dt:6.2f}s | {tokens:,} tokens | "
              f"{throughput:8.0f} tok/s | {r['num_chunks']} chunks")

    speedup = results[1][0] / results[8][0] if results[8][0] else 0
    print(f"  speedup (1 -> 8 workers): {speedup:.1f}x")

    # ── no-loss verification at scale ────────────────────────────────────
    processor = MapReduceProcessor(
        llm=FaithfulExtractor(), store=InMemoryContextStore(),
        max_chunk_tokens=600, max_context_tokens=8192,
        prompts=PromptTemplates.for_extraction(),
        max_workers=8,
    )
    r = processor.process(doc, task="Extract every fact verbatim",
                          session_id="bench-noloss")
    final = r["final_result"]
    preserved = sum(1 for f in facts if f in final)
    print(f"\n── no-loss verification ──")
    print(f"  facts planted : {len(facts)}")
    print(f"  facts in final: {preserved} ({100*preserved/len(facts):.1f}%)")
    missing = [f for f in facts if f not in final]
    if missing:
        print(f"  MISSING: {missing[:10]}")

    # ── real LLM run (optional) ─────────────────────────────────────────
    if use_real:
        from longloom import init_llm

        print("\n── real NVIDIA verification ──")
        llm = init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b",
                       api_key=os.environ["NVIDIA_API_KEY"],
                       temperature=0.0, max_tokens=2048, thinking=False)
        events: list[dict] = []
        rp = MapReduceProcessor(
            llm=llm, store=InMemoryContextStore(),
            max_chunk_tokens=900, max_context_tokens=8192,
            prompts=PromptTemplates.for_extraction(),
            max_workers=args.max_workers,
            on_event=events.append,
        )
        rr = rp.process(doc, task="Extract every fact and identifier verbatim",
                        session_id="bench-real")
        preserved = sum(1 for f in facts if f in rr["final_result"])
        usage_in = sum(e.get("usage", {}).get("input_tokens", 0)
                       for e in events if e["event"] == "map_completed")
        print(f"  real run: {rr['num_chunks']} chunks | facts preserved: "
              f"{preserved}/{len(facts)} | LLM input tokens: {usage_in:,}")

    print(f"\nsummary: {doc_chars:,} chars | {len(facts)} facts | "
          f"no-loss: {preserved}/{len(facts)} | parallel speedup: {speedup:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())