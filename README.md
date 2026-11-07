<div align="center">

# 🧶 LongLoom

**Weave documents of any length through LLM map/merge/repair — without losing a thread.**

[![PyPI](https://img.shields.io/pypi/v/longloom)](https://pypi.org/project/longloom/)
[![Python](https://img.shields.io/pypi/pyversions/longloom)](https://pypi.org/project/longloom/)
[![CI](https://github.com/sumitnemade/longloom/actions/workflows/ci.yml/badge.svg)](https://github.com/sumitnemade/longloom/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-typed-blue)](longloom/py.typed)
[![Tests](https://img.shields.io/badge/tests-41%20passing-brightgreen)](#test-it-yourself)

*Any LLM · Any storage · Measured no-loss · 2 core dependencies*

</div>

---

Your model accepts 8K, 128K, or 200K tokens — your documents don't care.
Annual reports, BRDs, legal contracts, log archives, codebases: they all blow
past the window, and naive truncation silently drops the facts you need.

**LongLoom** splits the problem the way humans do — read in chunks, extract,
combine, condense — as a LangGraph pipeline with **measured** information
preservation, over **any LLM** you already use and **any storage** you already run.

```text
              ┌───────────────┐
 big document ─► CHUNK + STORE ─► every byte persisted (SQLite / Postgres / Mongo / …)
              └───────┬───────┘
                      ▼   MAP      one LLM call per chunk — verbatim or summarizing
              ┌───────────────┐
              │  map results  │
              └───────┬───────┘
                      ▼   MERGE   window-sized hierarchical combine calls
              ┌───────────────┐           (scales past any single context window)
              │  merged draft │
              └───────┬───────┘
                      ▼   REPAIR  bounded condensation loop, only if still too big
              ┌───────────────┐
              │ final result  │  +  verification: every fact accounted for
              └───────────────┘
```

## Install

```bash
pip install "longloom[all]"      # batteries included
```

or just the core (two dependencies: `langgraph` + `langchain-core`):

```bash
pip install longloom
```

Python ≥ 3.10 · Optional extras: `[tiktoken]` `[nvidia]` `[openai]` `[postgres]` `[mysql]` `[mongodb]` `[supabase]` `[databases]`

## 60-second quickstart

```python
from longloom import MapReduceProcessor, SQLiteContextStore, init_llm

processor = MapReduceProcessor(
    llm=init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", thinking=False),
    store=SQLiteContextStore("ctx.db"),
    max_chunk_tokens=3000,
    max_context_tokens=8192,          # your model's real window
)

result = processor.process(
    annual_report,                    # 50K tokens? 500K? doesn't matter
    task="Summarize the key findings",
)
print(result["final_result"])
print(f"{result['num_chunks']} chunks · {result['total_input_tokens']:,} tokens · "
      f"{result['repair_attempts']} repairs")
```

Every run is **persisted and queryable** afterwards:

```python
processor.search_session(result["session_id"], "revenue")   # which chunks mention it
processor.retrieve_chunks(result["session_id"])              # original text, byte-faithful
processor.get_session_info(result["session_id"])             # chunk/token stats
```

## When facts must not be lost

Summarizing prompts measurably drop facts (we measured 2–10% on messy
documents — different facts every run). When you need *everything*:

```python
from longloom import PromptTemplates

processor = MapReduceProcessor(
    llm=llm,
    store=store,
    prompts=PromptTemplates.for_extraction(),   # verbatim line-copy mode
    ...
)
```

**Measured result: 100% fact preservation (41/41 planted facts in a log/JSON/
email/table/document mix — stable across repeated real-LLM runs on NVIDIA and
OpenAI).** The trade-off is output length: extraction copies, it doesn't
compress. Summaries → `DEFAULT_PROMPTS`; completeness → `for_extraction()`.

## Any LLM

```python
init_llm("nvidia",    "nvidia/nemotron-3-ultra-550b-a55b", thinking=False)
init_llm("openai",    "gpt-4o-mini")
init_llm("anthropic", "claude-sonnet-4-20250514")
init_llm("groq" | "ollama" | "mistral" | "deepseek" | "together", ...)
```

…or bring your own: any LangChain `BaseChatModel`, or a bare function —
the minimum contract is `str → str`:

```python
processor = MapReduceProcessor(llm=lambda prompt: my_gateway(prompt), store=store)
```

Provider quirks are handled for you: NVIDIA reasoning models get real model
IDs, a `thinking` toggle, and a `content=None` starvation guard; OpenAI
`extra_body` and temperature restrictions are abstracted away.

## Any storage

```python
from longloom import create_store

store = create_store("sqlite:///ctx.db")                     # default, single file
store = create_store("postgresql://user:pw@host/longloom")
store = create_store("mysql://user:pw@host/longloom")
store = create_store("mongodb://host:27017/longloom")
store = create_store("https://myteam.supabase.co")           # + SUPABASE_KEY
store = create_store("json://data")                          # human-readable JSONL
store = create_store("memory://")                            # ephemeral
```

| Backend | Extra | Notes |
|---|---|---|
| In-memory | — | zero-dependency default |
| SQLite | — (stdlib) | inline content or file-backed |
| JSONL files | — | one readable file per session |
| **PostgreSQL** | `[postgres]` | SQLAlchemy or psycopg2/psycopg |
| **MySQL / MariaDB** | `[mysql]` | SQLAlchemy or PyMySQL |
| **MongoDB** | `[mongodb]` | server-side regex search, `$group` stats |
| **Supabase** | `[supabase]` | REST via supabase-py — no DB connection needed |
| **Your own** | — | subclass `BaseContextStore` — implement four methods |

All backends pass the identical contract suite (round-trip, filters, budget,
upsert idempotency, search, stats, delete).

## Built for production

| | |
|---|---|
| **Hierarchical merge** | Verbatim runs beyond one window still complete — merge calls are grouped and combined recursively (verified: a 45K-token extraction survives an 8K window). |
| **Retry + pacing** | Exponential backoff for 429/5xx/timeouts, optional min-interval between calls for long matrix runs. |
| **Bounded repair** | Oversized results condense with a capped attempt count — no infinite loops, no silent truncation. |
| **Token-count guardrails** | tiktoken exact or zero-dep estimator; chunk sizes clamped so map results are always combinable. |
| **Typed & versioned** | `py.typed`, semantic versioning, CHANGELOG, per-provider extras. |

### Honest limits

- Map/merge **aggregates**; it doesn't reason globally — cross-chunk conflicts
  (two versions of the same figure) are preserved side by side, not reconciled.
- Final results don't cite source chunks (the store does — provenance output is
  on the roadmap).
- The map phase is sequential; parallelize externally for very large corpora.
- tiktoken approximates non-OpenAI tokenizers — keep 2× headroom on chunk sizes.

## Examples — every claim is a runnable script

```bash
git clone https://github.com/sumitnemade/longloom && cd longloom
pip install -e ".[dev]"
export NVIDIA_API_KEY=...        # or OPENAI_API_KEY=...
python examples/run_all.py       # 16 examples; skips gracefully what's missing
```

| # | Example | One-liner |
|---|---|---|
| 01 | quickstart | minimal pipeline, zero setup |
| 02 | all storage backends | every store via one URL |
| 03–06 | NVIDIA · OpenAI · Anthropic · local Ollama | same workflow, four vendors |
| 07 | custom callable | a bare function as the LLM (routing pattern) |
| 08–11 | PostgreSQL · MySQL · MongoDB · Supabase | one example per database |
| 12 | custom backend | a working Redis store in ~40 lines |
| 13 | advanced | custom prompts, token counters, brace-safety |
| 14 | **NVIDIA × every database** | real Nemotron over 8 backends |
| 15 | **OpenAI × every database** | real GPT over 8 backends |
| 16 | **no-loss extraction** | 41/41 facts on a hostile document mix |

## Test it yourself

```bash
pip install -e ".[dev]"
pytest tests/ -v                       # 41 tests; network-dependent ones self-skip
NVIDIA_API_KEY=... pytest tests/ -v    # + real NVIDIA, Postgres, MySQL, MongoDB
OPENAI_API_KEY=... pytest tests/ -v    # + real OpenAI end-to-end
```

## FAQ

**Do I need LangChain?** Only `langchain-core` (a lightweight dependency).
Your LLM can even be a plain function.

**What if my output is bigger than the window?** The repair loop condenses it,
and the merge stage itself groups results hierarchically — oversized inputs are
the library's job, not yours.

**Is my data sent anywhere?** Only to the LLM provider you configure. Storage
stays in whatever database *you* chose.

**Why not just use a 200K-token model?** Cost, latency, and attention decay.
Chunked extraction is cheaper per token and verifiably complete — and LongLoom
works with *any* window, so you're never locked to a model generation.

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and
conventions (every fix ships with a test; honest verification is the house
style). By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) © Sumit Nemade