<div align="center">

# 🧶 LongLoom

**Weave documents of any length through LLM map/merge/repair — without losing a thread.**

[![PyPI](https://img.shields.io/pypi/v/longloom)](https://pypi.org/project/longloom/)
[![Python](https://img.shields.io/pypi/pyversions/longloom)](https://pypi.org/project/longloom/)
[![CI](https://github.com/sumitnemade/longloom/actions/workflows/ci.yml/badge.svg)](https://github.com/sumitnemade/longloom/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Typed](https://img.shields.io/badge/typing-typed-blue)](longloom/py.typed)
[![Landing page](https://img.shields.io/badge/docs-longloom.dev-blueviolet)](https://sumitnemade.github.io/longloom/)
[![Tests](https://img.shields.io/badge/tests-50%20passing-brightgreen)](#test-it-yourself)

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

From [PyPI](https://pypi.org/project/longloom/) — the usual way:

```bash
pip install longloom                 # core: 2 dependencies only
pip install "longloom[all]"          # batteries included (all providers + databases)
pip install "longloom[tiktoken,nvidia]"   # pick exactly what you need
```

Extras: `[tiktoken]` `[nvidia]` `[openai]` `[postgres]` `[mysql]` `[mongodb]` `[supabase]` `[databases]` `[dev]`

Python ≥ 3.10.
**Tested with**: Python 3.10–3.13 · `langgraph` 1.2.x · `langchain-core` 1.7.x ·
NVIDIA Nemotron-3-Ultra and OpenAI GPT-4o-mini (real-API test suite).

**From source** (contributors or pinned forks):

```bash
git clone https://github.com/sumitnemade/longloom && cd longloom
pip install -e ".[dev]"              # editable install + test tooling
pytest tests/ -v                     # 48 tests; network-dependent ones self-skip
```

## 60-second quickstart

```python
import os
from longloom import MapReduceProcessor, SQLiteContextStore, init_llm

processor = MapReduceProcessor(
    llm=init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b",
                 api_key=os.environ["NVIDIA_API_KEY"],   # your app owns the key
                 thinking=False),
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
init_llm("nvidia",    "nvidia/nemotron-3-ultra-550b-a55b",
         api_key=os.environ["NVIDIA_API_KEY"], thinking=False)
init_llm("openai",    "gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"])
init_llm("anthropic", "claude-sonnet-4-20250514", api_key=os.environ["ANTHROPIC_API_KEY"])
init_llm("groq" | "mistral" | "deepseek" | "together", ..., api_key=...)
init_llm("ollama", "llama3.1")                               # no key needed
```

> 🔐 **Keys are constructor arguments, never environment reads.** LongLoom
> itself never touches `os.environ` for credentials — your application reads
> the secret (vault, env, wherever) and passes it explicitly. Explicit beats
> implicit when credentials are involved.

…or bring your own: any LangChain `BaseChatModel`, or a bare function —
the minimum contract is `str → str`:

```python
processor = MapReduceProcessor(llm=lambda prompt: my_gateway(prompt), store=store)
```

### Custom / self-hosted endpoints (vLLM, LiteLLM, Azure APIM, gateways)

```python
llm = init_llm(
    "custom", "acme-internal-model",
    base_url="http://vllm.internal:8000/v1",
    headers={"X-Tenant": "acme", "X-Trace-Id": "tr-1"},   # custom headers on every call
    api_key="internal-token",      # optional — local servers may need none
)
```

Wire-verified: requests hit your `base_url` with your headers attached, and
usage metadata from the endpoint flows into the observability events.

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

## Embedding in existing databases

No dedicated database needed — LongLoom lives inside **your** tables with
**your** column names, next to your application's own data:

```python
store = create_store(
    "postgresql://app@host/mydb",        # the database you already run
    table="app_document_chunks",         # your table
    columns={                            # your column names
        "id": "chunk_pk", "content": "text_body",
        "metadata": "meta_json", "session_id": "sess",
    },
    namespace="support-agent",           # several apps share one table safely
    init_schema=False,                   # your migrations own the DDL
)
```

| Option | Meaning | Supported by |
|---|---|---|
| `table=` | your table/collection name | SQL, MongoDB, Supabase |
| `columns=` | logical → physical field mapping | SQL, MongoDB, Supabase |
| `namespace=` + `namespace_field=` | discriminator so multiple apps share one table | SQL, MongoDB, Supabase |
| `init_schema=False` | your migrations own the DDL | SQL backends |
| `SqlContextStore.ddl()` / `table_sql(...)` | generate DDL matching your names, for your migration tooling | SQL, Supabase |

Same table, two apps, zero cross-talk — namespaces isolate sessions even with
identical session IDs. See `examples/17_embed_existing_database.py`.

## Built for production

| | |
|---|---|
| **Hierarchical merge** | Verbatim runs beyond one window still complete — merge calls are grouped and combined recursively (verified: a 45K-token extraction survives an 8K window). |
| **Retry + pacing** | Exponential backoff for 429/5xx/timeouts, optional min-interval between calls for long matrix runs. |
| **Bounded repair** | Oversized results condense with a capped attempt count — no infinite loops, no silent truncation. |
| **Token-count guardrails** | tiktoken exact or zero-dep estimator; chunk sizes clamped so map results are always combinable. |
| **Parallel map** | `max_workers=8` runs the map phase on a thread pool with shared rate-limit pacing — measured 4× on an 8-chunk document. |
| **Checkpoint/resume** | Map results are checkpointed per chunk; `resume=True` re-runs only what never completed after a crash. |
| **Observability** | `on_event=` emits structured events (per chunk, per LLM call with token usage, per repair) — forward to OTel/Datadog as you like. |
| **Retention & PII scrubbing** | `scrubber=` rewrites content before anything is stored; `delete_sessions_older_than(days)` sweeps old sessions across every backend. |
| **Conflict detection** | `detect_conflicts=True` runs an advisory pass that flags contradictory values side by side. |
| **Typed & versioned** | `py.typed`, semantic versioning, CHANGELOG, per-provider extras. |

### Honest limits

- Map/merge **aggregates**; it doesn't reason globally — `detect_conflicts=True`
  flags contradictions, but reconciling them is a human decision.
- A single line larger than the chunk budget is still hard-split (at whitespace,
  so identifier tokens survive) — multi-line records can span chunk seams.
- Async support is thread-offloaded (`process_async`), not a native async graph.
- MSSQL/Oracle dialects are SQL-generation-tested; live-server verification
  is on the roadmap.
- tiktoken approximates non-OpenAI tokenizers — keep 2× headroom on chunk sizes.

## Examples — every claim is a runnable script

```bash
git clone https://github.com/sumitnemade/longloom && cd longloom
pip install -e ".[dev]"
export NVIDIA_API_KEY=...        # or OPENAI_API_KEY=...
python examples/run_all.py       # 17 examples; skips gracefully what's missing
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
| 17 | **embed in existing database** | your table, your columns, namespaces |
| 18 | **enterprise gateway** | corporate LLM gateway + existing DB + tenant namespace |

## Test it yourself

```bash
pip install -e ".[dev]"
pytest tests/ -v                       # 50 tests; network-dependent ones self-skip
NVIDIA_API_KEY=... pytest tests/ -v    # + real NVIDIA, Postgres, MySQL, MongoDB
OPENAI_API_KEY=... pytest tests/ -v    # + real OpenAI end-to-end
```

## FAQ

**Do I need LangChain?** Only `langchain-core` (a lightweight dependency).
Your LLM can even be a plain function.

**What if my output is bigger than the window?** The repair loop condenses it,
and the merge stage itself groups results hierarchically — oversized inputs are
the library's job, not yours.

**Can I run the map phase in parallel?** Yes — `max_workers=8` on the
processor. Pacing and retries are shared and lock-guarded across workers.

**A run crashed halfway — do I start over?** No: `process(..., resume=True)`
reuses the checkpointed map results and maps only the missing chunks.

**Is my data sent anywhere?** Only to the LLM provider you configure — no
telemetry, no phone-home. Document content goes to that endpoint inside
pipeline prompts; chunks live in whatever database *you* chose; keys live in
memory only (never logged, never read from the environment). See
[SECURITY.md](SECURITY.md) for the full data-flow review.

**How do I handle PII?** Pass `scrubber=my_redactor` — content is rewritten
before anything is stored or mapped. For retention,
`store.delete_sessions_older_than(days)` sweeps old sessions on every backend.

**Why not just use a 200K-token model?** Cost, latency, and attention decay.
Chunked extraction is cheaper per token and verifiably complete — and LongLoom
works with *any* window, so you're never locked to a model generation.

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for setup and
conventions (every fix ships with a test; honest verification is the house
style). By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) — © The LongLoom Maintainers