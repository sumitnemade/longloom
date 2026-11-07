# 🧶 LongLoom

**Weave documents of any length through LLM map/merge/repair — without losing a thread.**

LongLoom takes documents far larger than any model's context window and turns them
into summaries, extracted fact lists, and structured specs — over **any LLM** and **any storage**, with **measured** (not assumed)
information preservation.

```text
                ┌──────────────┐
  500-page doc ─►  CHUNK+STORE ─► every byte persisted (SQLite / Postgres / Mongo / …)
                └──────┬───────┘
                       ▼        MAP   one LLM call per chunk, verbatim or summarizing
                ┌──────────────┐
                │ map results  │
                └──────┬───────┘
                       ▼        MERGE window-sized hierarchical combine calls
                ┌──────────────┐        (scales past any single context window)
                │ merged draft │
                └──────┬───────┘
                       ▼        REPAIR bounded condensation loop, if still too big
                ┌──────────────┐
                │ final output │  +  verification: every fact accounted for
                └──────────────┘
```

---

## Why LongLoom

| | |
|---|---|
| 🔌 **Any LLM. Any storage.** | One dependency seam each. LangChain chat models, 8 provider helpers, or a bare `str → str` callable. Memory → SQLite → JSONL → **PostgreSQL · MySQL · MongoDB · Supabase** → or subclass one ABC for your own (Redis ships as a 40-line example). |
| 🧵 **Measured no-loss mode** | `PromptTemplates.for_extraction()` copies fact-bearing lines verbatim instead of summarizing. **Measured result: 100% fact preservation (41/41) on messy unstructured documents — stable across repeated real-LLM runs on NVIDIA *and* OpenAI.** Summarizing prompts measurably drop 2–10% of facts; the verbatim preset eliminates it. |
| 📈 **Scales past any window** | Hierarchical merge combines map results in window-sized groups recursively — a **45k-token verbatim run survives an 8k window** (unit-tested against a window-enforcing fake). |
| 🔁 **Bounded repair loop** | Outputs that still exceed the budget are condensed with a capped attempt count — never an infinite loop, never silent truncation. |
| ✅ **Verify, don't trust** | Fact-preservation reports, lossless-chunking assertions, and a window-enforcing fake-LLM suite are first-class. **42 tests, including real-API and real-database integrations.** |
| 🧠 **Provider-native details handled** | NVIDIA reasoning models: real model IDs, a `thinking` toggle, and the `content=None` starvation guard. OpenAI quirks (`extra_body`, temperature restrictions) abstracted away. |
| 🚢 **Deploy-ready** | Typed package (`py.typed`), dynamic versioning, per-provider pip extras, retry with exponential backoff, XSS-hardened artifact viewers, MIT licensed. |

**Verified against real systems** — not mocks: 2 providers × 8 storage backends
(all green), real NVIDIA Nemotron and OpenAI GPT end-to-end with SQLite
persistence and reopen.

---

## Install

```bash
pip install longloom                    # core: langgraph + langchain-core only
pip install "longloom[tiktoken]"        # exact token counting (recommended)
pip install "longloom[databases]"       # + Postgres / MySQL / MongoDB / Supabase
pip install "longloom[all]"             # everything
```

Python ≥ 3.10. NVIDIA key: https://integrate.api.nvidia.com/

## 60-second start

```python
from longloom import (
    MapReduceProcessor, PromptTemplates, SQLiteContextStore, init_llm,
)

processor = MapReduceProcessor(
    llm=init_llm("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", thinking=False),
    store=SQLiteContextStore("ctx.db"),
    max_chunk_tokens=3000,
    max_context_tokens=8192,                       # the model's real window
    prompts=PromptTemplates.for_extraction(),      # no-loss verbatim mode
)

result = processor.process(huge_document, task="Summarize key themes")
print(result["final_result"])                      # verified fact-complete

# everything is persisted — retrieve, search, or delete later
processor.search_session(result["session_id"], "AlphaFold")
```

Swap providers or databases without touching the workflow:

```python
llm   = init_llm("openai", "gpt-4o-mini")            # or anthropic, groq, ollama…
llm   = lambda prompt: my_gateway(prompt)            # even a bare function
store = create_store("postgresql://user:pw@host/longloom")
store = create_store("mongodb://host:27017/longloom")
store = create_store("https://myteam.supabase.co")   # + SUPABASE_KEY
```

## Storage backends

| Backend | Extra | Connection string |
|---|---|---|
| In-memory | — (default) | `memory://` |
| JSONL files | — | `json://<dir>` |
| SQLite | — (stdlib) | `sqlite:///path.db` |
| **PostgreSQL** | `[postgres]` | `postgresql://user:pw@host/db` |
| **MySQL / MariaDB** | `[mysql]` | `mysql://user:pw@host/db` |
| **MongoDB** | `[mongodb]` | `mongodb://host:27017/db` |
| **Supabase** | `[supabase]` | `https://<proj>.supabase.co` + `SUPABASE_KEY` |
| **Your own** | — | subclass `BaseContextStore` — four methods |

SQL URLs prefer SQLAlchemy (pooling, every driver) and fall back to native
drivers. Other DB-API databases work with a custom `SqlDialect`.

## LLM providers

`init_llm("nvidia" | "openai" | "anthropic" | "groq" | "ollama" | "mistral" |
"deepseek" | "together", model)` maps straight to the provider's LangChain
integration (no meta-package needed), or pass any `BaseChatModel` / callable.

## The three prompts you can own

| Preset | Behaviour |
|---|---|
| `DEFAULT_PROMPTS` | summarizing map/merge — best for readable reports |
| `PromptTemplates.for_extraction()` | verbatim line-copy — best when **no fact may be lost** |
| your own `PromptTemplates(...)` | full control; rendering is brace-safe (JSON/logs in content can't break it) |

## Production hardening (built in, tested)

- **Hierarchical merge** — verbatim runs beyond a single window still complete
- **Retry + exponential backoff** — transient 429/503/timeout proof
- **Chunk-size guard** — map results are always combinable
- **XSS-safe artifact viewers** — untrusted documents can't inject script
- Honest known-limits list (cross-chunk conflict reasoning, provenance,
  sequential map phase, tokenizer approximation) — documented, not hidden

## Test it yourself

```bash
pip install -e ".[dev]"
pytest tests/ -v                              # 42 tests; real-API/DB ones self-skip
NVIDIA_API_KEY=... pytest tests/ -v           # + real NVIDIA + real Postgres/MySQL/Mongo
OPENAI_API_KEY=... pytest tests/ -v           # + real OpenAI

python examples/run_all.py                    # 17 runnable examples, full matrix
```

Highlights from `examples/`:

| # | Example | What it proves |
|---|---|---|
| 14 | NVIDIA × every database | real Nemotron over 8 backends, 0 failures |
| 15 | OpenAI × every database | real GPT over 8 backends, 0 failures |
| 16 | unstructured no-loss | 41/41 planted facts, stable across runs |


## Package layout

```
longloom/
├── processor.py        # MapReduceProcessor — the one entry point
├── graph.py            # LangGraph chunk→map→merge→repair (+ hierarchical merge)
├── llm.py              # LLMAdapter (retry) + init_llm() provider map
├── nvidia.py           # NVIDIAChatModel — reasoning-aware, requests-based
├── prompts.py          # brace-safe templates + verbatim extraction preset
├── chunks.py           # token-aware splitting, any counter
├── tokens.py           # TiktokenCounter / EstimateCounter
├── factory.py          # create_store(url) for every backend
└── stores/             # memory · sqlite · jsonl · sql(pg,mysql) · mongo · supabase
```

## License

MIT — see [LICENSE](LICENSE). Changelog: [CHANGELOG.md](CHANGELOG.md).