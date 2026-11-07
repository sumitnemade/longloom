# longloom Examples

Every LLM provider and every storage backend, one focused example each.
Run any of them directly, or all at once:

```bash
python examples/run_all.py
```

Examples auto-detect their requirements and print `SKIPPED: <reason>` when a
key/server isn't available — so the whole suite is always safe to run.

## The matrix

| # | Example | LLM | Storage | Needs | Status here |
|---|---------|-----|---------|-------|-------------|
| 01 | [quickstart](01_quickstart.py) | demo | memory | nothing | ✅ runs |
| 02 | [all storage backends](02_all_storage_backends.py) | demo | **every** store via URL | servers for SQL ones | ✅ runs (skips unconfigured) |
| 03 | [NVIDIA](03_llm_nvidia.py) | `nvidia/nemotron-3-ultra` | memory | `NVIDIA_API_KEY` | ⏳ needs key |
| 04 | [OpenAI](04_llm_openai.py) | `gpt-4o-mini` | memory | `OPENAI_API_KEY` | ✅ runs |
| 05 | [Anthropic](05_llm_anthropic.py) | `claude-sonnet-4` | memory | `ANTHROPIC_API_KEY` | ⏳ needs key |
| 06 | [Ollama local](06_llm_ollama_local.py) | local model via HTTP | memory | Ollama on :11434 | ⏳ needs Ollama |
| 07 | [custom callable](07_llm_custom_callable.py) | plain function / router | memory | nothing | ✅ runs |
| 08 | [PostgreSQL](08_store_postgres.py) | demo | PostgreSQL | `LONGCTX_PG_DSN` | ⏳ needs DSN |
| 09 | [MySQL](09_store_mysql.py) | demo | MySQL/MariaDB | `LONGCTX_MYSQL_DSN` | ⏳ needs DSN |
| 10 | [MongoDB](10_store_mongodb.py) | demo | MongoDB | local :27017 default | ✅ runs |
| 11 | [Supabase](11_store_supabase.py) | demo | Supabase (REST) | `SUPABASE_URL`+`SUPABASE_KEY` | ⏳ needs project |
| 12 | [custom backend (Redis)](12_store_redis_custom.py) | demo | **your own** store | local Redis :6379 | ✅ runs |
| 13 | [advanced customization](13_advanced_customization.py) | demo | memory | nothing (tiktoken optional) | ✅ runs |
| 14 | [NVIDIA × ALL databases](14_nvidia_all_databases.py) | **real NVIDIA** | **every** store | `NVIDIA_API_KEY` (+ servers) | ✅ runs |
| 15 | [OpenAI × ALL databases](15_openai_all_databases.py) | **real OpenAI** (gpt-4o-mini, low-context config) | **every** store | `OPENAI_API_KEY` (+ servers) | ✅ runs |
| 16 | [unstructured no-loss](16_unstructured_no_loss.py) | **real LLM** (NVIDIA or OpenAI) | SQLite | either API key | ✅ runs (100% preservation) |
| 17 | [embed in existing database](17_embed_existing_database.py) | demo LLM | **your** table/columns | nothing (SQLite) | ✅ runs |
| 18 | [enterprise gateway + existing DB](18_enterprise_gateway.py) | **real LLM** via callable | existing Postgres + tenant namespace | API key + a reachable DB | ✅ runs |


Example 16 feeds a messy document (logs, JSON, email thread, ASCII table,
config diffs, tickets) with 41 planted facts through the pipeline in
**verbatim extraction mode** (`PromptTemplates.for_extraction()`) and checks
every fact — 100% preservation, stable across repeated real-LLM runs.
Summarization prompts measurably drop facts (90–98% in trials); verbatim
copying plus keep-every-line merging eliminates the loss.

## What each demonstrates

- **01 quickstart** — the 10-line minimal pipeline.
- **02 all storage backends** — one LLM, every store via `create_store(url)`,
  repair path exercised, no residue left in shared databases.
- **03–06 LLM providers** — the identical workflow across NVIDIA, OpenAI,
  Anthropic, and a fully-local Ollama model (plain-HTTP callable, zero
  LangChain wrappers). Only the `llm=` argument ever changes.
- **07 custom callable** — the minimum LLM contract: a function. Includes a
  routing example (cheap model maps, big model merges).
- **08–11 databases** — PostgreSQL, MySQL, MongoDB, Supabase.
- **12 custom backend** — subclass `BaseContextStore`, implement four
  primitives, get the entire workflow on Redis.
- **13 advanced** — custom prompt templates (brace-safe), task steering,
  pluggable token counters, chunk retrieval for grounding.

## One-time setup for the skipped ones

```bash
export NVIDIA_API_KEY=...            # https://integrate.api.nvidia.com/
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
ollama serve && ollama pull llama3.1
export LONGCTX_PG_DSN="postgresql://user:pw@host/longloom"
export LONGCTX_MYSQL_DSN="mysql://user:pw@host/longloom"
export SUPABASE_URL="https://<proj>.supabase.co"
export SUPABASE_KEY="..."
pip install -e ".[all]"
python examples/run_all.py           # everything now runs
```