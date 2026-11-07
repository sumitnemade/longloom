# Changelog

All notable changes to LongLoom are documented here.

## [Unreleased]

### Added
- **Custom OpenAI-compatible provider** — `init_llm("custom", model, base_url=..., headers=...)`
  for vLLM/LiteLLM/gateways; `api_key` optional; wire-level tests assert the
  custom base_url and headers arrive on every request. `NVIDIAChatModel` also
  accepts extra `headers=`.

## [1.1.0] — 2026-11-07

### Security & operations
- **Retry hardening** — provider `Retry-After` headers honored (capped at
  120s), jittered exponential backoff to decorrelate parallel workers,
  deterministic `ValueError`s never retried.
- **Structured logging** — `longloom.*` stdlib loggers emit run lifecycle
  (INFO) and retry diagnostics (WARNING); prompts, completions and
  credentials are never logged (regression-tested).
- **SECURITY.md** — data-flow disclosure, reporting process, security
  features, supply-chain notes.
- **CI** — `pip-audit` advisory job against the installed dependency set.

### Added
- **PyPI live** — `longloom 1.0.0` published via Trusted Publishing;
  `pip install longloom` verified from a clean venv
- **Automated release pipeline** — `scripts/release.sh` (tests, version bump,
  changelog, tag) and a tag-triggered GitHub workflow publishing to PyPI via
  Trusted Publishing (OIDC); `twine check` pre-flight passes.
- **Explicit credentials** — API keys and Supabase url/key are constructor
  arguments only; the library never reads `os.environ` for secrets. Keys are
  required for every hosted provider (`ollama` exempt).
- **Embedding in existing databases** — every SQL/MongoDB/Supabase backend now
  accepts `columns` (logical → physical field names), `table` overrides, and a
  `namespace` discriminator (+ `namespace_field`) so multiple applications can
  share one table/collection safely; `init_schema=False` for user-owned
  migrations; `SqlContextStore.ddl()` and `table_sql(...)` generate
  schema-matching DDL. SQLite store rebased onto the generic SQL backend
  (feature parity). Validated by `tests/test_embed_existing.py` and
  `examples/17_embed_existing_database.py`.

## [1.0.0] — 2026-11-06

First production release.

### Core
- LangGraph chunk → map → merge → repair workflow with full dependency
  injection: any LLM (LangChain `BaseChatModel`, provider helper, or plain
  callable) × any storage (`BaseContextStore` implementations).
- Storage backends: in-memory, SQLite (inline or file-backed), JSONL,
  PostgreSQL, MySQL/MariaDB (generic DB-API + dialects), MongoDB, Supabase
  (REST via supabase-py); URL factory `create_store("postgresql://…")`.
- Provider helper `init_llm()` with direct integration imports (openai,
  anthropic, nvidia, groq, ollama, mistral, deepseek, together) — no
  `langchain` meta-package required.
- NVIDIA reasoning-model support: real model IDs, `thinking` toggle,
  `content=None` guard for reasoning-budget starvation.
- Pluggable token counters (tiktoken exact / zero-dep estimator).

### No-loss & scale
- `PromptTemplates.for_extraction()` — verbatim extraction preset; measured
  100% fact preservation on unstructured docs, stable across repeated
  real-LLM runs (NVIDIA + OpenAI).
- Hierarchical merge — window-sized recursive combine calls; survives
  verbatim runs far beyond a single model window (verified at 45k+ tokens
  against an 8k window).
- Retry with exponential backoff in `LLMAdapter` for transient API failures.

### Verified (real systems, not mocks)
- 2 real providers × 8 storage backends matrix — all green.
- Real NVIDIA + OpenAI end-to-end with SQLite persistence + reopen.

### Packaging
- Typed package (`py.typed`), dynamic version, per-provider extras, MIT license.
