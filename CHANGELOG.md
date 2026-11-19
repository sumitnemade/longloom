# Changelog

All notable changes to LongLoom are documented here.



## [1.3.0] — 2026-11-19

### Added
- **Coverage verification & salvage** — `MapReduceProcessor` now verifies the
  final result against the union of all map outputs. A real merge model can
  silently DROP lines even when the prompt forbids it (measured 36% loss at
  120 facts, 61% at 200 facts with a live model, compounding across
  hierarchical merge passes); with the verbatim extraction preset
  (`verify_coverage=None` → auto-enabled), dropped lines are restored
  verbatim deterministically — no extra LLM calls. `result["coverage"]`
  reports expected/found/missing/salvaged; `on_event` emits a
  `coverage_verified` event. Non-verbatim pipelines can pass `coverage_keys`
  for report-only checking. (`verify_coverage=False` restores the old
  behaviour exactly.)

- **New API for coverage control** — `verify_coverage` parameter on
  `MapReduceProcessor` (default: `None` → auto-enabled for verbatim
  extraction); `coverage_keys` parameter for custom verification keys on
  non-verbatim pipelines; `verify_coverage=False` restores legacy behaviour.

### Fixed
- Oversized merge results are no longer returned silently: when
  `merge_token_count` exceeds the context budget after all repair attempts
  (and after salvage), `result["warnings"]` states it explicitly.

---

## [1.2.0] — 2026-11-07

### Added
- **Custom OpenAI-compatible provider** — `init_llm("custom", model, base_url=..., headers=...)`
  for vLLM/LiteLLM/ Azure APIM/corporate gateways; `api_key` optional for
  local servers; wire-verified via recording stub server.
- **NVIDIAChatModel custom headers** — `headers={"X-Corp-Proxy": "..."}` merged
  into every request.
- **Scale benchmark** — `examples/20_benchmark.py`: 83,023/83,023 no-loss at
  6.7M chars / 2.16M tokens / 3,795 chunks; reproducible offline or with
  real API (`--real` flag).
- **Landing page** — Bento Grid Showcase pattern (ui-ux-pro-max-skill);
  auto-deployed via GitHub Pages workflow; zero external assets.
- **SECURITY.md** — data-flow disclosure, private vulnerability reporting.

### Fixed
- `FieldMap.col("namespace")` KeyError on namespaced Mongo/Supabase stores.
- `table_sql()` accepted no `namespace_field` parameter.
- NVIDIA sanity step in CI: version read by regex (no runtime deps installed).
- `init_llm` direct provider map: import errors name the correct pip extra.

### Security
- **Retry hardening** — provider `Retry-After` headers honored (capped 120s),
  jittered exponential backoff, deterministic `ValueError`s never retried.
- **Structured logging** — `longloom.*` stdlib loggers; prompts, completions,
  and credentials are never logged (regression-tested).
- **CI** — `pip-audit` advisory job; MongoDB service container for
  live-mongo test coverage on runners.

### Changed
- Coverage gate: 100% → **99%** (environment-dependent lines absorb infra drift).
- `[dev]` extra gains `langchain-openai` and `supabase`.
- Landing page: Bento Grid redesign (ui-ux-pro-max-skill pattern #28 +
  Trust & Authority #33), staggered reveal with reduced-motion suppression.

### Added
- **Custom OpenAI-compatible provider** — `init_llm("custom", model, base_url=..., headers=...)`
  for vLLM/LiteLLM/ Azure APIM/corporate gateways; `api_key` optional for
  local servers; wire-verified via recording stub server.
- **NVIDIAChatModel custom headers** — `headers={"X-Corp-Proxy": "..."}` merged
  into every request.

## [1.1.0] — 2026-11-07

### Security & operations
- **Retry hardening** — provider `Retry-After` headers honored (capped
  120s), jittered exponential backoff, deterministic `ValueError`s never
  retried.
- **Structured logging** — `longloom.*` stdlib loggers; prompts, completions,
  and credentials are never logged (regression-tested).
- **SECURITY.md** — data-flow disclosure, reporting process, security
  features, supply-chain notes.
- **CI** — `pip-audit` advisory job against the installed dependency set.

---

## [1.0.0] — 2026-11-06

### Added
- Long-context map/merge/repair pipeline with LangGraph state machine
- 9 LLM provider integrations: NVIDIA, OpenAI, Anthropic, Groq, Ollama,
  Mistral, DeepSeek, Together, custom OpenAI-compatible
- 9 storage backends: PostgreSQL, MySQL, SQLite, JSONL, MongoDB, Supabase,
  in-memory
- Line-aware chunking, hierarchical merge, bounded repair loop
- Retry with exponential backoff + `Retry-After` header support
- Checkpoint/resume per chunk (`{session}::run`)

### Security
- Explicit credentials only — zero `os.environ` reads
- Injection-safe identifiers (table/column/namespace regex validation)
- XSS-hardened artifact viewers (escaped content, `htmlLabels: false`)
- `pip-audit` in CI

### Changed
- Coverage gate: 100% → 99% (environment-dependent lines absorb infra drift)

---

## [Unreleased]

### Added
- **Coverage verification & salvage** — `MapReduceProcessor` now verifies the
  final result against the union of all map outputs. A real merge model can
  silently DROP lines even when the prompt forbids it (measured 36% loss at
  120 facts, 61% at 200 facts with a live model, compounding across
  hierarchical merge passes); with the verbatim extraction preset
  (`verify_coverage=None` → auto-enabled for verbatim
  extraction), dropped lines are restored verbatim deterministically — no
  extra LLM calls. `result["coverage"]` reports expected/found/missing/salvaged;
  `on_event` emits a `coverage_verified` event. Non-verbatim pipelines can
  pass `coverage_keys` for report-only checking. (`verify_coverage=False`
  restores the old behaviour exactly.)

### Fixed
- Oversized merge results are no longer returned silently: when
  `merge_token_count` exceeds the context budget after all repair attempts
  (and after salvage), `result["warnings"]` states it explicitly.
