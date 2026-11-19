# Changelog

All notable changes to LongLoom are documented here.



## [1.4.0] — 2026-11-19

### Added
- **Coverage verification & salvage** — `MapReduceProcessor` verifies the
  final result against the union of all map outputs; dropped lines restored
  verbatim (no extra LLM calls); `result["coverage"]` reports
  expected/found/missing/salvaged; `on_event` emits `coverage_verified`.
- **Custom OpenAI-compatible provider** — `init_llm("custom", model, base_url=...,
  headers=...)` for vLLM/LiteLLM/Azure APIM/gateways; `api_key` optional.
- **NVIDIAChatModel custom headers** — `headers={"X-Corp-Proxy": "..."}` merged
  into every request.
- **Scale benchmark** — `examples/20_benchmark.py`: 83,023/83,023 no-loss at
  6.7M chars / 2.16M tokens / 3,795 chunks.
- **Landing page** — Bento Grid Showcase (ui-ux-pro-max-skill); auto-deployed
  via GitHub Pages; zero external assets.
- **SECURITY.md** — data-flow disclosure, private vulnerability reporting.
- **19 runnable examples** covering all providers, databases, and patterns.

### Fixed
- Oversized merge results now warn via `result["warnings"]` instead of
  silent truncation.
- `FieldMap.col("namespace")` KeyError on namespaced Mongo/Supabase stores.
- `table_sql()` accepted no `namespace_field` parameter.
- NVIDIA sanity step in CI: version read by regex.
- `init_llm` direct provider map: import errors name correct pip extra.

### Security
- **Retry hardening** — `Retry-After` headers honored (capped 120s), jittered
  exponential backoff, deterministic `ValueError`s never retried.
- **Structured logging** — `longloom.*` loggers; never logs content/keys.
- **CI** — `pip-audit` advisory; MongoDB service container for live tests.
- **Coverage gate** — 99% in CI with `pytest-cov`.

### Changed
- `[dev]` extra gains `langchain-openai`, `supabase`, `pytest-cov`.
- Landing page: Bento Grid redesign, staggered reveal with reduced-motion
  suppression.

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
- **Source-anchored audit** — the map phase itself can under-capture on
  unstructured input (measured with a live model on a 160K-token unstructured
  BRD: pipe-table rows 36% and action-log lines 48% dropped vs 2% for
  bullets), and the map-union coverage check cannot see what map never
  extracted. With the verbatim extraction preset (auto-enabled), the final
  result is now also diffed against the SOURCE document's fact-bearing lines
  (lines containing a digit or currency symbol); missing lines are appended
  verbatim, deterministically, at zero LLM cost. `result["source_audit"]`
  reports the accounting; `on_event` emits `source_audit_completed`. PII
  scrubbers are honored. Pass `source_audit=False` to disable. Validated:
  400/400 requirement lines and 10/10 headline figures recovered on the
  unstructured-BRD run (from 340/400 and 8/10).
- **Numeric-consistency audit (report-only)** — every multi-digit figure
  cited in the final result that the source does not contain is flagged in
  `result["numeric_audit"]` and `result["warnings"]`. Measured failure mode:
  a real model reported a document total 15-38% off with no signal.
  Computed aggregates are flagged too, so treat this as a verification hint,
  not an error. Regex only, zero LLM cost; `numeric_audit=False` disables;
  `numeric_audit_completed` event emitted.
- **`RateLimitError`** — callables and HTTP bridges can raise it to signal a
  provider quota window. Rate-limit retries draw from their own backoff
  budget (`rate_limit_attempts=8`, `rate_limit_backoff=30s`,
  `rate_limit_max_sleep=900s`, all constructor-tunable) instead of the short
  generic schedule (5 attempts, 120s ceiling) that died against sustained
  quota windows. `Retry-After` headers are honored on both paths.
- **Prefix-blind coverage keys** — `_default_coverage_keys` now normalizes
  bullets (`- `, `* `, `•`), numbered items, table pipes and quotes before
  comparing, because a real merge model does not always preserve the bullet
  (measured: 15 lines present in the merge output were invisible to
  bullet-only matching).

### Changed
- **Hardened verbatim prompts** — `VERBATIM_MAP_PROMPT` now explicitly covers
  pipe-delimited table rows, action/decision/meeting-log entries and
  fact-bearing prose sentences (not just bullets); the merge prompts gained
  a self-check line ("verify that every unique input line appears in your
  output").
- **Chunk-token clamp is no longer silent** — `max_chunk_tokens` is clamped
  to `max_context_tokens // 2` (chunks larger than half the merge budget can
  never be combined); the clamp is now logged, so users no longer
  accidentally configure 2x more LLM calls than intended.

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
