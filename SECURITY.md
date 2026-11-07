# Security Policy

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** (Repository → Security →
Report a vulnerability). Please do **not** open a public issue for security
reports. Reports are acknowledged within 7 days; fixes ship in the next
patch/minor release.

## Data flow — what LongLoom touches

Understanding this is the core of any security review:

| Data | Where it goes |
|---|---|
| Document content (possibly sensitive) | **Only** to the LLM endpoint *you* configure, inside the prompts the pipeline builds |
| Chunk text + metadata | **Only** to the storage backend *you* configure (your databases, your tenancy rules) |
| Progress events | **Only** to the `on_event` callback *you* supply (if any) |
| API keys / credentials | Held in memory for the process lifetime; **never** logged, never read from the environment, never persisted |

LongLoom makes no other network calls: no telemetry, no phone-home, no
license checks, no update pings.

## Security features

- **Explicit credentials** — `api_key=` is a constructor argument; the library
  contains zero `os.environ` reads for secrets (verified by test).
- **No content in logs** — the library logs event names, durations and token
  counts; prompts and completions are never logged.
- **Injection-validated identifiers** — table/collection/column names supplied
  via `FieldMap` are regex-validated (SQL identifier grammar) before any
  query is built.
- **XSS-hardened viewers** — generated HTML artifacts HTML-escape document
  content and disable mermaid `htmlLabels`.
- **Bounded operations** — retry sleeps are capped; repair loops are
  attempt-limited; the map phase cannot loop.

## Supply chain

- Core dependencies: `langgraph` + `langchain-core` only; everything else is
  an opt-in extra.
- Every release artifact is built by CI from a tagged commit and attached to
  the GitHub Release; releases publish via OpenID Connect (Trusted
  Publishing) — no long-lived tokens exist.
- CI runs `pip-audit` against the installed dependency set.

## Known considerations (by design, documented)

- Document content transits to the configured LLM provider — use a gateway or
  self-hosted endpoint (`init_llm("custom", base_url=...)`) when data must not
  leave your perimeter.
- Scrubbing PII is pluggable (`scrubber=`) but not automatic — enterprises
  should integrate their own detector (e.g. Presidio) at that seam.
- `sqlite`/`jsonfile`/`memory` stores write plaintext by design; encrypt at
  the volume level or point at your managed database for at-rest encryption.
