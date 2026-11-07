# Contributing to LongLoom

Thanks for helping weave. This guide gets you productive in five minutes.

## Development setup

```bash
git clone https://github.com/sumitnemade/longloom && cd longloom
pip install -e ".[dev]"          # pytest + tiktoken + DB drivers
pytest tests/ -v                 # should pass with zero network access
```

Network-dependent tests (real LLM APIs, real databases) **self-skip** without
credentials. To run the full verification matrix locally:

```bash
export NVIDIA_API_KEY=...        # or OPENAI_API_KEY=...
pytest tests/ -v                 # + real NVIDIA/OpenAI, Postgres, MySQL, MongoDB
```

A running local stack makes even more tests go live: MongoDB (`:27017`),
Redis (`:6379`), PostgreSQL/MySQL via `LONGCTX_PG_DSN` / `LONGCTX_MYSQL_DSN`.

## House rules

1. **Every fix ships with a test.** The test suite is the library's honesty
   contract — regression tests are written *before* the fix when possible
   (see `tests/test_production_hardening.py` for the pattern).
2. **Verify claims on real systems.** Performance and preservation numbers in
   the README must be reproducible by an example or a test. No aspirational
   benchmarks.
3. **Provider-agnostic by design.** New vendor integrations go through
   `init_llm`/`LLMAdapter` seams, never hardcoded into `graph.py`.
   New storage goes behind `BaseContextStore` (implement the four primitives;
   the contract suite in `tests/store_contract.py` must pass unchanged).
4. **Honest limits over marketing.** If a feature has caveats, document them
   in the README's limits section and in the docstring.
5. **Compatibility**: Python ≥ 3.10; core dependencies stay at two
   (`langgraph`, `langchain-core`) — everything else is an extra.

## Pull requests

- One logical change per PR; rebase, don't merge.
- Add a `CHANGELOG.md` entry under *Unreleased*.
- CI runs the offline suite on Python 3.10–3.13 plus a package build —
  green CI is required. Real-API checks run locally with your keys.
- Type annotations required for new public APIs (`py.typed` is shipped).

## Releasing (maintainers)

```bash
scripts/release.sh 1.2.0 --push
# runs tests, bumps __version__, moves the CHANGELOG, tags v1.2.0 —
# pushing the tag triggers .github/workflows/release.yml → PyPI (trusted publishing)
```

## Where to start

Good first issues: docs improvements, new `create_store` schemes, additional
provider quirks in `init_llm`, extra contract-suite backends.