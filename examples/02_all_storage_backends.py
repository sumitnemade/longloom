#!/usr/bin/env python3
"""02 — One LLM, EVERY storage backend, via create_store() URLs.

Probes each backend; unreachable/unconfigured ones are skipped gracefully.
Currently-running local servers (MongoDB, Redis-less SQL, etc.) run for real.
"""
import uuid

from _common import DemoLLM, SAMPLE_DOC, show

from longloom import MapReduceProcessor, create_store

CANDIDATES = [
    ("memory", "memory://"),
    ("sqlite", "sqlite:///examples_ctx.db"),
    ("jsonl", "json://examples_ctx_data"),
    ("mongodb", "mongodb://127.0.0.1:27017/longloom_examples"),
    ("postgres", "postgresql://@/longloom_examples"),          # needs local trust auth
    ("mysql", "mysql://root@127.0.0.1/longloom_examples"),     # needs local grants
    ("supabase", "https://placeholder.supabase.co"),          # needs SUPABASE_KEY env
]


def main() -> int:
    ok = skipped = 0
    for name, url in CANDIDATES:
        try:
            store = create_store(url)
            store.get_session_info("probe")  # touch the backend
        except Exception as exc:
            print(f"[{name:>9}] SKIPPED — {type(exc).__name__}: {str(exc).splitlines()[0][:80]}")
            skipped += 1
            continue

        session_id = f"ex_{name}_{uuid.uuid4().hex[:8]}"
        processor = MapReduceProcessor(
            llm=DemoLLM(), store=store,
            max_chunk_tokens=600, max_context_tokens=300,  # tiny window → exercises repair
        )
        try:
            result = processor.process(SAMPLE_DOC, task="Summarize key themes",
                                       session_id=session_id)
            show(f"store={name}", result, processor)
            assert result["repair_attempts"] >= 1, "repair should have triggered"
            ok += 1
        finally:
            processor.delete_session(session_id)
            # Leave no residue in shared backends.
            if hasattr(store, "client"):                      # MongoDB
                store.client.drop_database(store.collection.database.name)
            if hasattr(store, "close"):
                store.close()

    print(f"\n{ok} backend(s) verified live, {skipped} skipped (unconfigured/unreachable).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())