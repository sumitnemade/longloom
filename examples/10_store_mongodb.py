#!/usr/bin/env python3
"""10 — MongoDB storage backend (real server).

Requires a MongoDB server and pymongo:
    pip install 'longloom[mongodb]'

    export LONGCTX_MONGO_URI="mongodb://host:27017"
    python examples/10_store_mongodb.py

Runs out of the box against a local default MongoDB (127.0.0.1:27017).
Server-side regex search and $group aggregation are used.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

from _common import DemoLLM, SAMPLE_DOC, guard, show

from longloom import MapReduceProcessor, create_store


def main() -> int:
    uri = os.environ.get("LONGCTX_MONGO_URI", "mongodb://127.0.0.1:27017")
    try:
        store = create_store(f"{uri}/longloom_examples")
        store.get_session_info("probe")
    except Exception as exc:
        guard(f"MongoDB reachable at {uri} (install driver: pip install "
              f"'longloom[mongodb]') — {type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:100]}", False)

    processor = MapReduceProcessor(
        llm=DemoLLM(), store=store,
        max_chunk_tokens=600, max_context_tokens=8192,
    )
    result = processor.process(SAMPLE_DOC, task="Summarize key themes",
                               session_id="mongo-demo")
    show(f"MongoDB ({uri})", result, processor)

    info = processor.get_session_info("mongo-demo")
    print(f"  session info (from $group aggregation): {info}")
    processor.delete_session("mongo-demo")
    store.client.drop_database(store.collection.database.name)
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())