#!/usr/bin/env python3
"""09 — MySQL / MariaDB storage backend (real server).

Requires a MySQL server and either SQLAlchemy or PyMySQL installed:
    pip install 'longloom[mysql]'

    export LONGCTX_MYSQL_DSN="mysql://user:password@host:3306/longloom"
    python examples/09_store_mysql.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

from _common import DemoLLM, SAMPLE_DOC, guard, show

from longloom import MapReduceProcessor, create_store


def main() -> int:
    candidates = [os.environ["LONGCTX_MYSQL_DSN"]] if os.environ.get("LONGCTX_MYSQL_DSN") else [
        "mysql://longloom:longloom_test@127.0.0.1:3307/longloom",  # user-space instance
        "mysql://root@127.0.0.1/mysql",
    ]
    store, dsn = None, None
    for candidate in candidates:
        try:
            store = create_store(candidate)
            store.get_session_info("probe")
            dsn = candidate
            break
        except Exception:
            continue
    if store is None:
        guard("MySQL reachable (set LONGCTX_MYSQL_DSN; install driver: "
              "pip install 'longloom[mysql]')", False)

    processor = MapReduceProcessor(
        llm=DemoLLM(), store=store,
        max_chunk_tokens=600, max_context_tokens=8192,
    )
    result = processor.process(SAMPLE_DOC, task="Summarize key themes",
                               session_id="mysql-demo")
    show(f"MySQL ({dsn.split('@')[-1]})", result, processor)
    processor.delete_session("mysql-demo")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())