#!/usr/bin/env python3
"""19 — Expose LongLoom as an HTTP microservice (Java/Spark pipelines).

For stacks that aren't Python: wrap the library behind a tiny FastAPI
service. Keys are passed at app construction (this app reads them from the
environment at startup — the library itself never does).

    pip install "longloom[all]" fastapi uvicorn
    export NVIDIA_API_KEY=...
    uvicorn examples.19_http_service:app --port 8300   # if run as a module
    python examples/19_http_service.py                 # direct run

    curl -X POST http://127.0.0.1:8300/process \\
         -H "Content-Type: application/json" \\
         -d '{"content": "<your very long document>", "task": "Summarize"}'
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    print("SKIPPED: pip install 'longloom[all]' fastapi uvicorn")
    raise SystemExit(3)

from longloom import MapReduceProcessor, SQLiteContextStore, init_llm

# ── app wiring (constructed once, at startup) ────────────────────────────
if os.environ.get("OPENAI_API_KEY"):
    _llm = init_llm("openai", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    api_key=os.environ["OPENAI_API_KEY"], timeout=180)
elif os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY"):
    _llm = init_llm("nvidia", os.environ.get(
        "NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        api_key=os.environ.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_NIM_API_KEY"),
        temperature=0.2, max_tokens=1024, thinking=False)
else:
    print("SKIPPED: set NVIDIA_API_KEY or OPENAI_API_KEY")
    raise SystemExit(3)

processor = MapReduceProcessor(
    llm=_llm,
    store=SQLiteContextStore("http_service_ctx.db"),   # swap for your own DB
    max_chunk_tokens=3000,
    max_context_tokens=8192,
    max_workers=4,                                     # parallel map
)


class ProcessRequest(BaseModel):
    content: str
    task: str = "Summarize the key findings"
    session_id: str | None = None


app = FastAPI(title="LongLoom", description="Long-context map/merge/repair service")


@app.post("/process")
def process(req: ProcessRequest):
    """Runs the full pipeline; blocks until done (thread-offloaded internally)."""
    try:
        return processor.process(req.content, task=req.task,
                                 session_id=req.session_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"pipeline failed: {exc}") from exc


@app.get("/sessions/{session_id}")
def session_info(session_id: str):
    info = processor.get_session_info(session_id)
    if info.get("chunk_count", 0) == 0:
        raise HTTPException(status_code=404, detail="unknown session")
    return info


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8300)