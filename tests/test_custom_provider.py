"""Custom LLM providers: base_url, custom headers, optional api_key.

Verified at the WIRE level: a recording OpenAI-compatible stub server proves
the request actually arrives at the custom base_url with the custom headers
attached — the same contract vLLM / LiteLLM / corporate gateways expose.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from longloom import InMemoryContextStore, LLMAdapter, MapReduceProcessor, init_llm
from fake_llm import KEY_CONCEPTS, make_doc


class StubOpenAI:
    """Minimal OpenAI-compatible /chat/completions recorder."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8907):
        self.requests: list[dict] = []
        self._server = ThreadingHTTPServer((host, port), self._handler())
        self._server.daemon_threads = True
        self.host, self.port = host, port

    def _handler(stub: "StubOpenAI"):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                stub.requests.append({
                    "path": self.path,
                    "model": body.get("model"),
                    "messages": body.get("messages", []),
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                })
                payload = json.dumps({
                    "id": "stub-1", "object": "chat.completion",
                    "choices": [{"index": 0,
                                 "message": {"role": "assistant",
                                             "content": "STUB RESPONSE ok"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2,
                              "total_tokens": 5},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return Handler

    def start(self) -> "StubOpenAI":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@pytest.fixture()
def stub():
    stub = StubOpenAI().start()
    yield stub
    stub.stop()


def test_custom_provider_hits_base_url_with_custom_headers(stub):
    llm = init_llm("custom", "acme-internal-model",
                   base_url=stub.base_url,
                   headers={"X-Tenant": "acme", "X-Trace-Id": "tr-1"},
                   api_key="internal-token",
                   timeout=30)
    adapter = LLMAdapter(llm)
    text = adapter.complete("Say something.")

    assert text == "STUB RESPONSE ok"
    req = stub.requests[-1]
    assert req["path"] == "/v1/chat/completions"
    assert req["model"] == "acme-internal-model"
    assert req["headers"]["x-tenant"] == "acme"          # custom headers on the wire
    assert req["headers"]["x-trace-id"] == "tr-1"
    assert req["headers"]["authorization"] == "Bearer internal-token"


def test_custom_provider_api_key_is_optional(stub):
    llm = LLMAdapter(init_llm("custom", "local-model", base_url=stub.base_url, timeout=30))
    assert llm.complete("hi") == "STUB RESPONSE ok"      # no key, no raise
    req = stub.requests[-1]
    assert req["headers"].get("authorization", "").startswith("Bearer not-needed")


def test_custom_provider_full_pipeline(stub):
    """The entire chunk→map→merge pipeline runs against the custom endpoint."""
    events: list[dict] = []
    processor = MapReduceProcessor(
        llm=init_llm("custom", "acme-internal-model",
                     base_url=stub.base_url,
                     headers={"X-Tenant": "acme"},
                     api_key="internal-token", timeout=60),
        store=InMemoryContextStore(),
        max_chunk_tokens=400, max_context_tokens=8192,
        max_workers=3,
        on_event=events.append,
    )
    result = processor.process(make_doc(KEY_CONCEPTS, repeats=30),
                               task="Summarize", session_id="custom-provider")

    assert result["num_chunks"] >= 2
    assert "STUB RESPONSE ok" in result["final_result"]
    # usage metadata flowed back from the custom endpoint through the events
    map_events = [e for e in events if e["event"] == "map_completed"]
    assert map_events and all(e["usage"]["total_tokens"] == 5 for e in map_events)
    # every pipeline LLM call hit the custom base_url
    assert len(stub.requests) >= result["num_chunks"] + 1  # map(s) + merge
    assert all(r["model"] == "acme-internal-model" for r in stub.requests)


def test_nvidia_model_merges_custom_headers():
    from longloom import NVIDIAChatModel

    model = NVIDIAChatModel(api_key="k", headers={"X-Corp-Proxy": "edge-1"})
    headers = model._headers()
    assert headers["Authorization"] == "Bearer k"
    assert headers["X-Corp-Proxy"] == "edge-1"


def test_usage_extracted_from_custom_endpoint(stub):
    adapter = LLMAdapter(init_llm("custom", "m", base_url=stub.base_url, timeout=30))
    _, meta = adapter.complete_with_meta("hello")
    assert meta["usage"] and meta["usage"]["total_tokens"] == 5