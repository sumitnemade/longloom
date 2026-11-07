"""A minimal PostgREST-compatible HTTP stub for local Supabase verification.

Implements exactly the REST surface longloom's SupabaseContextStore uses via
supabase-py (which wraps postgrest-py):

* ``POST   /rest/v1/{table}?on_conflict=id``   upsert (Prefer: resolution=merge-duplicates)
* ``GET    /rest/v1/{table}?select=*&col=eq.v&col=in.(a,b)&order=col.asc``
* ``DELETE /rest/v1/{table}?col=eq.v``

This verifies the client code path locally; it is NOT the Supabase cloud.
"""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

_REST_RE = re.compile(r"^/rest/v1/([A-Za-z_][A-Za-z0-9_]*)$")


class StubPostgREST:
    def __init__(self, host: str = "127.0.0.1", port: int = 8900):
        self.tables: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self._server.daemon_threads = True
        self.host, self.port = host, port

    # -- request handling -------------------------------------------------

    def _make_handler(stub: "StubPostgREST"):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _table(self) -> Tuple[str, Dict[str, List[str]]]:
                m = _REST_RE.match(urlparse(self.path).path)
                if not m:
                    raise ValueError(f"unhandled path {self.path}")
                return m.group(1), parse_qs(urlparse(self.path).query)

            def do_POST(self):
                table, q = self._table()
                rows = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                on_conflict = (q.get("on_conflict") or ["id"])[0]
                bucket = stub.tables.setdefault(table, {})
                for row in rows:
                    bucket[str(row[on_conflict])] = row
                self._json(201, rows)

            def do_GET(self):
                table, q = self._table()
                rows = list(stub.tables.get(table, {}).values())
                rows = _apply_filters(rows, q)
                order = (q.get("order") or [""])[0]
                if order:
                    col = order.split(".")[0]
                    rows.sort(key=lambda r: r.get(col, 0))
                self._json(200, rows)

            def do_DELETE(self):
                table, q = self._table()
                bucket = stub.tables.get(table, {})
                for key in [k for k, r in bucket.items() if _matches(r, q)]:
                    del bucket[key]
                self._json(204, [])

            def _json(self, status: int, body: Any):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        return Handler

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "StubPostgREST":
        import threading

        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


def _matches(row: Dict[str, Any], q: Dict[str, List[str]]) -> bool:
    for col, values in q.items():
        if col in ("select", "order", "on_conflict", "limit", "offset", "columns"):
            continue
        for value in values:  # PostgREST encodes filters as "col=op.value"
            op, _, raw = value.partition(".")
            if op == "eq" and str(row.get(col)) != raw:
                return False
            if op == "in" and str(row.get(col)) not in _parse_in(raw):
                return False
    return True


def _apply_filters(rows: List[Dict[str, Any]], q: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    return [r for r in rows if _matches(r, q)]


def _parse_in(value: str) -> List[str]:
    inner = value.strip("()")
    return [part.strip().strip('"') for part in inner.split(",") if part.strip()]


if __name__ == "__main__":
    stub = StubPostgREST().start()
    print(f"PostgREST stub on {stub.url()} — Ctrl+C to stop")
    import signal

    signal.pause()