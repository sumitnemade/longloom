"""Unstructured-input fixture: messy mixed-format document + planted facts.

The document mimics real-world noise (logs, JSON, email thread, ASCII table,
meeting notes, config, tickets). FACTS lists distinctive tokens planted
exactly once each; no-loss = every token found in the pipeline output.
"""
from __future__ import annotations

FACTS = [
    # log lines
    "tr-8842-cdn", "12483ms", "ERR_504_UPSTREAM", "ap-south-1",
    "u-55291", "10.44.7.12", "pixel-8-pro",
    # incident JSON
    "INC-20931", "db-replica-3", "cache-node-9", "watchdog-v2.4.1",
    "12409", "48210", "war-8842", "ravi.kulkarni@northwind.io",
    # email thread
    "meera.iyer@northwind.io", "v3.12.0", "RB-77123", "$6,150", "2026-11-11",
    # host table
    "orders-api-1", "eu-west-2", "orders-api-2", "search-svc-4", "us-east-2",
    "877",
    # meeting decisions
    "85 to 140", "v3.11.4", "v3.12.1", "500GB", "2026-11-08",
    "$18,400", "PO-40812", "sara.osei", "dmitri.volkov@northwind.io",
    # config snippet
    "idle_timeout_s = 47", "access_ttl_min = 12", "auth.northwind.io",
    # tickets
    "TKT-55291", "TKT-55292", "2026-11-07",
]

_DOC = """
===== raw gateway log — 2026-11-04 (paste from pager, not cleaned) =====
03:12:09 UTC ERROR api-gateway-7 upstream timeout svc=payments-svc trace_id=tr-8842-cdn latency=12483ms code=ERR_504_UPSTREAM
03:12:31 UTC WARN  api-gateway-7 retry scheduled attempt=3 queue_depth=412 region=ap-south-1
03:13:02 UTC INFO  auth-svc login success user=u-55291 mfa=totp ip=10.44.7.12 device=pixel-8-pro
03:14:55 UTC ERROR payments-svc connection refused (see INC-20931)

===== watchdog payload (verbatim json from slack webhook) =====
{"incident_id": "INC-20931", "severity": "sev1", "affected": ["db-replica-3", "cache-node-9"],
 "detected_by": "watchdog-v2.4.1", "customers_impacted": 12409, "loss_usd": 48210,
 "bridge": "https://meet.corp.inc/war-8842", "oncall": "ravi.kulkarni@northwind.io"}

===== email thread: re: postmortem draft v0.3 =====
From: meera.iyer@northwind.io  09:41 IST
root cause = connection pool exhaustion; billing-svc v3.12.0 leaked sockets
finance ok'd refund batch RB-77123 for $6,150 — batch closes 2026-11-11
note: dmitri says dashboards under-count; ignore p50, trust p99 numbers below

===== host snapshot (from ops csv, column order is hostname/region/disk%/p99) =====
orders-api-1    eu-west-2   81%    322ms   degraded
orders-api-2    eu-west-2   44%    198ms   healthy
search-svc-4    us-east-2   92%    877ms   critical

===== incident call decisions (scribbled) =====
- raise pool_max from 85 to 140, staged: canary 10% first
- pin billing-svc to v3.11.4; leak fix ships as v3.12.1
- search-svc-4 volume +500GB deadline 2026-11-08, owner sara.osei
- emergency capacity $18,400 approved (PO-40812), dmitri.volkov@northwind.io signs

===== config drift found during triage (diff against prod) =====
[pool]
idle_timeout_s = 47
[jwt]
access_ttl_min = 12
issuer = https://auth.northwind.io/v2

===== support tickets opened =====
TKT-55291 sev2 checkout 502s during peak — assignee sara.osei, fix eta 2026-11-07
TKT-55292 sev4 docs anchor broken (#refund-policy)
"""


def build_unstructured_doc() -> str:
    doc = _DOC
    for fact in FACTS:
        # Real unstructured docs repeat identifiers across sections; each
        # fact only needs to be planted (preservation counts distinct facts).
        assert doc.count(fact) >= 1, f"fact {fact!r} missing from input"
    return doc


def _variants(fact: str) -> list:
    """Accept LLM formatting normalization (key = value → key: value) —
    information-preserving reformatting is not information loss."""
    return [fact, fact.replace(" = ", ": "), fact.replace(" = ", ":"), fact.replace("=", ":")]


def preservation_report(output: str, facts: list | None = None) -> dict:
    """Fact-by-fact presence check.

    Tolerates information-preserving normalization only:
    * formatting: "key = value" → "key: value"
    * number grouping: 48210 → 48,210 (same value, presentation only)
    * spacing: "1200 ms" → "1200ms"
    Byte-level fidelity is guaranteed separately by the lossless chunk store.
    """
    facts = facts if facts is not None else FACTS
    low = output.lower()
    low_nocomma = low.replace(",", "")
    low_nospace = "".join(low.split())

    def _hit(fact: str) -> bool:
        if any(v.lower() in low for v in _variants(fact)):
            return True
        if fact.replace(",", "").lower() in low_nocomma:      # number grouping
            return True
        return fact.replace(" ", "").lower() in low_nospace   # spacing

    missing = [f for f in facts if not _hit(f)]
    found = [f for f in facts if f not in missing]
    return {
        "total": len(facts),
        "preserved": len(found),
        "missing": missing,
        "rate": 100.0 * len(found) / len(facts),
    }