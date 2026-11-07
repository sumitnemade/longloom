"""Physical naming configuration — embed LongLoom inside existing databases.

Real deployments don't provision dedicated databases for chunk storage: the
chunks live in an existing table/collection next to the application's own
data. :class:`FieldMap` is the single knob for that:

* ``table``           — physical table/collection name (app-owned naming)
* ``columns``         — logical → physical field names (your schema, not ours)
* ``namespace``       — a discriminator value stored on every row, so several
                        applications can share ONE table without seeing each
                        other's chunks
* ``namespace_field`` — the column/field carrying that discriminator

All physical names must be valid identifiers (defence against SQL injection
through the mapping).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = ["DEFAULT_COLUMNS", "FieldMap"]

#: Canonical logical field order used by SQL persistence.
LOGICAL_FIELDS = (
    "id", "content_hash", "metadata", "token_count", "created_at",
    "chunk_index", "total_chunks", "session_id", "content",
)

DEFAULT_COLUMNS: Dict[str, str] = {name: name for name in LOGICAL_FIELDS}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class FieldMap:
    """Physical naming for a store: table, column overrides, namespace."""

    table: str = "context_chunks"
    columns: Dict[str, str] = field(default_factory=dict)
    namespace: Optional[str] = None
    namespace_field: str = "namespace"

    def __post_init__(self) -> None:
        if not _IDENT_RE.match(self.table or ""):
            raise ValueError(f"invalid table/collection name: {self.table!r}")
        if not _IDENT_RE.match(self.namespace_field or ""):
            raise ValueError(f"invalid namespace field: {self.namespace_field!r}")
        unknown = set(self.columns or {}) - set(LOGICAL_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown logical field(s): {sorted(unknown)}. "
                f"Valid: {list(LOGICAL_FIELDS)}")
        for logical, physical in (self.columns or {}).items():
            if not _IDENT_RE.match(physical or ""):
                raise ValueError(f"invalid physical name for {logical!r}: {physical!r}")

    def col(self, logical: str) -> str:
        """Physical name for a logical field."""
        return (self.columns or {}).get(logical, DEFAULT_COLUMNS[logical])

    @property
    def all_columns(self) -> Dict[str, str]:
        """Logical → physical, in canonical order (namespace appended)."""
        out = {name: self.col(name) for name in LOGICAL_FIELDS}
        if self.namespace is not None:
            out["namespace"] = self.namespace_field
        return out
