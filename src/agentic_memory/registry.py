from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# ---------------------------------------------------------------------------
# Schema version policy
# ---------------------------------------------------------------------------
# The loader accepts any version in ``SUPPORTED_SCHEMA_VERSIONS``. Writers
# emit ``REGISTRY_SCHEMA_VERSION``. v2 added per-record ``backend`` / ``origin``
# / ``graph_namespace`` fields; all are optional so v1 manifests load cleanly
# and the implicit backend is "lightrag" (the only substrate this bridge has
# ever served).
#
# Coordinate any further bump with the upstream registry materializer
# (``tools/hermes_adapter/agentic_memory_registry_materialize.py``).
# ---------------------------------------------------------------------------

SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1", "2"})
REGISTRY_SCHEMA_VERSION = "2"
assert REGISTRY_SCHEMA_VERSION in SUPPORTED_SCHEMA_VERSIONS

Backend = Literal["lightrag", "graphiti"]
Origin = Literal["repo-product", "repo-embedded", "human-curated"]


class VaultRecord(BaseModel):
    """One workspace row in ``fleet_registry.toml`` (``[[vaults]]``).

    Schema v2 additions (all optional, defaults preserve v1 semantics):

    - ``backend``: which Tier-3 substrate serves this workspace. ``None``
      means v1 implicit default = ``"lightrag"``. Use :func:`effective_backend`
      to resolve to a concrete value.
    - ``origin``: vault origin classification per
      ``docs/00_Core/VAULT_TAXONOMY.md`` in template-repo.
    - ``graph_namespace``: for ``backend="graphiti"``, the per-call ``group_id``
      that partitions the shared Neo4j graph. Defaults to ``id``. Ignored for
      ``backend="lightrag"`` (LightRAG isolates by per-workspace HTTP server).
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(..., min_length=1, description="Stable workspace id for MCP tools.")
    endpoint: HttpUrl
    vault_root: str | None = None
    enabled: bool = True
    backend: Backend | None = None
    origin: Origin | None = None
    graph_namespace: str | None = None


class FleetRegistry(BaseModel):
    """Root document loaded from ``AGENTIC_MEMORY_REGISTRY_PATH``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=REGISTRY_SCHEMA_VERSION, min_length=1)
    vaults: list[VaultRecord] = Field(default_factory=list)


def effective_backend(record: VaultRecord) -> Backend:
    """Return the effective backend, defaulting to ``"lightrag"`` for v1 rows."""
    return record.backend or "lightrag"


def effective_graph_namespace(record: VaultRecord) -> str:
    """Return the effective Graphiti ``group_id`` for the workspace.

    Defaults to the workspace ``id`` when ``graph_namespace`` is unset.
    Callers should still gate on ``effective_backend(record) == "graphiti"``
    before using this value — for ``lightrag`` workspaces the namespace
    concept does not apply.
    """
    return record.graph_namespace or record.id


def load_registry(path: Path) -> FleetRegistry:
    if not path.is_file():
        raise FileNotFoundError(f"AGENTIC_MEMORY_REGISTRY_PATH not found: {path}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        reg = FleetRegistry.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid fleet registry TOML ({path}): {exc}") from exc
    if reg.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported registry schema_version {reg.schema_version!r}; "
            f"this bridge supports {sorted(SUPPORTED_SCHEMA_VERSIONS)!r}. "
            "Coordinate with the upstream registry materializer before changing the contract."
        )
    return reg


def parse_allowlist(raw: str | None) -> frozenset[str] | None:
    """Return ``None`` when allowlist is unset or empty (all enabled vaults allowed)."""
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    ids = [p for p in parts if p]
    if not ids:
        return None
    return frozenset(ids)


def allowlist_human(ids: frozenset[str] | None) -> str:
    if ids is None:
        return "(all enabled registry workspaces)"
    return ", ".join(sorted(ids))


def apply_allowlist(
    vaults: list[VaultRecord],
    allowlist: frozenset[str] | None,
) -> list[VaultRecord]:
    """Filter registry rows by allowlist (if set)."""
    enabled = [v for v in vaults if v.enabled]
    if allowlist is None:
        return enabled
    return [v for v in enabled if v.id in allowlist]
