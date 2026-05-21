from __future__ import annotations

import ipaddress
import os
import socket
import tomllib
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from agentic_memory.types import ALL_SEARCH_MODES, SearchMode

# ---------------------------------------------------------------------------
# Schema version policy
# ---------------------------------------------------------------------------
# The loader accepts any version in ``SUPPORTED_SCHEMA_VERSIONS``. Writers
# emit ``REGISTRY_SCHEMA_VERSION``. v2 added per-record ``backend`` / ``origin``
# / ``graph_namespace`` fields; all are optional so v1 manifests load cleanly
# and the implicit backend is "lightrag" (the only substrate this bridge has
# ever served).
# ---------------------------------------------------------------------------

SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1", "2"})
REGISTRY_SCHEMA_VERSION = "2"
assert REGISTRY_SCHEMA_VERSION in SUPPORTED_SCHEMA_VERSIONS

Backend = Literal["lightrag", "graphiti"]
Origin = Literal["repo-product", "repo-embedded", "human-curated"]

_BLOCKED_HOSTS = frozenset({"metadata.google.internal"})
_BLOCKED_HOSTNAMES = frozenset({"localhost"})


def _allow_private_endpoints() -> bool:
    raw = os.environ.get("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_non_public_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if hasattr(addr, "is_global"):
        return not addr.is_global
    return bool(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
    )


def _normalize_hostname(host: str) -> str:
    """Lowercase and strip a trailing dot so FQDN forms match blocked-host checks."""
    return host.lower().rstrip(".")


def _reject_private_host(host: str) -> None:
    host = _normalize_hostname(host)
    if host in _BLOCKED_HOSTS or host in _BLOCKED_HOSTNAMES:
        raise ValueError(
            f"private or link-local endpoint blocked: {host!r} "
            "(set AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS=1 for local dev)"
        )
    if host.endswith(".localhost") or host.endswith(".local"):
        raise ValueError(
            f"private or link-local endpoint blocked: {host!r} "
            "(set AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS=1 for local dev)"
        )
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        try:
            for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
                resolved = ipaddress.ip_address(info[4][0])
                if _is_non_public_address(resolved):
                    raise ValueError(
                        f"private or link-local endpoint blocked: {host!r} "
                        "(set AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS=1 for local dev)"
                    ) from None
        except socket.gaierror as exc:
            raise ValueError(
                f"could not resolve endpoint hostname {host!r}; "
                "refuse registry endpoints with unknown DNS in strict mode"
            ) from exc
        return
    if _is_non_public_address(addr):
        raise ValueError(
            f"private or link-local endpoint blocked: {host!r} "
            "(set AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS=1 for local dev)"
        )


def validate_endpoint_url(url: str) -> str:
    """Reject unsafe endpoint URLs unless private endpoints are explicitly allowed."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("endpoint must use http or https")
    if parsed.username or parsed.password:
        raise ValueError("endpoint must not embed credentials in the URL")
    host = _normalize_hostname(parsed.hostname or "")
    if not host:
        raise ValueError("endpoint hostname is required")
    if host in _BLOCKED_HOSTS:
        raise ValueError(f"blocked host: {host!r}")
    if not _allow_private_endpoints():
        _reject_private_host(host)
    return url


class VaultRecord(BaseModel):
    """One workspace row in ``fleet_registry.toml`` (``[[vaults]]``)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(..., min_length=1, description="Stable workspace id for MCP tools.")
    endpoint: HttpUrl
    vault_root: str | None = None
    enabled: bool = True
    backend: Backend | None = None
    origin: Origin | None = None
    graph_namespace: str | None = None
    allowed_modes: list[SearchMode] | None = None

    @field_validator("endpoint", mode="before")
    @classmethod
    def _validate_endpoint(cls, value: object) -> object:
        return validate_endpoint_url(str(value))

    @field_validator("allowed_modes")
    @classmethod
    def _validate_allowed_modes(cls, modes: list[SearchMode] | None) -> list[SearchMode] | None:
        if modes is None:
            return None
        unknown = [m for m in modes if m not in ALL_SEARCH_MODES]
        if unknown:
            raise ValueError(f"unknown allowed_modes: {unknown}")
        return modes


class FleetRegistry(BaseModel):
    """Root document loaded from ``AGENTIC_MEMORY_REGISTRY_PATH``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default=REGISTRY_SCHEMA_VERSION, min_length=1)
    vaults: list[VaultRecord] = Field(default_factory=list)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _coerce_schema_version(cls, value: object) -> str:
        return str(value)


def effective_backend(record: VaultRecord) -> Backend:
    """Return the effective backend, defaulting to ``"lightrag"`` for v1 rows."""
    return record.backend or "lightrag"


def effective_graph_namespace(record: VaultRecord) -> str:
    """Return the effective Graphiti ``group_id`` for the workspace."""
    return record.graph_namespace or record.id


def allowed_modes_for(record: VaultRecord) -> frozenset[str]:
    """Modes permitted for ``query_knowledge_graph`` on this workspace."""
    if record.allowed_modes is None:
        return ALL_SEARCH_MODES
    return frozenset(record.allowed_modes)


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


def workspace_list_human(ids: frozenset[str]) -> str:
    items = sorted(ids)
    return ", ".join(items) if items else "(none)"


def apply_allowlist(
    vaults: list[VaultRecord],
    allowlist: frozenset[str] | None,
) -> list[VaultRecord]:
    """Filter registry rows by allowlist (if set)."""
    enabled = [v for v in vaults if v.enabled]
    if allowlist is None:
        return enabled
    return [v for v in enabled if v.id in allowlist]


def warn_unknown_allowlist_ids(
    vaults: list[VaultRecord],
    allowlist: frozenset[str] | None,
    *,
    log: object,
) -> None:
    if allowlist is None:
        return
    known = {v.id for v in vaults if v.enabled}
    unknown = allowlist - known
    if unknown:
        log.warning(  # type: ignore[union-attr]
            "Allowlist ids not in enabled registry: %s",
            sorted(unknown),
        )
