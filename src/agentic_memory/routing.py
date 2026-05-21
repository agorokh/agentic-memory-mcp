from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, PrivateAttr

from agentic_memory.registry import VaultRecord, apply_allowlist, effective_backend, validate_endpoint_url
from agentic_memory.types import SearchMode

_LOG = logging.getLogger("agentic_memory.routing")


class WorkspaceLookupError(LookupError):
    """Raised when ``workspace`` is unknown or disabled in the registry."""


def _env_int(name: str, default: str) -> int:
    raw = os.environ.get(name, default).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: str) -> float:
    raw = os.environ.get(name, default).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _http_limits() -> httpx.Limits:
    max_conn = _env_int("AGENTIC_MEMORY_HTTP_MAX_CONNECTIONS", "64")
    max_keep = _env_int("AGENTIC_MEMORY_HTTP_MAX_KEEPALIVE", "16")
    if max_conn < 1:
        raise ValueError("AGENTIC_MEMORY_HTTP_MAX_CONNECTIONS must be >= 1")
    if max_keep < 0:
        raise ValueError("AGENTIC_MEMORY_HTTP_MAX_KEEPALIVE must be >= 0")
    return httpx.Limits(
        max_connections=max_conn,
        max_keepalive_connections=max_keep,
    )


def _http_timeout(timeout_s: float) -> httpx.Timeout:
    read_s = _env_float("AGENTIC_MEMORY_QUERY_READ_TIMEOUT_S", str(timeout_s))
    if read_s <= 0:
        raise ValueError("AGENTIC_MEMORY_QUERY_READ_TIMEOUT_S must be > 0")
    return httpx.Timeout(connect=5.0, read=read_s, write=10.0, pool=5.0)


def _upstream_ok_status(status: int) -> bool:
    """Only 2xx responses count as successful upstream HTTP (matches tool payloads)."""
    return 200 <= status < 300


async def probe_lightrag_endpoint(client: httpx.AsyncClient, endpoint: HttpUrl | str) -> bool:
    """Return True when ``/health`` or ``/`` returns 2xx (redirects are not success)."""
    try:
        base = (await asyncio.to_thread(validate_endpoint_url, str(endpoint))).rstrip("/")
    except ValueError as exc:
        _LOG.debug("probe %s rejected during endpoint validation: %s", endpoint, exc)
        return False
    for path in ("/health", "/"):
        try:
            resp = await client.get(f"{base}{path}")
            if _upstream_ok_status(resp.status_code):
                return True
        except httpx.HTTPError as exc:
            _LOG.debug("probe %s%s failed: %s", base, path, exc)
    return False


class Router(BaseModel):
    """Workspace routing + LightRAG HTTP client."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vaults_by_id: dict[str, VaultRecord]
    allowlist: frozenset[str] | None
    client: httpx.AsyncClient = Field(..., exclude=True)
    _owns_client: bool = PrivateAttr(default=True)

    @classmethod
    async def build(
        cls,
        *,
        vaults: list[VaultRecord],
        allowlist: frozenset[str] | None,
        timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> Router:
        effective = apply_allowlist(vaults, allowlist)
        seen: set[str] = set()
        for v in effective:
            if v.id in seen:
                raise ValueError(f"Duplicate workspace id {v.id!r} in fleet registry.")
            seen.add(v.id)
        by_id = {v.id: v for v in effective}
        owns = client is None
        if client is None:
            client = httpx.AsyncClient(
                timeout=_http_timeout(timeout_s),
                limits=_http_limits(),
                follow_redirects=False,
            )
        inst = cls(vaults_by_id=by_id, allowlist=allowlist, client=client)
        inst._owns_client = owns
        return inst

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def visible_workspaces(self) -> list[str]:
        return sorted(self.vaults_by_id)

    def resolve_workspace(self, workspace: str | None) -> str:
        visible = self.visible_workspaces()
        if workspace is None or workspace == "":
            if len(visible) == 1:
                return visible[0]
            raise ValueError(
                json.dumps(
                    {
                        "error": "workspace_required",
                        "visible_workspaces": visible,
                        "hint": "Pass workspace= explicitly when more than one entry is visible.",
                    },
                    separators=(",", ":"),
                )
            )
        if workspace not in self.vaults_by_id:
            raise WorkspaceLookupError(
                json.dumps(
                    {
                        "error": "workspace_unknown",
                        "requested": workspace,
                        "visible_workspaces": visible,
                    },
                    separators=(",", ":"),
                )
            )
        return workspace

    async def _validated_base_url(self, workspace_id: str) -> str:
        rec = self.vaults_by_id[workspace_id]
        url = await asyncio.to_thread(validate_endpoint_url, str(rec.endpoint))
        return url.rstrip("/")

    def _map_mode(self, search_mode: SearchMode) -> str:
        if search_mode in ("mix", "global", "hybrid", "local", "naive"):
            return search_mode
        if search_mode == "semantic":
            return "local"
        if search_mode == "keyword":
            return "naive"
        raise ValueError(f"unsupported search_mode: {search_mode!r}")

    def _build_query_body(
        self,
        *,
        prompt: str,
        search_mode: SearchMode,
        context_only: bool,
        prompt_only: bool,
    ) -> dict[str, Any]:
        mode = self._map_mode(search_mode)
        only_ctx = context_only or prompt_only
        body: dict[str, Any] = {
            "query": prompt,
            "mode": mode,
            "only_need_context": only_ctx,
            "include_references": True,
            "include_chunk_content": context_only and not prompt_only,
            "enable_rerank": True,
        }
        return body

    async def query_lightrag(
        self,
        *,
        workspace_id: str,
        prompt: str,
        search_mode: SearchMode,
        context_only: bool,
        prompt_only: bool,
    ) -> tuple[int | None, Any]:
        backend = effective_backend(self.vaults_by_id[workspace_id])
        if backend != "lightrag":
            return None, {
                "error": "unsupported_backend",
                "workspace": workspace_id,
                "backend": backend,
                "detail": "This bridge only implements LightRAG HTTP read paths.",
            }
        base = await self._validated_base_url(workspace_id)
        body = self._build_query_body(
            prompt=prompt,
            search_mode=search_mode,
            context_only=context_only,
            prompt_only=prompt_only,
        )
        resp = await self.client.post(f"{base}/query", json=body)
        try:
            data = resp.json()
        except json.JSONDecodeError:
            data = {"raw": resp.text}
        return resp.status_code, data

    async def get_health_json(self, workspace_id: str) -> tuple[int | None, Any]:
        backend = effective_backend(self.vaults_by_id[workspace_id])
        if backend != "lightrag":
            return None, {
                "error": "unsupported_backend",
                "workspace": workspace_id,
                "backend": backend,
            }
        base = await self._validated_base_url(workspace_id)
        resp = await self.client.get(f"{base}/health")
        try:
            return resp.status_code, resp.json()
        except json.JSONDecodeError:
            return resp.status_code, {"raw": resp.text}
