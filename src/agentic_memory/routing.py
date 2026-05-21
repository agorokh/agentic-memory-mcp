from __future__ import annotations

import json
import logging
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, PrivateAttr

from agentic_memory.registry import VaultRecord, apply_allowlist

_LOG = logging.getLogger("agentic_memory.routing")

SearchMode = Literal["mix", "semantic", "keyword", "global", "hybrid", "local", "naive"]


class WorkspaceLookupError(LookupError):
    """Raised when ``workspace`` is unknown or disabled in the registry."""


async def probe_lightrag_endpoint(client: httpx.AsyncClient, endpoint: HttpUrl | str) -> bool:
    """Return True when ``/health`` or ``/`` returns a 2xx/3xx (not 4xx/5xx)."""
    base = str(endpoint).rstrip("/")
    for path in ("/health", "/"):
        try:
            resp = await client.get(f"{base}{path}")
            if 200 <= resp.status_code < 400:
                return True
        except httpx.HTTPError as exc:
            _LOG.debug("probe %s%s failed: %s", base, path, exc)
    return False


class Router(BaseModel):
    """Workspace routing + LightRAG HTTP client.

    One shared :class:`httpx.AsyncClient` multiplexes all workspace base URLs; ``httpx.Limits``
    constrain the client process-wide (not as independent per-host pools). Prefer accurate
    capacity planning over assuming strict per-endpoint isolation.
    """

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
        limits = httpx.Limits(max_keepalive_connections=4, max_connections=32)
        owns = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=timeout_s, limits=limits)
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
                    indent=2,
                )
            )
        if workspace not in self.vaults_by_id:
            raise WorkspaceLookupError(
                json.dumps(
                    {
                        "error": "workspace_unknown",
                        "requested": workspace,
                        "visible_workspaces": visible,
                        "allowed_workspaces": sorted(self.allowlist)
                        if self.allowlist is not None
                        else None,
                    },
                    indent=2,
                )
            )
        return workspace

    def _base_url(self, workspace_id: str) -> str:
        rec = self.vaults_by_id[workspace_id]
        return str(rec.endpoint).rstrip("/")

    def _map_mode(self, search_mode: SearchMode) -> str:
        # LightRAG HTTP ``mode`` values; map vendor-specific names conservatively.
        if search_mode in ("mix", "global", "hybrid", "local", "naive"):
            return search_mode
        if search_mode == "semantic":
            return "local"
        if search_mode == "keyword":
            return "naive"
        return "mix"

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
        limit: int,
        context_only: bool,
        prompt_only: bool,
    ) -> tuple[int | None, Any]:
        base = self._base_url(workspace_id)
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
        if limit > 0:
            data = self._maybe_truncate_payload(data, limit)
        return resp.status_code, data

    def _maybe_truncate_payload(self, data: Any, limit: int) -> Any:
        """Coarse cap on serialized size using ``limit`` as a rough token proxy.

        When the full ``indent=2`` JSON exceeds the cap, return a wrapper dict whose
        ``preview`` is sized so a *single* ``json.dumps(..., indent=2)`` stays within
        the cap (avoids embedding a JSON string that then gets escaped a second time).
        """
        cap = limit * 400
        try:
            text = json.dumps(data, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return data
        if len(text) <= cap:
            return data
        lo, hi = 0, min(len(text), cap)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = {"truncated": True, "limit": limit, "preview": text[:mid]}
            serialized = json.dumps(candidate, ensure_ascii=False, indent=2)
            if len(serialized) <= cap:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return {"truncated": True, "limit": limit, "preview": text[:best]}

    async def get_health_json(self, workspace_id: str) -> tuple[int | None, Any]:
        base = self._base_url(workspace_id)
        resp = await self.client.get(f"{base}/health")
        try:
            return resp.status_code, resp.json()
        except json.JSONDecodeError:
            return resp.status_code, {"raw": resp.text}
