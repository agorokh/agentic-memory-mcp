from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx
from mcp.server.fastmcp import FastMCP

from agentic_memory.audit import log_call
from agentic_memory.json_util import cap_serialized_tool_payload, tool_json
from agentic_memory.registry import (
    FleetRegistry,
    VaultRecord,
    allowed_modes_for,
    effective_backend,
    effective_graph_namespace,
    load_registry,
    parse_allowlist,
    warn_unknown_allowlist_ids,
    workspace_list_human,
)
from agentic_memory.routing import Router, SearchMode, WorkspaceLookupError, probe_lightrag_endpoint
from agentic_memory.types import MAX_PROMPT_CHARS, MAX_TOOL_LIMIT

_LOG = logging.getLogger("agentic_memory.server")

_PROBE_SEM: asyncio.Semaphore | None = None
_PROBE_SEM_LIMIT: int | None = None

_CLIENT_ERROR_RESERVED = frozenset({"workspace", "http_status", "ok", "error", "detail"})
_QUERY_ERROR_RESERVED = _CLIENT_ERROR_RESERVED | frozenset({"result"})


def _probe_concurrency_limit() -> int:
    raw = os.environ.get("AGENTIC_MEMORY_MAX_PROBE_CONCURRENCY", "8").strip()
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError(
            "AGENTIC_MEMORY_MAX_PROBE_CONCURRENCY must be a positive integer"
        ) from exc
    if n < 1:
        raise ValueError("AGENTIC_MEMORY_MAX_PROBE_CONCURRENCY must be >= 1")
    return min(n, 64)


def _probe_semaphore() -> asyncio.Semaphore:
    global _PROBE_SEM, _PROBE_SEM_LIMIT
    limit = _probe_concurrency_limit()
    if _PROBE_SEM is None or _PROBE_SEM_LIMIT != limit:
        _PROBE_SEM = asyncio.Semaphore(limit)
        _PROBE_SEM_LIMIT = limit
    return _PROBE_SEM


def _merge_error_fields(
    payload: dict[str, Any],
    fields: dict[str, Any],
    *,
    reserved: frozenset[str],
) -> None:
    for key, value in fields.items():
        if key not in reserved:
            payload[key] = value


def configure_logging() -> None:
    level = os.environ.get("AGENTIC_MEMORY_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )


def tool_preamble(router: Router) -> str:
    visible = frozenset(router.visible_workspaces())
    hint = workspace_list_human(visible)
    return (
        f"Effective workspace universe: {hint}. "
        "Never infer workspace across domains; pass workspace explicitly when more than one "
        "workspace is visible."
    )


async def _probe_workspace(client: httpx.AsyncClient, endpoint: str, *, timeout_s: float = 5.0) -> bool:
    async with _probe_semaphore():
        try:
            return await asyncio.wait_for(
                probe_lightrag_endpoint(client, endpoint),
                timeout=timeout_s,
            )
        except TimeoutError:
            return False


async def bootstrap_router() -> tuple[Router, FleetRegistry, frozenset[str] | None]:
    raw_path = os.environ.get("AGENTIC_MEMORY_REGISTRY_PATH", "").strip()
    if not raw_path:
        raise SystemExit(
            "AGENTIC_MEMORY_REGISTRY_PATH is required "
            "(absolute or relative path to fleet_registry.toml)."
        )
    path = Path(raw_path).expanduser().resolve()
    reg = await asyncio.to_thread(load_registry, path)
    allowlist = parse_allowlist(os.environ.get("AGENTIC_MEMORY_ALLOWED_WORKSPACES"))
    warn_unknown_allowlist_ids(reg.vaults, allowlist, log=_LOG)
    router = await Router.build(vaults=reg.vaults, allowlist=allowlist)
    if not router.vaults_by_id:
        raise SystemExit(
            "No workspaces are visible after applying AGENTIC_MEMORY_ALLOWED_WORKSPACES "
            "and enabled flags in the fleet registry. Fix the allowlist or registry entries."
        )

    async def _probe_visible(rec: VaultRecord) -> None:
        if effective_backend(rec) != "lightrag":
            return
        ok = await _probe_workspace(router.client, str(rec.endpoint))
        if not ok:
            _LOG.warning(
                "Startup probe: workspace %r at %s is unreachable; "
                "verify_server_health and queries may report errors until LightRAG is up.",
                rec.id,
                str(rec.endpoint),
            )

    await asyncio.gather(*(_probe_visible(v) for v in router.vaults_by_id.values()))
    return router, reg, allowlist


def _log_query_call(
    *,
    workspace: str,
    prompt: str | None,
    latency_ms: float,
    result_size: int,
    http_status: int | None = None,
) -> None:
    log_call(
        workspace=workspace,
        tool="query_knowledge_graph",
        prompt=prompt,
        latency_ms=latency_ms,
        http_status=http_status,
        result_size=result_size,
    )


def _client_error_payload(
    *,
    code: str,
    workspace: str | None = None,
    detail: str | None = None,
    **fields: Any,
) -> str:
    payload: dict[str, Any] = {
        "workspace": workspace,
        "http_status": None,
        "ok": False,
        "error": code,
    }
    if detail is not None:
        payload["detail"] = detail
    _merge_error_fields(payload, fields, reserved=_CLIENT_ERROR_RESERVED)
    return tool_json(payload)


def _query_error_payload(
    *,
    workspace: str | None,
    code: str,
    detail: str | None = None,
    **fields: Any,
) -> str:
    payload: dict[str, Any] = {
        "workspace": workspace,
        "http_status": None,
        "ok": False,
        "result": None,
        "error": code,
    }
    if detail is not None:
        payload["detail"] = detail
    _merge_error_fields(payload, fields, reserved=_QUERY_ERROR_RESERVED)
    return tool_json(payload)


def _query_tool_payload(workspace: str, status: int | None, data: Any) -> dict[str, Any]:
    ok = status is not None and status < 400
    payload: dict[str, Any] = {
        "workspace": workspace,
        "http_status": status,
        "ok": ok,
        "result": data,
    }
    if not ok:
        payload["error"] = "upstream_http_error"
    return payload


def _health_tool_payload(
    workspace: str,
    status: int | None,
    data: Any,
    *,
    status_key: str,
) -> dict[str, Any]:
    return {
        "workspace": workspace,
        "http_status": status,
        "ok": status is not None and status < 400,
        status_key: data if isinstance(data, dict) else {"raw": data},
    }


def _workspace_resolution_error(exc: BaseException) -> tuple[str, dict[str, Any]]:
    """Parse resolve_workspace failures into an MCP error code and structured fields."""
    message = str(exc)
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return "workspace_resolution_failed", {"detail": message}
    if not isinstance(payload, dict):
        return "workspace_resolution_failed", {"detail": message}
    fields = dict(payload)
    code = str(fields.pop("error", "workspace_resolution_failed"))
    return code, fields


def build_mcp(router: Router) -> FastMCP:
    preamble = tool_preamble(router)
    mcp = FastMCP("agentic-memory")

    def _resolve_workspace(
        workspace: str | None,
    ) -> tuple[str | None, tuple[str, dict[str, Any]] | None]:
        try:
            return router.resolve_workspace(workspace), None
        except (ValueError, WorkspaceLookupError) as exc:
            return None, _workspace_resolution_error(exc)

    async def _run_tool(
        *,
        tool: str,
        workspace: str | None,
        prompt: str | None,
        run: Callable[[str], Awaitable[tuple[int | None, Any]]],
    ) -> str:
        t0 = time.perf_counter()
        ws, err = _resolve_workspace(workspace)
        if err is not None:
            code, fields = err
            text = _client_error_payload(code=code, workspace=workspace, **fields)
            log_call(
                workspace=workspace or "*",
                tool=tool,
                prompt=prompt,
                latency_ms=(time.perf_counter() - t0) * 1000,
                http_status=None,
                result_size=len(text),
            )
            return text
        assert ws is not None
        try:
            status, data = await run(ws)
        except httpx.HTTPError:
            latency = (time.perf_counter() - t0) * 1000
            log_call(
                workspace=ws,
                tool=tool,
                prompt=prompt,
                latency_ms=latency,
                http_status=None,
                result_size=0,
            )
            return _client_error_payload(
                workspace=ws,
                code="http_error",
                detail="upstream unreachable",
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            log_call(
                workspace=ws,
                tool=tool,
                prompt=prompt,
                latency_ms=latency,
                http_status=None,
                result_size=0,
            )
            return _client_error_payload(
                workspace=ws,
                code=f"{tool}_failed",
                detail=str(exc),
            )
        latency = (time.perf_counter() - t0) * 1000
        text = tool_json(data)
        log_call(
            workspace=ws,
            tool=tool,
            prompt=prompt,
            latency_ms=latency,
            http_status=status,
            result_size=len(text),
        )
        return text

    @mcp.tool(
        name="query_knowledge_graph",
        description=(
            "Read-path LightRAG query over HTTP (POST /query). "
            "``limit`` caps serialized JSON returned to the MCP client (~limit × 400 characters); "
            "it does not set LightRAG recall/top_k. "
            "``semantic`` maps to LightRAG ``local``; ``keyword`` maps to ``naive``. "
            f"{preamble}"
        ),
    )
    async def query_knowledge_graph(
        prompt: str,
        workspace: str | None = None,
        search_mode: SearchMode = "mix",
        limit: int = 60,
        context_only: bool = False,
        prompt_only: bool = False,
    ) -> str:
        t0 = time.perf_counter()
        audit_ws = workspace or "*"

        def _reject(text: str, *, ws: str | None = None) -> str:
            _log_query_call(
                workspace=ws or audit_ws,
                prompt=prompt,
                latency_ms=(time.perf_counter() - t0) * 1000,
                http_status=None,
                result_size=len(text),
            )
            return text

        if limit <= 0 or limit > MAX_TOOL_LIMIT:
            return _reject(
                _query_error_payload(
                    workspace=audit_ws,
                    code="invalid_limit",
                    detail=f"`limit` must be between 1 and {MAX_TOOL_LIMIT}.",
                )
            )
        if len(prompt) > MAX_PROMPT_CHARS:
            return _reject(
                _query_error_payload(
                    workspace=audit_ws,
                    code="prompt_too_large",
                    detail=f"`prompt` must be at most {MAX_PROMPT_CHARS} characters.",
                )
            )
        ws, err = _resolve_workspace(workspace)
        if err is not None:
            code, fields = err
            return _reject(
                _query_error_payload(workspace=audit_ws, code=code, **fields),
            )
        assert ws is not None
        rec = router.vaults_by_id[ws]
        if effective_backend(rec) != "lightrag":
            return _reject(
                _query_error_payload(
                    workspace=ws,
                    code="unsupported_backend",
                    backend=effective_backend(rec),
                    detail="This bridge only implements LightRAG HTTP read paths.",
                ),
                ws=ws,
            )
        if search_mode not in allowed_modes_for(rec):
            return _reject(
                _query_error_payload(
                    workspace=ws,
                    code="mode_not_allowed",
                    search_mode=search_mode,
                    allowed_modes=sorted(allowed_modes_for(rec)),
                ),
                ws=ws,
            )
        try:
            status, data = await router.query_lightrag(
                workspace_id=ws,
                prompt=prompt,
                search_mode=search_mode,
                context_only=context_only,
                prompt_only=prompt_only,
            )
        except httpx.HTTPError:
            return _reject(
                _query_error_payload(
                    workspace=ws,
                    code="http_error",
                    detail="upstream unreachable",
                ),
                ws=ws,
            )
        except Exception as exc:
            return _reject(
                _query_error_payload(workspace=ws, code="query_failed", detail=str(exc)),
                ws=ws,
            )
        payload = _query_tool_payload(ws, status, data)
        if limit > 0:
            payload = cap_serialized_tool_payload(payload, limit)
        text = tool_json(payload)
        _log_query_call(
            workspace=ws,
            prompt=prompt,
            latency_ms=(time.perf_counter() - t0) * 1000,
            http_status=status,
            result_size=len(text),
        )
        return text

    @mcp.tool(
        name="get_graph_metadata",
        description=f"Return LightRAG /health JSON for a workspace. {preamble}",
    )
    async def get_graph_metadata(workspace: str | None = None) -> str:
        async def run(ws: str) -> tuple[int | None, Any]:
            status, data = await router.get_health_json(ws)
            return status, _health_tool_payload(ws, status, data, status_key="health")

        return await _run_tool(
            tool="get_graph_metadata",
            workspace=workspace,
            prompt=None,
            run=run,
        )

    @mcp.tool(
        name="verify_server_health",
        description=(
            "Probe LightRAG HTTP health for one workspace or all visible workspaces when "
            'workspace is "*". '
            f"{preamble}"
        ),
    )
    async def verify_server_health(workspace: str = "*") -> str:
        t0 = time.perf_counter()
        if workspace.strip() == "*":
            targets = router.visible_workspaces()
        else:
            ws, err = _resolve_workspace(workspace)
            if err is not None:
                code, fields = err
                text = _client_error_payload(code=code, workspace=workspace, **fields)
                log_call(
                    workspace=workspace,
                    tool="verify_server_health",
                    prompt=None,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    http_status=None,
                    result_size=len(text),
                )
                return text
            assert ws is not None
            targets = [ws]

        async def _probe_row(ws_id: str) -> dict[str, Any]:
            rec = router.vaults_by_id[ws_id]
            if effective_backend(rec) != "lightrag":
                return {
                    "workspace": ws_id,
                    "reachable": False,
                    "backend": effective_backend(rec),
                    "detail": "only LightRAG HTTP endpoints are probed",
                }
            ok = await _probe_workspace(router.client, str(rec.endpoint))
            return {
                "workspace": ws_id,
                "reachable": ok,
                "endpoint": str(rec.endpoint),
            }

        rows = list(await asyncio.gather(*(_probe_row(ws) for ws in targets)))
        latency = (time.perf_counter() - t0) * 1000
        text = tool_json({"workspaces": rows})
        audit_workspace = "*" if len(targets) != 1 else targets[0]
        log_call(
            workspace=audit_workspace,
            tool="verify_server_health",
            prompt=None,
            latency_ms=latency,
            http_status=None,
            result_size=len(text),
        )
        return text

    @mcp.tool(
        name="check_indexing_status",
        description=(
            "Return /health payload as coarse indexing/server status (LightRAG-specific fields "
            f"vary by version). {preamble}"
        ),
    )
    async def check_indexing_status(workspace: str | None = None) -> str:
        async def run(ws: str) -> tuple[int | None, Any]:
            status, data = await router.get_health_json(ws)
            return status, _health_tool_payload(
                ws, status, data, status_key="pipeline_status"
            )

        return await _run_tool(
            tool="check_indexing_status",
            workspace=workspace,
            prompt=None,
            run=run,
        )

    list_ws_desc = (
        "List enabled workspaces visible to this bridge after allowlist filtering. " + preamble
    )

    @mcp.tool(
        name="list_workspaces",
        description=list_ws_desc,
    )
    async def list_workspaces() -> str:
        t0 = time.perf_counter()
        out: list[dict[str, Any]] = []
        for vid in sorted(router.vaults_by_id):
            rec = router.vaults_by_id[vid]
            backend = effective_backend(rec)
            out.append(
                {
                    "id": rec.id,
                    "endpoint": str(rec.endpoint),
                    "vault_root": rec.vault_root,
                    "enabled": True,
                    "backend": backend,
                    "origin": rec.origin,
                    "graph_namespace": (
                        effective_graph_namespace(rec) if backend == "graphiti" else None
                    ),
                    "allowed_modes": sorted(allowed_modes_for(rec)),
                    "query_supported": backend == "lightrag",
                }
            )
        latency = (time.perf_counter() - t0) * 1000
        text = tool_json(out)
        log_call(
            workspace="*",
            tool="list_workspaces",
            prompt=None,
            latency_ms=latency,
            http_status=None,
            result_size=len(text),
        )
        return text

    return mcp


async def _async_main() -> None:
    router, _reg, _ = await bootstrap_router()
    try:
        mcp = build_mcp(router)
        await mcp.run_stdio_async()
    finally:
        await router.aclose()


def main() -> None:
    configure_logging()
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
