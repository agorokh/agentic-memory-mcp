from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from agentic_memory.audit import log_call
from agentic_memory.registry import (
    FleetRegistry,
    VaultRecord,
    allowlist_human,
    effective_backend,
    effective_graph_namespace,
    load_registry,
    parse_allowlist,
)
from agentic_memory.routing import Router, SearchMode, WorkspaceLookupError, probe_lightrag_endpoint

_LOG = logging.getLogger("agentic_memory.server")


def configure_logging() -> None:
    level = os.environ.get("AGENTIC_MEMORY_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )


def tool_preamble(router: Router) -> str:
    visible = router.visible_workspaces()
    hint = allowlist_human(frozenset(visible)) if visible else "(none)"
    return (
        f"Effective workspace universe: {hint}. "
        "Never infer workspace across domains; pass workspace explicitly when more than one "
        "workspace is visible."
    )


async def bootstrap_router() -> tuple[Router, FleetRegistry, frozenset[str] | None]:
    raw_path = os.environ.get("AGENTIC_MEMORY_REGISTRY_PATH", "").strip()
    if not raw_path:
        raise SystemExit(
            "AGENTIC_MEMORY_REGISTRY_PATH is required "
            "(absolute or relative path to fleet_registry.toml)."
        )
    path = Path(raw_path).expanduser()
    reg = load_registry(path)
    allowlist = parse_allowlist(os.environ.get("AGENTIC_MEMORY_ALLOWED_WORKSPACES"))
    router = await Router.build(vaults=reg.vaults, allowlist=allowlist)
    if not router.vaults_by_id:
        raise SystemExit(
            "No workspaces are visible after applying AGENTIC_MEMORY_ALLOWED_WORKSPACES "
            "and enabled flags in the fleet registry. Fix the allowlist or registry entries."
        )

    async def _probe_visible(rec: VaultRecord) -> None:
        try:
            ok = await asyncio.wait_for(
                probe_lightrag_endpoint(router.client, rec.endpoint),
                timeout=5.0,
            )
        except TimeoutError:
            ok = False
        if not ok:
            _LOG.warning(
                "Startup probe: workspace %r at %s is unreachable; "
                "verify_server_health and queries may report errors until LightRAG is up.",
                rec.id,
                str(rec.endpoint),
            )

    await asyncio.gather(*(_probe_visible(v) for v in router.vaults_by_id.values()))
    return router, reg, allowlist


def build_mcp(router: Router) -> FastMCP:
    preamble = tool_preamble(router)
    mcp = FastMCP("agentic-memory")

    @mcp.tool(
        name="query_knowledge_graph",
        description=(
            f"Read-path LightRAG query over HTTP (POST /query). "
            f"The ``limit`` argument caps the maximum serialized JSON returned to the client "
            f"(roughly ``limit`` × 400 characters); it does not set LightRAG recall/top_k. "
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
        if limit <= 0:
            return json.dumps(
                {"error": "invalid_limit", "detail": "`limit` must be a positive integer."},
                indent=2,
            )
        ws: str
        try:
            ws = router.resolve_workspace(workspace)
        except (ValueError, WorkspaceLookupError) as exc:
            return str(exc)
        try:
            status, data = await router.query_lightrag(
                workspace_id=ws,
                prompt=prompt,
                search_mode=search_mode,
                limit=limit,
                context_only=context_only,
                prompt_only=prompt_only,
            )
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            log_call(
                workspace=ws,
                tool="query_knowledge_graph",
                prompt=prompt,
                latency_ms=latency,
                http_status=None,
                result_size=0,
            )
            return json.dumps({"error": "query_failed", "detail": str(exc)}, indent=2)
        latency = (time.perf_counter() - t0) * 1000
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        log_call(
            workspace=ws,
            tool="query_knowledge_graph",
            prompt=prompt,
            latency_ms=latency,
            http_status=status,
            result_size=len(payload),
        )
        return payload

    @mcp.tool(
        name="get_graph_metadata",
        description=f"Return LightRAG /health JSON for a workspace. {preamble}",
    )
    async def get_graph_metadata(workspace: str | None = None) -> str:
        t0 = time.perf_counter()
        try:
            ws = router.resolve_workspace(workspace)
        except (ValueError, WorkspaceLookupError) as exc:
            return str(exc)
        try:
            status, data = await router.get_health_json(ws)
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            log_call(
                workspace=ws,
                tool="get_graph_metadata",
                prompt=None,
                latency_ms=latency,
                http_status=None,
                result_size=0,
            )
            return json.dumps({"error": "metadata_failed", "detail": str(exc)}, indent=2)
        latency = (time.perf_counter() - t0) * 1000
        body = {"workspace": ws, "http_status": status, "health": data}
        text = json.dumps(body, ensure_ascii=False, indent=2)
        log_call(
            workspace=ws,
            tool="get_graph_metadata",
            prompt=None,
            latency_ms=latency,
            http_status=status,
            result_size=len(text),
        )
        return text

    @mcp.tool(
        name="verify_server_health",
        description=(
            f"Probe LightRAG HTTP health for one workspace or all visible workspaces when "
            f'workspace is "*". {preamble}'
        ),
    )
    async def verify_server_health(workspace: str = "*") -> str:
        t0 = time.perf_counter()
        targets: list[str]
        if workspace.strip() == "*":
            targets = router.visible_workspaces()
        else:
            try:
                targets = [router.resolve_workspace(workspace)]
            except (ValueError, WorkspaceLookupError) as exc:
                return str(exc)

        async def _probe_row(ws_id: str) -> dict[str, Any]:
            try:
                ok = await asyncio.wait_for(
                    probe_lightrag_endpoint(router.client, router.vaults_by_id[ws_id].endpoint),
                    timeout=5.0,
                )
            except TimeoutError:
                ok = False
            return {
                "workspace": ws_id,
                "reachable": ok,
                "endpoint": str(router.vaults_by_id[ws_id].endpoint),
            }

        rows = list(await asyncio.gather(*(_probe_row(ws) for ws in targets)))
        latency = (time.perf_counter() - t0) * 1000
        text = json.dumps({"workspaces": rows}, indent=2)
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
            f"Return /health payload as coarse indexing/server status (LightRAG-specific fields "
            f"vary by version). {preamble}"
        ),
    )
    async def check_indexing_status(workspace: str | None = None) -> str:
        t0 = time.perf_counter()
        try:
            ws = router.resolve_workspace(workspace)
        except (ValueError, WorkspaceLookupError) as exc:
            return str(exc)
        try:
            status, data = await router.get_health_json(ws)
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            log_call(
                workspace=ws,
                tool="check_indexing_status",
                prompt=None,
                latency_ms=latency,
                http_status=None,
                result_size=0,
            )
            return json.dumps({"error": "indexing_status_failed", "detail": str(exc)}, indent=2)
        body = {
            "workspace": ws,
            "http_status": status,
            "pipeline_status": data if isinstance(data, dict) else {"raw": data},
        }
        latency = (time.perf_counter() - t0) * 1000
        text = json.dumps(body, indent=2)
        log_call(
            workspace=ws,
            tool="check_indexing_status",
            prompt=None,
            latency_ms=latency,
            http_status=status,
            result_size=len(text),
        )
        return text

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
                }
            )
        latency = (time.perf_counter() - t0) * 1000
        text = json.dumps(out, indent=2)
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
