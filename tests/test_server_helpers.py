from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from agentic_memory.json_util import tool_error, tool_json
from agentic_memory.registry import REGISTRY_SCHEMA_VERSION, load_registry
from agentic_memory.routing import Router
from agentic_memory.server import _query_tool_payload, build_mcp


def test_query_tool_payload_success() -> None:
    payload = _query_tool_payload("w", 200, {"answer": "ok"})
    assert payload["ok"] is True
    assert payload["http_status"] == 200
    assert "error" not in payload


def test_query_tool_payload_upstream_error() -> None:
    payload = _query_tool_payload("w", 502, {"detail": "bad"})
    assert payload["ok"] is False
    assert payload["error"] == "upstream_http_error"


def test_tool_json_compact_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTIC_MEMORY_JSON_PRETTY", raising=False)
    text = tool_json({"a": 1})
    assert "\n" not in text


def test_tool_error_shape() -> None:
    text = tool_error("invalid_limit", detail="nope")
    data = json.loads(text)
    assert data["error"] == "invalid_limit"


@pytest.mark.asyncio
async def test_query_tool_rejects_invalid_limit(tmp_path: Path) -> None:
    p = _write_registry(tmp_path)
    router = await _router_from_registry(p)
    mcp = build_mcp(router)
    fn = _tool_fn(mcp, "query_knowledge_graph")
    out = await fn(prompt="hi", limit=0)
    data = json.loads(out)
    assert data["ok"] is False
    assert data["http_status"] is None
    assert data["result"] is None
    assert data["error"] == "invalid_limit"
    await router.aclose()


@pytest.mark.asyncio
async def test_query_tool_rejects_unsupported_mode(tmp_path: Path) -> None:
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                "[[vaults]]",
                'id = "w"',
                'endpoint = "http://memory.test"',
                "enabled = true",
                'allowed_modes = ["keyword"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    router = await _router_from_registry(p)
    mcp = build_mcp(router)
    fn = _tool_fn(mcp, "query_knowledge_graph")
    out = await fn(prompt="hi", search_mode="semantic", limit=10)
    data = json.loads(out)
    assert data["ok"] is False
    assert data["http_status"] is None
    assert data["result"] is None
    assert data["error"] == "mode_not_allowed"
    await router.aclose()


@pytest.mark.asyncio
async def test_query_tool_accepts_workspace_id_with_brace_prefix(tmp_path: Path) -> None:
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                "[[vaults]]",
                'id = "{team-a}"',
                'endpoint = "http://memory.test"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    router = await _router_from_registry(p)
    mcp = build_mcp(router)
    fn = _tool_fn(mcp, "query_knowledge_graph")
    out = await fn(prompt="hi", workspace="{team-a}", limit=10)
    data = json.loads(out)
    assert data["ok"] is True
    await router.aclose()


@pytest.mark.asyncio
async def test_query_tool_audits_rejected_mode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                "[[vaults]]",
                'id = "w"',
                'endpoint = "http://memory.test"',
                "enabled = true",
                "allowed_modes = []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    caplog.set_level("INFO", logger="agentic_memory.audit")
    router = await _router_from_registry(p)
    mcp = build_mcp(router)
    fn = _tool_fn(mcp, "query_knowledge_graph")
    await fn(prompt="hi", limit=10)
    assert any("query_knowledge_graph" in r.message for r in caplog.records)
    await router.aclose()


@pytest.mark.asyncio
async def test_query_tool_rejects_empty_allowed_modes(tmp_path: Path) -> None:
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                "[[vaults]]",
                'id = "w"',
                'endpoint = "http://memory.test"',
                "enabled = true",
                "allowed_modes = []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    router = await _router_from_registry(p)
    mcp = build_mcp(router)
    fn = _tool_fn(mcp, "query_knowledge_graph")
    out = await fn(prompt="hi", limit=10)
    data = json.loads(out)
    assert data["error"] == "mode_not_allowed"
    await router.aclose()


@pytest.mark.asyncio
async def test_query_tool_wraps_success_with_http_status(tmp_path: Path) -> None:
    p = _write_registry(tmp_path)
    router = await _router_from_registry(p)
    mcp = build_mcp(router)
    fn = _tool_fn(mcp, "query_knowledge_graph")
    out = await fn(prompt="hi", limit=60)
    data = json.loads(out)
    assert data["ok"] is True
    assert data["http_status"] == 200
    assert "result" in data
    await router.aclose()


def _write_registry(tmp_path: Path) -> Path:
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                "[[vaults]]",
                'id = "w"',
                'endpoint = "http://memory.test"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return p


async def _router_from_registry(path: Path) -> Router:
    reg = load_registry(path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/query":
            return httpx.Response(200, json={"echo": "ok"})
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return await Router.build(vaults=reg.vaults, allowlist=None, client=client)


def _tool_fn(mcp: object, name: str):
    manager = mcp._tool_manager  # type: ignore[attr-defined]
    return manager._tools[name].fn
