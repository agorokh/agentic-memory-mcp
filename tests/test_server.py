from __future__ import annotations

import asyncio
import logging
import socket
from pathlib import Path

import httpx
import pytest
from agentic_memory.registry import REGISTRY_SCHEMA_VERSION, load_registry
from agentic_memory.routing import Router
from agentic_memory.server import bootstrap_router, build_mcp


def test_bootstrap_router_raises_when_no_visible_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTIC_MEMORY_REGISTRY_PATH", str(p))
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOWED_WORKSPACES", "ghost")

    async def body() -> None:
        await bootstrap_router()

    with pytest.raises(SystemExit) as exc:
        asyncio.run(body())
    assert "No workspaces are visible" in str(exc.value)


def test_startup_probe_emits_warning_for_dead_port(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _host, free_port = sock.getsockname()
    sock.close()
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                "[[vaults]]",
                'id = "dead"',
                f'endpoint = "http://127.0.0.1:{free_port}"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTIC_MEMORY_REGISTRY_PATH", str(p))
    caplog.set_level(logging.WARNING, logger="agentic_memory.server")

    async def body() -> None:
        router, _, _ = await bootstrap_router()
        await router.aclose()

    asyncio.run(body())
    assert any("Startup probe" in r.getMessage() for r in caplog.records)


def test_build_mcp_registers_five_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setenv("AGENTIC_MEMORY_REGISTRY_PATH", str(p))

    async def body() -> None:
        transport = httpx.MockTransport(
            lambda r: (
                httpx.Response(200, json={"ok": True})
                if r.url.path in ("/health", "/")
                else httpx.Response(404)
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            reg = load_registry(p)
            router = await Router.build(
                vaults=reg.vaults,
                allowlist=frozenset({"w"}),
                client=client,
            )
            mcp = build_mcp(router)
            tools = await mcp.list_tools()
            assert len(tools) == 5
            query_tool = next(t for t in tools if t.name == "query_knowledge_graph")
            assert "400" in (query_tool.description or "")
            assert "limit" in (query_tool.description or "").lower()
            desc_blob = " ".join((t.description or "") for t in tools)
            assert "Effective workspace universe" in desc_blob
            assert "Effective workspace universe: w" in (query_tool.description or "")
            await router.aclose()

    asyncio.run(body())
