from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from agentic_memory.registry import REGISTRY_SCHEMA_VERSION, load_registry
from agentic_memory.routing import Router, WorkspaceLookupError, probe_lightrag_endpoint
from agentic_memory.server import tool_preamble


def _mock_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/query":
            body = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={
                    "echo_mode": body.get("mode"),
                    "only_need_context": body.get("only_need_context"),
                },
            )
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


def test_query_lightrag_maps_semantic_mode(tmp_path: Path) -> None:
    async def body() -> None:
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
        reg = load_registry(p)
        transport = _mock_transport()
        async with httpx.AsyncClient(transport=transport) as client:
            router = await Router.build(vaults=reg.vaults, allowlist=None, client=client)
            status, data = await router.query_lightrag(
                workspace_id="w",
                prompt="hello",
                search_mode="semantic",
                limit=1000,
                context_only=False,
                prompt_only=False,
            )
            assert status == 200
            assert isinstance(data, dict)
            assert data.get("echo_mode") == "local"

    asyncio.run(body())


def test_resolve_workspace_defaults_when_single_visible(tmp_path: Path) -> None:
    async def body() -> None:
        p = tmp_path / "fleet_registry.toml"
        p.write_text(
            "\n".join(
                [
                    f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                    "[[vaults]]",
                    'id = "only"',
                    'endpoint = "http://memory.test"',
                    "enabled = true",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        reg = load_registry(p)
        async with httpx.AsyncClient(transport=_mock_transport()) as client:
            router = await Router.build(
                vaults=reg.vaults,
                allowlist=frozenset({"only"}),
                client=client,
            )
            assert router.resolve_workspace(None) == "only"

    asyncio.run(body())


def test_concurrent_queries_share_one_async_client(tmp_path: Path) -> None:
    async def body() -> None:
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
        reg = load_registry(p)
        transport = _mock_transport()
        async with httpx.AsyncClient(transport=transport) as client:
            router = await Router.build(vaults=reg.vaults, allowlist=None, client=client)

            async def one() -> None:
                await router.query_lightrag(
                    workspace_id="w",
                    prompt="a",
                    search_mode="mix",
                    limit=1000,
                    context_only=False,
                    prompt_only=False,
                )

            await asyncio.gather(one(), one())
            assert id(router.client) == id(client)

    asyncio.run(body())


def test_tool_preamble_lists_visible_workspace_ids(tmp_path: Path) -> None:
    async def body() -> None:
        p = tmp_path / "fleet_registry.toml"
        p.write_text(
            "\n".join(
                [
                    f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                    "[[vaults]]",
                    'id = "alpha"',
                    'endpoint = "http://a.test"',
                    "enabled = true",
                    "[[vaults]]",
                    'id = "beta"',
                    'endpoint = "http://b.test"',
                    "enabled = true",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        reg = load_registry(p)
        async with httpx.AsyncClient(transport=_mock_transport()) as client:
            router = await Router.build(
                vaults=reg.vaults,
                allowlist=frozenset({"alpha", "beta"}),
                client=client,
            )
            text = tool_preamble(router)
            assert "alpha" in text
            assert "beta" in text

    asyncio.run(body())


def test_resolve_workspace_rejects_unknown_with_structured_payload(tmp_path: Path) -> None:
    async def body() -> None:
        p = tmp_path / "fleet_registry.toml"
        p.write_text(
            "\n".join(
                [
                    f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                    "[[vaults]]",
                    'id = "allowed"',
                    'endpoint = "http://memory.test"',
                    "enabled = true",
                    "[[vaults]]",
                    'id = "hidden"',
                    'endpoint = "http://other.test"',
                    "enabled = true",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        reg = load_registry(p)
        async with httpx.AsyncClient(transport=_mock_transport()) as client:
            router = await Router.build(
                vaults=reg.vaults,
                allowlist=frozenset({"allowed"}),
                client=client,
            )
            with pytest.raises(WorkspaceLookupError) as exc:
                router.resolve_workspace("hidden")
            assert "workspace_unknown" in str(exc.value)
            assert "hidden" in str(exc.value)

    asyncio.run(body())


def test_query_truncation_keeps_final_json_under_cap(tmp_path: Path) -> None:
    async def body() -> None:
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
        reg = load_registry(p)
        huge = {"blob": "Z" * 8000}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/query":
                return httpx.Response(200, json=huge)
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            router = await Router.build(vaults=reg.vaults, allowlist=None, client=client)
            _status, data = await router.query_lightrag(
                workspace_id="w",
                prompt="q",
                search_mode="mix",
                limit=2,
                context_only=False,
                prompt_only=False,
            )
            assert isinstance(data, dict)
            assert data.get("truncated") is True
            out = json.dumps(data, ensure_ascii=False, indent=2)
            assert len(out) <= 2 * 400

    asyncio.run(body())


def test_probe_requires_success_status() -> None:
    async def body() -> None:
        transport = httpx.MockTransport(
            lambda r: (
                httpx.Response(404, json={"err": "nope"})
                if r.url.path == "/health"
                else httpx.Response(404)
            )
        )
        async with httpx.AsyncClient(transport=transport) as client:
            ok = await probe_lightrag_endpoint(client, "http://probe.test")
            assert ok is False

    asyncio.run(body())
