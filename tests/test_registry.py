from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import httpx
import pytest
from agentic_memory.registry import (
    REGISTRY_SCHEMA_VERSION,
    FleetRegistry,
    allowed_modes_for,
    apply_allowlist,
    effective_backend,
    effective_graph_namespace,
    load_registry,
    parse_allowlist,
)
from agentic_memory.routing import Router


def test_parse_allowlist_none_and_empty() -> None:
    assert parse_allowlist(None) is None
    assert parse_allowlist("") is None
    assert parse_allowlist("  ,  , ") is None


def test_parse_allowlist_nonempty() -> None:
    assert parse_allowlist(" a , b ") == frozenset({"a", "b"})


def test_load_registry_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                'schema_version = "1"',
                "",
                "[[vaults]]",
                'id = "agent_factory"',
                'endpoint = "http://127.0.0.1:8020/"',
                'vault_root = "/tmp/vault"',
                "enabled = true",
                "",
                "[[vaults]]",
                'id = "financial"',
                'endpoint = "http://127.0.0.1:8120/"',
                "enabled = false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    reg = load_registry(p)
    assert len(reg.vaults) == 2
    assert reg.vaults[0].id == "agent_factory"


def test_registry_rejects_unknown_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    p = tmp_path / "bad.toml"
    p.write_text(
        "\n".join(
            [
                'schema_version = "1"',
                "[[vaults]]",
                'id = "x"',
                'endpoint = "http://127.0.0.1:1/"',
                "enabled = true",
                "extra_field = 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid fleet registry"):
        load_registry(p)


def test_registry_rejects_bad_schema_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                'schema_version = "99"',
                "[[vaults]]",
                'id = "x"',
                'endpoint = "http://127.0.0.1:1/"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported registry schema_version"):
        load_registry(p)


def test_router_rejects_duplicate_workspace_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                "[[vaults]]",
                'id = "dup"',
                'endpoint = "http://127.0.0.1:1/"',
                "enabled = true",
                "[[vaults]]",
                'id = "dup"',
                'endpoint = "http://127.0.0.1:2/"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    reg = load_registry(p)

    async def body() -> None:
        transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True}))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ValueError, match="Duplicate workspace id"):
                await Router.build(vaults=reg.vaults, allowlist=None, client=client)

    asyncio.run(body())


def test_v1_manifest_still_loads_and_defaults_backend_to_lightrag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                'schema_version = "1"',
                "[[vaults]]",
                'id = "legacy_ws"',
                'endpoint = "http://127.0.0.1:8020/"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    reg = load_registry(p)
    rec = reg.vaults[0]
    assert rec.backend is None
    assert rec.origin is None
    assert rec.graph_namespace is None
    assert effective_backend(rec) == "lightrag"
    assert effective_graph_namespace(rec) == "legacy_ws"


def test_v2_manifest_loads_with_backend_and_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                'schema_version = "2"',
                "[[vaults]]",
                'id = "alpaca_trading"',
                'endpoint = "http://m2pro:8100/"',
                'backend = "graphiti"',
                'origin = "repo-embedded"',
                'graph_namespace = "alpaca_trading"',
                "enabled = true",
                "[[vaults]]",
                'id = "divorce_proceedings"',
                'endpoint = "http://127.0.0.1:8060/"',
                'backend = "lightrag"',
                'origin = "repo-product"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    reg = load_registry(p)
    assert len(reg.vaults) == 2
    alpaca, divorce = reg.vaults
    assert effective_backend(alpaca) == "graphiti"
    assert alpaca.origin == "repo-embedded"
    assert effective_graph_namespace(alpaca) == "alpaca_trading"
    assert effective_backend(divorce) == "lightrag"
    assert divorce.origin == "repo-product"


def test_v2_manifest_rejects_unknown_backend(tmp_path: Path) -> None:
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                'schema_version = "2"',
                "[[vaults]]",
                'id = "x"',
                'endpoint = "http://127.0.0.1:1/"',
                'backend = "mem0"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid fleet registry"):
        load_registry(p)


def test_v2_manifest_rejects_unknown_origin(tmp_path: Path) -> None:
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                'schema_version = "2"',
                "[[vaults]]",
                'id = "x"',
                'endpoint = "http://127.0.0.1:1/"',
                'origin = "operator-built"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid fleet registry"):
        load_registry(p)


def test_graph_namespace_defaults_to_id_when_unset() -> None:
    reg = FleetRegistry.model_validate(
        tomllib.loads(
            "\n".join(
                [
                    'schema_version = "2"',
                    "[[vaults]]",
                    'id = "my_ws"',
                    'endpoint = "http://m2pro:8100/"',
                    'backend = "graphiti"',
                    "enabled = true",
                ]
            )
        )
    )
    assert effective_graph_namespace(reg.vaults[0]) == "my_ws"


def test_allowed_modes_empty_list_denies_all() -> None:
    reg = FleetRegistry.model_validate(
        tomllib.loads(
            "\n".join(
                [
                    'schema_version = "2"',
                    "[[vaults]]",
                    'id = "w"',
                    'endpoint = "http://memory.test/"',
                    "enabled = true",
                    "allowed_modes = []",
                ]
            )
        )
    )
    assert allowed_modes_for(reg.vaults[0]) == frozenset()


def test_allowed_modes_defaults_to_all_modes() -> None:
    reg = FleetRegistry.model_validate(
        tomllib.loads(
            "\n".join(
                [
                    'schema_version = "2"',
                    "[[vaults]]",
                    'id = "w"',
                    'endpoint = "http://memory.test/"',
                    "enabled = true",
                ]
            )
        )
    )
    modes = allowed_modes_for(reg.vaults[0])
    assert "mix" in modes
    assert "semantic" in modes


def test_apply_allowlist_filters_disabled_and_allowlist() -> None:
    reg = FleetRegistry.model_validate(
        tomllib.loads(
            "\n".join(
                [
                    f'schema_version = "{REGISTRY_SCHEMA_VERSION}"',
                    "[[vaults]]",
                    'id = "a"',
                    'endpoint = "http://127.0.0.1:1/"',
                    "enabled = true",
                    "[[vaults]]",
                    'id = "b"',
                    'endpoint = "http://127.0.0.1:2/"',
                    "enabled = false",
                    "[[vaults]]",
                    'id = "c"',
                    'endpoint = "http://127.0.0.1:3/"',
                    "enabled = true",
                ]
            )
        )
    )
    out = apply_allowlist(reg.vaults, frozenset({"a", "c"}))
    assert [v.id for v in out] == ["a", "c"]
    out2 = apply_allowlist(reg.vaults, None)
    assert [v.id for v in out2] == ["a", "c"]
