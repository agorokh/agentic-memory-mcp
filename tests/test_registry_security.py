from __future__ import annotations

from pathlib import Path

import pytest
from agentic_memory.registry import load_registry, validate_endpoint_url


def test_validate_endpoint_blocks_metadata_host() -> None:
    with pytest.raises(ValueError, match="blocked host"):
        validate_endpoint_url("http://metadata.google.internal/")


def test_validate_endpoint_blocks_private_ip_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", raising=False)
    with pytest.raises(ValueError, match="private or link-local"):
        validate_endpoint_url("http://127.0.0.1:8020/")


def test_validate_endpoint_blocks_localhost_hostname_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", raising=False)
    with pytest.raises(ValueError, match="private or link-local"):
        validate_endpoint_url("http://localhost:8020/")


def test_validate_endpoint_allows_private_when_flag_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    assert validate_endpoint_url("http://127.0.0.1:8020/") == "http://127.0.0.1:8020/"


def test_validate_endpoint_requires_hostname() -> None:
    with pytest.raises(ValueError, match="hostname is required"):
        validate_endpoint_url("http:///path")


def test_registry_rejects_credentials_in_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    p = tmp_path / "fleet_registry.toml"
    p.write_text(
        "\n".join(
            [
                'schema_version = "2"',
                "[[vaults]]",
                'id = "x"',
                'endpoint = "http://user:pass@127.0.0.1:8020/"',
                "enabled = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid fleet registry"):
        load_registry(p)
