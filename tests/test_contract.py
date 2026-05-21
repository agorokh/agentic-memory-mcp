from __future__ import annotations

from pathlib import Path

import pytest
from agentic_memory.registry import load_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_REGISTRY = REPO_ROOT / "examples" / "fleet_registry.example.toml"


def test_example_fleet_registry_loads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
    reg = load_registry(EXAMPLE_REGISTRY)
    assert reg.schema_version in {"1", "2"}
    enabled = [v for v in reg.vaults if v.enabled]
    assert len(enabled) >= 2
    assert all(v.allowed_modes for v in enabled)
