from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _allow_private_endpoints_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests use localhost/memory.test endpoints; production defaults stay strict."""
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")
