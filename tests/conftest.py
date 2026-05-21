from __future__ import annotations

import socket

import pytest

_TEST_HOSTS = frozenset(
    {"memory.test", "other.test", "probe.test", "a.test", "b.test", "m2pro"},
)


@pytest.fixture(autouse=True)
def _allow_private_endpoints_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests use localhost/memory.test endpoints; production defaults stay strict."""
    monkeypatch.setenv("AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS", "1")


@pytest.fixture(autouse=True)
def _resolve_test_hostnames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve *.test registry hosts to loopback so DNS fail-closed checks can run in CI."""
    real_getaddrinfo = socket.getaddrinfo

    def _getaddrinfo(
        host: str | bytes | None,
        port: str | int | None,
        *args: object,
        **kwargs: object,
    ) -> list[tuple]:
        if isinstance(host, str) and host in _TEST_HOSTS:
            host = "127.0.0.1"
        return real_getaddrinfo(host, port, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket, "getaddrinfo", _getaddrinfo)
