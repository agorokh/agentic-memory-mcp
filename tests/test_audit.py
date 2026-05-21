from __future__ import annotations

import json

import pytest
from agentic_memory.audit import log_call


def test_log_call_redacts_prompt_by_default(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENTIC_MEMORY_LOG_PROMPTS", raising=False)
    caplog.set_level("INFO", logger="agentic_memory.audit")
    log_call(
        workspace="w",
        tool="query_knowledge_graph",
        prompt="secret",
        latency_ms=1.0,
        http_status=200,
        result_size=10,
    )
    assert len(caplog.records) == 1
    row = json.loads(caplog.records[0].message)
    assert "prompt" not in row
    assert row["prompt_hash"]
    assert row["tool"] == "query_knowledge_graph"


def test_log_call_includes_prompt_when_flag_set(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTIC_MEMORY_LOG_PROMPTS", "true")
    caplog.set_level("INFO", logger="agentic_memory.audit")
    log_call(
        workspace="w",
        tool="query_knowledge_graph",
        prompt="secret",
        latency_ms=1.0,
        http_status=200,
        result_size=10,
    )
    row = json.loads(caplog.records[0].message)
    assert row["prompt"] == "secret"
