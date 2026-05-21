from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

_LOG = logging.getLogger("agentic_memory.audit")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


_TRUEY_LOG_PROMPTS = frozenset({"1", "true", "yes", "on"})


def _env_flag_truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in _TRUEY_LOG_PROMPTS


def log_call(
    *,
    workspace: str,
    tool: str,
    prompt: str | None,
    latency_ms: float,
    http_status: int | None,
    result_size: int,
) -> None:
    """Emit one structured JSON line per MCP tool call."""
    log_prompts = _env_flag_truthy(os.environ.get("AGENTIC_MEMORY_LOG_PROMPTS"))
    row: dict[str, Any] = {
        "ts": time.time(),
        "workspace": workspace,
        "tool": tool,
        "latency_ms": round(latency_ms, 2),
        "http_status": http_status,
        "result_size": result_size,
    }
    if prompt is None:
        row["prompt_hash"] = ""
    else:
        row["prompt_hash"] = _prompt_hash(prompt)
    if log_prompts and prompt is not None:
        row["prompt"] = prompt
    _LOG.info("%s", json.dumps(row, ensure_ascii=False))
