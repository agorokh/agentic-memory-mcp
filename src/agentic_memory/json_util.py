from __future__ import annotations

import json
import os
from typing import Any

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def tool_json(obj: Any) -> str:
    """Serialize MCP tool output; compact by default for smaller stdio payloads."""
    if os.environ.get("AGENTIC_MEMORY_JSON_PRETTY", "").strip().lower() in _TRUTHY:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def truncate_json_value(data: Any, limit: int) -> Any:
    """Coarse cap on serialized size using ``limit`` as a rough token proxy (~limit × 400 chars)."""
    cap = limit * 400
    try:
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return data
    if len(text) <= cap:
        return data
    lo, hi = 0, min(len(text), cap)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = {"truncated": True, "limit": limit, "preview": text[:mid]}
        serialized = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) <= cap:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return {"truncated": True, "limit": limit, "preview": text[:best]}


def cap_serialized_tool_payload(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    """Ensure the full MCP tool JSON envelope fits within ``limit * 400`` characters."""
    if limit <= 0:
        return payload
    if len(tool_json(payload)) <= limit * 400:
        return payload
    result = payload.get("result")
    lo, hi = 1, limit
    best: dict[str, Any] = {
        **payload,
        "result": {"truncated": True, "limit": limit, "preview": ""},
    }
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = {**payload, "result": truncate_json_value(result, mid)}
        if len(tool_json(candidate)) <= limit * 400:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best
