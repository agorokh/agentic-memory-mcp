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
