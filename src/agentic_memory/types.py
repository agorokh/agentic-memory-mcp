from __future__ import annotations

from typing import Literal

SearchMode = Literal["mix", "semantic", "keyword", "global", "hybrid", "local", "naive"]

ALL_SEARCH_MODES: frozenset[str] = frozenset(
    {"mix", "semantic", "keyword", "global", "hybrid", "local", "naive"}
)

MAX_TOOL_LIMIT = 500
MAX_PROMPT_CHARS = 32_000
