# agentic-memory-mcp

[![CI](https://github.com/agorokh/agentic-memory-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/agorokh/agentic-memory-mcp/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![MCP](https://img.shields.io/badge/MCP-1.12%2B-green.svg)](https://modelcontextprotocol.io/)

**A read-path MCP server that exposes [LightRAG](https://github.com/HKUDS/LightRAG) backends to MCP-compatible agents** (Claude Code, Cursor, Claude Desktop, any MCP client). Workspace-aware, with per-host allowlists and a registry-driven multi-tenant model so one bridge process can serve many LightRAG instances.

Used by the [Choosing memory for enterprise agents](https://agorokh.github.io/applied-ai-research/2026-05-19_choosing-memory-for-enterprise-agents/) study as the live read path against the SDLC-history corpus.

```
┌─────────────────────┐      stdio MCP        ┌─────────────────────┐    HTTP /query     ┌─────────────────┐
│  MCP client         │──────────────────────▶│  agentic-memory-mcp │──────────────────▶│  LightRAG       │
│  (Claude Code etc.) │                       │  (this server)      │                   │  workspace A    │
└─────────────────────┘                       │                     │                   └─────────────────┘
                                              │  fleet_registry.toml│                   ┌─────────────────┐
                                              │  + allowlist        │──────────────────▶│  LightRAG       │
                                              │                     │                   │  workspace B    │
                                              └─────────────────────┘                   └─────────────────┘
```

## What it does

- **Read-only.** No writes to the LightRAG backend (ingest stays in your separate ingest pipeline). The bridge only calls `POST /query` and `GET /health`.
- **Workspace-aware.** Each MCP tool call carries a `workspace` parameter; the bridge looks up the workspace's HTTP endpoint from a fleet registry TOML file.
- **Allowlist per host.** `AGENTIC_MEMORY_ALLOWED_WORKSPACES` env var limits which workspaces this process can route to. Different host machines see different workspace subsets.
- **Audit trail.** Every tool call is logged (prompt-redacted by default; `AGENTIC_MEMORY_LOG_PROMPTS=1` to include).
- **Pooled HTTP.** A single shared `httpx.AsyncClient` handles all workspaces.

## Tools exposed via MCP

| Tool | What it does | LightRAG endpoint |
|---|---|---|
| `query_knowledge_graph` | Run a query in any of `mix`, `semantic`, `keyword`, `global`, `hybrid`, `local`, `naive` modes. | `POST /query` |
| `verify_server_health` | Reachability check on a workspace endpoint (or `workspace="*"` to probe every visible one). | `GET /health` / `GET /` |
| `get_graph_metadata` | Per-workspace `/health` payload, annotated with the workspace ID. | `GET /health` |
| `check_indexing_status` | Same `/health` payload with operator-friendly labels (fields vary by LightRAG version). | `GET /health` |
| `list_workspaces` | Returns enabled workspaces visible to this process after allowlist filtering. | (none — registry-only) |

No `set_active_workspace` — every tool call passes `workspace` explicitly, so a single MCP client session can fan out across multiple workspaces.

## Quick start

```bash
# 1. Install
pip install -e .   # (PyPI publish pending; use editable install from a clone)

# 2. Point at a fleet registry. See examples/fleet_registry.example.toml for the schema.
cp examples/fleet_registry.example.toml ./fleet_registry.toml
$EDITOR fleet_registry.toml  # set workspace ids + endpoint URLs

export AGENTIC_MEMORY_REGISTRY_PATH=$PWD/fleet_registry.toml

# 3. (Optional) allowlist a subset for this host
export AGENTIC_MEMORY_ALLOWED_WORKSPACES=my_workspace_a,my_workspace_b

# 4. Run the MCP server (stdio)
python -m agentic_memory.server
```

Connect from Claude Code by adding to `.mcp.json`:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "python",
      "args": ["-m", "agentic_memory.server"],
      "env": {
        "AGENTIC_MEMORY_REGISTRY_PATH": "/abs/path/to/fleet_registry.toml"
      }
    }
  }
}
```

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `AGENTIC_MEMORY_REGISTRY_PATH` | yes | Absolute path to a `fleet_registry.toml` (see `examples/fleet_registry.example.toml`). |
| `AGENTIC_MEMORY_ALLOWED_WORKSPACES` | no | Comma-separated workspace IDs visible to this process. Empty = all **enabled** registry rows. |
| `AGENTIC_MEMORY_LOG_PROMPTS` | no | `1` / `true` / `yes` / `on` (case-insensitive) → audit logs include raw prompts (avoid in shared logs). |
| `AGENTIC_MEMORY_LOG_LEVEL` | no | Python log level for bridge stderr (default `INFO`). |

## Fleet registry schema

The registry is TOML with schema version `1`. Each `[[vaults]]` row pins one workspace to one HTTP endpoint, with an explicit `enabled` flag, a `backend` discriminator, and an optional `allowed_modes` list. See [`examples/fleet_registry.example.toml`](examples/fleet_registry.example.toml) for a complete commented example.

```toml
schema_version = 1

[[vaults]]
id = "my_workspace_a"
endpoint = "http://localhost:8020"
backend = "lightrag"
enabled = true
allowed_modes = ["mix", "hybrid", "semantic"]
```

A separate upstream pipeline (in your own repo) is expected to materialise this file from your fleet declaration. The bridge does not validate that pipeline; it just consumes the rendered TOML.

## Trust model

- **The bridge is read-only.** It only calls `POST /query` and `GET /health` on the LightRAG endpoints. There is no MCP tool that can write or delete from the backend.
- **The allowlist is host-scoped.** Even if the registry lists 12 workspaces, `AGENTIC_MEMORY_ALLOWED_WORKSPACES` lets you restrict any one host to a 3-workspace subset. Disabled registry rows (`enabled = false`) are never visible regardless of allowlist.
- **Audit logging is on by default.** Prompts are redacted unless explicitly enabled via `AGENTIC_MEMORY_LOG_PROMPTS`. The audit log uses structured JSON; one line per tool call.

## Why this exists

Most agentic-memory bridges either:
1. Lock you to one backend (one workspace per process, hard to scale across projects), or
2. Expose write capabilities on the read path (one bug = corrupted index).

This bridge is deliberately the opposite: one process serves many workspaces, the workspace is a per-call parameter, and the write path stays out of the agent's reach. It is the same shape that the [Choosing memory for enterprise agents](https://agorokh.github.io/applied-ai-research/2026-05-19_choosing-memory-for-enterprise-agents/) study uses, and the [canary harness](https://agorokh.github.io/applied-ai-research/2026-05-19_choosing-memory-for-enterprise-agents/artefacts/canary-harness/) measures retrieval quality against this exact MCP surface.

## Development

```bash
pip install -e ".[dev]"
pytest                # 28 tests (registry + routing + server)
```

## Companion projects

- [**sdlc-dial-adapter**](https://github.com/agorokh/sdlc-dial-adapter) — Anthropic Messages API → OpenAI chat-completions translator; lets Claude Code run against any OpenAI-compatible gateway (including EPAM AI DIAL).
- [**applied-ai-research**](https://github.com/agorokh/applied-ai-research) — practitioner notes that cite this bridge in their methodology.

## License

[Apache-2.0](LICENSE). Patent grant included.
