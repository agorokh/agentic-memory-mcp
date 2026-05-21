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
- **Pooled HTTP.** A single shared `httpx.AsyncClient` handles all workspaces (tunable via env; see below).

## Tools exposed via MCP

| Tool | What it does | LightRAG endpoint |
|---|---|---|
| `query_knowledge_graph` | Query with `mix`, `semantic`, `keyword`, `global`, `hybrid`, `local`, or `naive` modes. Returns JSON with `http_status`, `ok`, and `result`. | `POST /query` |
| `verify_server_health` | Reachability check (`workspace="*"` probes all visible workspaces). | `GET /health` / `GET /` |
| `get_graph_metadata` | Per-workspace `/health` payload. | `GET /health` |
| `check_indexing_status` | Same `/health` payload with operator-friendly labels. | `GET /health` |
| `list_workspaces` | Registry metadata after allowlist filtering (`query_supported` flag per row). | (registry-only) |

### `query_knowledge_graph` parameters

| Parameter | Default | Notes |
|---|---|---|
| `prompt` | (required) | Forwarded to LightRAG; max 32 000 characters. |
| `workspace` | auto when one visible | Required when multiple workspaces are visible. |
| `search_mode` | `mix` | `semantic` → LightRAG `local`; `keyword` → `naive`. |
| `limit` | `60` | Caps **serialized MCP JSON** (~`limit × 400` chars), not LightRAG `top_k`. |
| `context_only` / `prompt_only` | `false` | Map to LightRAG `only_need_context` / chunk inclusion. |

No `set_active_workspace` — every tool call passes `workspace` explicitly.

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Copy and edit the example registry (schema v2)
cp examples/fleet_registry.example.toml ./fleet_registry.toml
$EDITOR fleet_registry.toml

# Localhost/private IPs require:
export AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS=1

export AGENTIC_MEMORY_REGISTRY_PATH=$PWD/fleet_registry.toml

# 3. (Optional) allowlist a subset for this host
export AGENTIC_MEMORY_ALLOWED_WORKSPACES=my_workspace_a,my_workspace_b

# 4. Run the MCP server (stdio)
agentic-memory-mcp
# or: python -m agentic_memory.server
```

Connect from Claude Code by adding to `.mcp.json`:

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "agentic-memory-mcp",
      "env": {
        "AGENTIC_MEMORY_REGISTRY_PATH": "/abs/path/to/fleet_registry.toml",
        "AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS": "1"
      }
    }
  }
}
```

## Environment variables

| Variable | Required | Meaning |
|---|---|---|
| `AGENTIC_MEMORY_REGISTRY_PATH` | yes | Path to `fleet_registry.toml` (prefer absolute). |
| `AGENTIC_MEMORY_ALLOWED_WORKSPACES` | no | Comma-separated workspace IDs. Empty = all **enabled** rows. |
| `AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS` | no | `1` / `true` to allow loopback and RFC1918 endpoints in the registry. |
| `AGENTIC_MEMORY_LOG_PROMPTS` | no | Log raw prompts in audit JSON (avoid in shared log stacks). |
| `AGENTIC_MEMORY_LOG_LEVEL` | no | Python log level (default `INFO`). |
| `AGENTIC_MEMORY_HTTP_MAX_CONNECTIONS` | no | Shared httpx pool size (default `64`). |
| `AGENTIC_MEMORY_HTTP_MAX_KEEPALIVE` | no | Keepalive connections (default `16`). |
| `AGENTIC_MEMORY_QUERY_READ_TIMEOUT_S` | no | Read timeout for upstream HTTP (default `120`). |
| `AGENTIC_MEMORY_MAX_PROBE_CONCURRENCY` | no | Parallelism for startup / `verify_server_health("*")` (default `8`). |
| `AGENTIC_MEMORY_JSON_PRETTY` | no | Pretty-print MCP tool JSON when set truthy. |

## Fleet registry schema (v2)

Supported reader versions: `"1"` and `"2"`. Writers should emit `schema_version = "2"`.

```toml
schema_version = "2"

[[vaults]]
id = "my_workspace_a"
endpoint = "http://localhost:8020"
backend = "lightrag"          # only lightrag is queryable today
enabled = true
allowed_modes = ["mix", "hybrid", "semantic"]   # optional; default = all modes
origin = "repo-product"       # optional
graph_namespace = "..."       # optional; for graphiti metadata only
```

- **`backend = "graphiti"`** rows are listed in `list_workspaces` but **cannot be queried** until a Graphiti read path exists.
- **`allowed_modes`** restricts `search_mode` per workspace when set; an empty list denies all modes.
- Endpoints must not embed credentials. Metadata hosts (e.g. cloud metadata URLs) are blocked.

A separate upstream pipeline is expected to materialise this file from your fleet declaration.

## Trust model

- **Read-only bridge** — only `POST /query` and `GET /health` on configured endpoints.
- **Host allowlist** — `AGENTIC_MEMORY_ALLOWED_WORKSPACES` limits visible workspaces per process.
- **Audit logging** — structured JSON on stderr; prompts hashed unless `AGENTIC_MEMORY_LOG_PROMPTS` is set.

### Operational security

- **No MCP or upstream HTTP authentication** — isolate the bridge process and restrict who can edit `fleet_registry.toml`.
- **Registry endpoints are capability URLs** — the bridge will fetch whatever URL is configured (SSRF risk). Use network egress controls in production; set `AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS` only for local dev.
- **Tool responses may include internal URLs** from `list_workspaces` / health tools.

See [SECURITY.md](SECURITY.md) for coordinated disclosure and hardening notes.

## Why this exists

Most agentic-memory bridges either lock you to one backend per process, or expose write capabilities on the read path. This bridge is deliberately the opposite: one process serves many workspaces, workspace is a per-call parameter, and the write path stays out of the agent's reach.

## Development

```bash
pip install -e ".[dev]"
pytest --cov=agentic_memory --cov-fail-under=73 -v
ruff check src tests
```

## Companion projects

- [**sdlc-dial-adapter**](https://github.com/agorokh/sdlc-dial-adapter) — Anthropic Messages API → OpenAI chat-completions translator.
- [**applied-ai-research**](https://github.com/agorokh/applied-ai-research) — practitioner notes citing this bridge.

## License

[Apache-2.0](LICENSE). Patent grant included.
