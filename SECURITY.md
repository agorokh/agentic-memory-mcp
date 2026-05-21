# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please report security issues via [GitHub Security Advisories](https://github.com/agorokh/agentic-memory-mcp/security/advisories/new) or a private issue if you do not have advisory access.

Do not open public issues for undisclosed vulnerabilities.

## Threat model (summary)

`agentic-memory-mcp` is a **stdio MCP bridge** that reads a local TOML fleet registry and performs outbound HTTP to configured LightRAG endpoints.

Trust boundaries:

1. **MCP client** — anyone who can attach to the process can invoke tools for all visible workspaces.
2. **Fleet registry file** — controls outbound URLs (SSRF surface). Protect file integrity and permissions (`chmod 600`).
3. **LightRAG upstream** — response bodies are returned to agents (indirect prompt-injection risk).

## Hardening recommendations

- Run one bridge process per trust zone with a minimal `AGENTIC_MEMORY_ALLOWED_WORKSPACES`.
- Do not enable `AGENTIC_MEMORY_LOG_PROMPTS` in shared log aggregators.
- Use TLS endpoints in production; keep `AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS` disabled outside local dev.
- Restrict egress from the bridge host to known LightRAG backends (firewall / NetworkPolicy).
- Treat `list_workspaces` and health tool output as sensitive (may expose internal URLs).

## Known limitations

- No MCP-level authentication.
- No upstream HTTP authentication (add at reverse proxy or extend the bridge).
- `backend = "graphiti"` is metadata-only; queries require `backend = "lightrag"`.
