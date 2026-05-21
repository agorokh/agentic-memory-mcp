# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added

- `allowed_modes` per workspace in fleet registry (enforced on query).
- Endpoint URL validation (block metadata hosts, optional private IP policy via `AGENTIC_MEMORY_ALLOW_PRIVATE_ENDPOINTS`).
- Query tool responses include `workspace`, `http_status`, `ok`, and `result`.
- Graphiti workspaces rejected on LightRAG query/health paths (`unsupported_backend`).
- Audit tests; example registry contract test; SSRF policy tests.
- `SECURITY.md`, env-tunable httpx pool/timeouts, probe concurrency limit.
- `agentic-memory-mcp` console script documented; `py.typed` for type consumers.

### Changed

- Example registry and README aligned to schema v2.
- MCP tool JSON compact by default (`AGENTIC_MEMORY_JSON_PRETTY` for indented output).
- Workspace errors no longer embed host allowlist in client-facing JSON.

### Fixed

- Example `fleet_registry.example.toml` now loads successfully.
- `schema_version = 1` (integer) in TOML coerced to string for validation.

## [0.1.0] - 2026-05-19

Initial public release: read-path MCP bridge to LightRAG HTTP backends.
