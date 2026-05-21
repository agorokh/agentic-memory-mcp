FROM python:3.12-slim

# Run as a non-root user for stdio MCP transports.
RUN useradd --create-home --shell /bin/bash mcp
USER mcp
WORKDIR /home/mcp/app

# Install the package + runtime deps.
COPY --chown=mcp:mcp pyproject.toml ./
COPY --chown=mcp:mcp src ./src
COPY --chown=mcp:mcp README.md LICENSE ./
RUN pip install --no-cache-dir --user .

ENV PATH="/home/mcp/.local/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# AGENTIC_MEMORY_REGISTRY_PATH must be provided at runtime — mount the
# fleet_registry.toml into the container and set the env var to its path.
# Example:
#   docker run --rm -i \
#     -v $PWD/fleet_registry.toml:/etc/agentic-memory/fleet_registry.toml:ro \
#     -e AGENTIC_MEMORY_REGISTRY_PATH=/etc/agentic-memory/fleet_registry.toml \
#     ghcr.io/agorokh/agentic-memory-mcp:latest

ENTRYPOINT ["python", "-m", "agentic_memory.server"]
