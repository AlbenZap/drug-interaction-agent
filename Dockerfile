FROM python:3.11-slim

# Node.js is required for Phoenix MCP (npx @arizeai/phoenix-mcp@latest subprocess)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get purge -y curl && \
    rm -rf /var/lib/apt/lists/* && \
    npm install -g @arizeai/phoenix-mcp@latest

# Install uv
RUN pip install uv --no-cache-dir

WORKDIR /app

# Install dependencies only (cached — invalidated only when pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code then install the local package
COPY agent/ ./agent/
RUN uv sync --frozen --no-dev

# Pre-compile all Python packages to bytecode so cold start imports are instant
RUN uv run python -m compileall -q /app/.venv/lib 2>/dev/null || true
RUN uv run python -m compileall -q /app/agent 2>/dev/null || true

ENV PORT=8080
EXPOSE 8080

CMD ["uv", "run", "streamlit", "run", "agent/app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
