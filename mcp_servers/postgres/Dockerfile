FROM python:3.11-slim

WORKDIR /app

COPY mcp_servers/postgres/requirements.txt /tmp/mcp-requirements.txt
RUN pip install --no-cache-dir -r /tmp/mcp-requirements.txt

COPY mcp_servers /app/mcp_servers

ENV PYTHONPATH=/app
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8787

EXPOSE 8787

CMD ["python", "-m", "mcp_servers.postgres", "--transport", "streamable-http"]
