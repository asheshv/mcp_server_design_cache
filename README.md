# 🧠 Design Cache MCP Server (Postgres + Psycopg 3)

A high-performance Model Context Protocol (MCP) server acting as "external memory" for AI agents. It caches design conversations in a hierarchical structure to save tokens and maintain project context.

## 🚀 Features
- **Token Efficiency**: Offloads design history to PostgreSQL so the AI only pulls what it needs.
- **Hierarchical Caching**: Separate storage for high-level project goals and granular idea discussions.
- **Full-Text Search (FTS)**: Sub-millisecond retrieval using Postgres GIN indexes.
- **Secure by Design**: Role-Based Access Control (RBAC), SQL injection validation, and rate limiting (60 RPM).
- **Modern Backend**: Built with Python Asyncio and Psycopg 3.

---

## 🛠️ Prerequisites
- **Docker Desktop** (Recommended)
- **Python 3.10+** (If running without Docker)
- **PostgreSQL 15+** (If running without Docker)

---

## 📦 One-Click Deployment (Docker)

1. **Prepare Folder**: Ensure `server.py`, `init.sql`, `Dockerfile`, `requirements.txt`, and `docker-compose.yml` are in the same directory.
2. **Launch Services**:
   ```bash
   docker-compose up -d --build
   ```
3. **Verify Health**:
   ```bash
   docker exec -it mcp_server python -c "import psycopg; print('Database Driver: Ready')"
   ```

## 🤖 Configuring AI Agents

1. **Claude Code (CLI)**

   Add the server to your global MCP config:

   ```bash
   claude mcp add design-cache --command docker --args ["exec", "-i", "mcp_server", "python", "server.py"]
   ```

2. **Cursor IDE / Windsurf**

   - Go to **Settings > Features > MCP**.
   - Click + **Add new MCP server**.
   - **Name**: `design-cache` | **Type**: `stdio`.
   - **Command**: `docker exec -i mcp_server python server.py`

3. **Claude Desktop (macOS/Windows)**

   Edit your `claude_desktop_config.json`:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `C:\Users\<username>\AppData\Roaming\Claude\claude_desktop_config.json`

   Add this entry to your mcpServers list:

   ```json
   "design-cache": {
      "command": "docker",
      "args": ["exec", "-i", "mcp_server", "python", "server.py"]
   }
   ```

## 🔧 Available Tools
- **search_design**: Finds relevant snippets using keyword/full-text search.
- **store_note**: Saves a new design decision or idea to the cache.
- **summarize_and_cleanup**: Merges multiple ideas into a project-level summary.
- **health_check**: Verifies the connection pool and database availability.

## 🛡️ Security Notes
1. **Least Privilege**: Uses design_readonly for searches and design_readwrite for modifications.
2. **SQL Sanitization**: All inputs scanned for patterns like UNION SELECT or ; DROP TABLE.
3. **Rate Limiting**: Capped at 60 requests per minute.
4. **Connection Lifecycle**: Connections recycled every 30 minutes via Psycopg 3.
