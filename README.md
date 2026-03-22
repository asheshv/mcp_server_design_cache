# 🧠 Design Cache MCP Server (Postgres + Psycopg 3)

A high-performance Model Context Protocol (MCP) server acting as "external memory" for AI agents. It caches design conversations in a hierarchical structure to save tokens and maintain project context.

## 🚀 Features
- **Token Efficiency**: Offloads design history to PostgreSQL so the AI only pulls what it needs.
- **Hierarchical Caching**: Separate storage for high-level project goals and granular idea discussions.
- **Full-Text Search (FTS)**: Sub-millisecond retrieval using Postgres GIN indexes.
- **Semantic Vector Search**: Advanced context retrieval using local embeddings (`sentence-transformers`).
- **Secure by Design**: Role-Based Access Control (RBAC), Parameterized Queries, and rate limiting (60 RPM).
- **Modern Backend**: Built with Python Asyncio and Psycopg 3.

---

## 🛠️ Prerequisites
- **Docker Desktop** (Recommended)
- **Python 3.13+** (If running without Docker)
- **PostgreSQL 18+** (If running without Docker)

---

## 📦 Local Deployment (Recommended)

Running the Python server locally allows the AI to natively read and write to your local file system (for linking markdown specs to the cache).

1. **Run Setup Script**:
   ```bash
   chmod +x setup_local.sh
   ./setup_local.sh
   ```
   *(This starts the Postgres database in Docker, creates a Python `venv`, and installs dependencies including `sentence-transformers`.)*

2. **Run Server**:
   ```bash
   source venv/bin/activate
   DB_HOST=localhost python server.py
   ```

## 🤖 Configuring AI Agents

1. **Claude Code (CLI)**

   Add the server to your global MCP config:

   ```bash
   claude mcp add design-cache --command /path/to/venv/bin/python --args ["/path/to/server.py"]
   ```

2. **Cursor IDE / Windsurf**

   - Go to **Settings > Features > MCP**.
   - Click + **Add new MCP server**.
   - **Name**: `design-cache` | **Type**: `stdio`.
   - **Command**: `/path/to/venv/bin/python /path/to/server.py`

3. **Claude Desktop (macOS/Windows)**

   Edit your `claude_desktop_config.json` and add this entry to your mcpServers list:

   ```json
   "design-cache": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/server.py"]
   }
   ```

## 🔧 Available Tools
- **search_design**: Finds relevant snippets using keyword/full-text search.
- **store_note**: Saves a new design decision or idea to the cache.
- **summarize_and_cleanup**: Merges multiple ideas into a project-level summary.
- **health_check**: Verifies the connection pool and database availability.

## 🛡️ Security Notes
1. **Least Privilege**: Uses design_readonly for searches and design_readwrite for modifications.
2. **Parameterized Queries**: Uses `psycopg3` binding to natively prevent SQL injection without blocking valid markdown characters.
3. **Rate Limiting**: Capped at 60 requests per minute.
4. **Connection Lifecycle**: Connections recycled every 30 minutes via Psycopg 3.
