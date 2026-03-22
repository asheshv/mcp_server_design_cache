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

3. **Run Server**:
   ```bash
   source venv/bin/activate
   # Load environment variables from .env
   export $(grep -v '^#' .env | xargs)
   python server.py
   ```

## 🤖 Configuring AI Agents

1. **Claude Code (CLI)**

   Add the server to your global MCP config:

   ```bash
   claude mcp add design-cache --command /path/to/venv/bin/python --args ["/path/to/server.py"]
   ```

2. **AntiGravity / Cursor / Windsurf**

   - Go to **Settings > Features > MCP**.
   - Click + **Add new MCP server**.
   - **Name**: `design-cache` | **Type**: `stdio`.
   - **Command**: `/path/to/venv/bin/python /path/to/server.py`

3. **Claude Desktop (macOS/Windows)**

   Edit your `claude_desktop_config.json` and add this entry to your mcpServers list:

   ```json
   "design-cache": {
      "command": "/path/to/venv/bin/python",
      "args": ["/path/to/server.py"],
      "env": {
        "DB_HOST": "localhost",
        "DB_READ_PASS": "your_password",
        "DB_WRITE_PASS": "your_password"
      }
   }
   ```

## 📖 Usage Examples

Check out the [examples/usage.md](examples/usage.md) file to see how AI agents interact with the cache during brainstorming, context retrieval, and decision formalization.

## 🔧 Available Tools
- **search_design**: Finds relevant snippet abstracts using keyword/full-text search.
- **expand_design_note**: Retrieves the full, unabridged text for a specific cached idea.
- **store_note**: Saves a new design decision or idea to the cache.
- **summarize_and_cleanup**: Merges multiple ideas into a project-level summary.
- **health_check**: Verifies the connection pool and database availability.

## 🛡️ Security Notes
1. **Environment Variables**: Sensitive credentials (passwords) are strictly managed via environment variables and never hardcoded in the repository. Use `.env.example` as a template.
2. **Least Privilege**: Uses `design_readonly` for searches and `design_readwrite` for modifications.
3. **Parameterized Queries**: Uses `psycopg3` binding to natively prevent SQL injection without blocking valid markdown characters.
4. **Rate Limiting**: Capped at 60 requests per minute.
5. **Connection Lifecycle**: Connections recycled every 30 minutes via Psycopg 3.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
