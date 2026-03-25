# 🧠 Design Cache MCP Server (Postgres + Psycopg 3)

A Model Context Protocol (MCP) server acting as **shared, persistent design memory** for AI agents. It stores design conversations in a hierarchical structure backed by PostgreSQL, enabling cross-session context and semantic search over your project's design history.

## 🚀 Features
- **Cross-Session Persistence**: Design decisions survive beyond a single AI conversation — pick up where you left off days or weeks later.
- **Multi-Agent Sharing**: Use Claude Code, Cursor, Windsurf, or any MCP-compatible tool on the same project with shared design context.
- **Hierarchical Caching**: Separate storage for high-level project goals and granular idea discussions.
- **Hybrid Search**: Full-Text Search (FTS) with GIN indexes + Semantic Vector Search using local embeddings (`sentence-transformers`).
- **Secure by Design**: Role-Based Access Control (RBAC), Parameterized Queries, and rate limiting (60 RPM).
- **Modern Backend**: Built with Python Asyncio and Psycopg 3.

## 🤔 When to Use This

**Good fit:**
- You use **multiple AI tools** on the same project (e.g., Claude Code + Cursor + Windsurf) and need shared design context across them.
- You work in a **team** where multiple people's AI agents need access to the same design decisions.
- You want **semantic search** over past design discussions — finding related decisions by meaning, not just keywords.
- Your projects have **long-running design phases** where decisions accumulate over weeks/months.

**Probably not needed if:**
- You use a **single AI tool** with built-in memory (e.g., Claude Code's memory system, CLAUDE.md files) — the built-in persistence likely covers your needs.
- Your projects are small enough that a few markdown files capture the full design context.

## ⚠️ Honest Notes

This server was originally framed as a "token saving" tool. In practice:

- **MCP tool definitions consume tokens.** This server registers ~18 tools whose schemas are sent with every API call, adding overhead regardless of whether they're used.
- **Retrieved content still lands in the conversation context** and consumes tokens the same way as reading a file would.
- **The search-then-expand pattern** (250-char abstracts first, full content on demand) is a reasonable optimization, but most AI tools already do this naturally with file reads.

The real value is **persistence and sharing**, not token reduction.

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
- **search_design**: Hybrid keyword + semantic search. Returns abstracts to minimize context size.
- **expand_design_note**: Retrieves the full, unabridged text for a specific cached idea.
- **store_note**: Saves a new design decision or idea to the cache.
- **update_note**: Modifies an existing note (re-embeds automatically if content changes).
- **delete_note**: Permanently removes a specific note.
- **summarize_and_cleanup**: Merges multiple ideas into a project-level summary and deletes originals.
- **get_recent_activity**: Lists recent notes with optional tag filtering.
- **get_project_context**: Auto-detects project from `.design_cache` file and shows status.
- **export_project_to_markdown**: Exports full design history as a Markdown document.
- **generate_spec_from_cache**: Transforms a cached idea into a technical spec template.
- **generate_adr_from_cache**: Transforms a cached idea into an ADR template.
- **sync_doc_status**: Links a physical file to a cache entry and updates its status.
- **link_external_file_to_cache**: Links a local Markdown file to a cache entry.
- **set_retention_policy**: Configures cleanup policy per project.
- **get_retention_policies**: Shows active retention policies.
- **run_smart_cleanup**: Deletes expired notes based on retention policies.
- **get_compression_opportunities**: Analyzes note density and suggests summarization targets.
- **health_check**: Verifies database connection health and pgvector availability.

## 🛡️ Security Notes
1. **Environment Variables**: Sensitive credentials (passwords) are strictly managed via environment variables and never hardcoded in the repository. Use `.env.example` as a template.
2. **Least Privilege**: Uses `design_readonly` for searches and `design_readwrite` for modifications.
3. **Parameterized Queries**: Uses `psycopg3` binding to natively prevent SQL injection without blocking valid markdown characters.
4. **Rate Limiting**: Capped at 60 requests per minute.
5. **Connection Lifecycle**: Connections recycled every 5 minutes via Psycopg 3.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
