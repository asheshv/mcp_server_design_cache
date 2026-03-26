# MCP Server: Design Cache

## Done
- [x] **Project Auto-Detection**: `.design_cache` local config support
- [x] **Onboarding Prompt**: `@mcp.prompt() onboard` for session initialization
- [x] **Project Context Tool**: `get_project_context` for instant status sync
- [x] **HNSW Indexing**: High-performance vector search in PG18
- [x] **Adaptive Compression**: `summarize_and_cleanup` based on context density
- [x] **Tag Logic Expansion**: `AND`, `OR`, and `NOT` logic for tag search
- [x] **Pagination**: `limit` and `offset` on `search_design` and `get_recent_activity`
- [x] **Tags System**: `tags[]` array column and intersection filtering
- [x] **Robust Migrations**: Automated DB schema upgrades with advisory locking
- [x] **Retention Policy CRUD**: `set_retention_policy`, `get_retention_policies`
- [x] **Note CRUD**: `update_note` (with re-embedding) and `delete_note`
- [x] **Auto-Compress Cleanup**: `run_smart_cleanup` respects `auto_compress` flag — summarizes or hard-deletes expired notes
- [x] **Bulk Markdown Import**: `import_markdown` splits by `##` headings into separate notes
- [x] **Reciprocal Rank Fusion Search**: Split FTS + vector queries merged via RRF for better index utilization
- [x] **Embedding LRU Cache**: 128-entry TTL cache eliminates repeated `model.encode()` calls
- [x] **ONNX Runtime Support**: Docker image 3GB → 400MB with optional ONNX backend
- [x] **Module Split**: `server.py` split into `config.py`, `db.py`, `embedding.py`, `utils.py`
- [x] **GitHub Actions CI**: Lint (ruff) + test (pytest) + Docker build
- [x] **Integration Tests**: 9 tests against real PostgreSQL + pgvector
- [x] **Security Hardening**: Path traversal prevention, conninfo injection fix, credential fail-fast, symlink rejection, content size limits
- [x] **Transaction Safety**: `FOR UPDATE` locking, explicit transactions, advisory lock on migrations

## Backlog

### High Value
| Task | Description | Effort |
|------|-------------|--------|
| Per-client rate limiting | Key rate limits on MCP `client_id` instead of global per-tool counter | Medium |
| Semantic deduplication | Check vector similarity (>0.95) before `store_note` to warn about duplicates | Low |
| Note versioning | Keep edit history instead of overwriting; `note_history` table with `version` column | Medium |
| Webhook on note changes | Emit events (created/updated/deleted) for external tool integration (Slack, Linear) | High |

### Medium Value
| Task | Description | Effort |
|------|-------------|--------|
| Project archival | `archive_project` tool: export, compress, mark inactive; `unarchive_project` to reverse | Medium |
| Cross-project search | `search_all` tool that searches across all projects with optional project filter list | Low |
| Note relationships | `link_notes(id_a, id_b, relationship)` to build a decision graph between notes | Medium |
| Scheduled cleanup cron | Background task running `run_smart_cleanup` on configurable interval during lifespan | Low |

### Nice to Have
| Task | Description | Effort |
|------|-------------|--------|
| Export to Notion/Confluence | Direct API integration beyond markdown export | High |
| Token budget awareness | `search_design` accepts `max_tokens` param, returns results that fit within budget | Low |
| CLI companion | Standalone `design-cache` CLI (click/typer) wrapping MCP tools for non-AI use | Medium |
| Metrics endpoint | Expose cache hit rate, query latency, note count, storage usage for monitoring | Medium |
