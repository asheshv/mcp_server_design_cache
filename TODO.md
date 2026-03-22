# MCP Server: Design Cache

## Done ✅
- [x] **Project Auto-Detection**: Added `.design_cache` local config support.
- [x] **Onboarding Prompt**: Implemented `@mcp.prompt() onboard` for session initialization.
- [x] **Project Context Tool**: Added `get_project_context` for instant status sync.
- [x] **Pagination**: Added `limit` and `offset` to `search_design` and `get_recent_activity`.
- [x] **Tags System**: Implemented `tags[]` array column and intersection filtering.
- [x] **Robust Migrations**: Automated DB schema upgrades on startup via `MIGRATIONS`.
- [x] **Retention Policy Retrieval**: Added `get_retention_policies` tool.
- [x] **CRUD Tools**: Implemented `update_note` (with re-embedding) and `delete_note`.
- [x] **Bug Fixes**: Cleanup query fixed, ADR status persistence added, and Export overflow handled.

## Upcoming / Ideas 💡
- [x] **Tag Logic Expansion**: Support `OR` and `NOT` logic for tag search.
- [ ] **Batch Storage**: Implement `batch_store_notes` for bulk data ingestion.
- [x] **Adaptive Compression**: Automate `summarize_and_cleanup` based on token usage patterns.
- [ ] **Search Web Sync**: Tool to fetch external documentation and store in the cache directly.
- [ ] **HNSW Indexing**: Explore HNSW for faster vector search on massive datasets.
- [ ] **Export to PDF/DOCX**: Enhancing the project export tool.
