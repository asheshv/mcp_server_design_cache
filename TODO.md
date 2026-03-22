# TODO

## Features
- [x] Add `update_note` tool to seamlessly edit existing cache content.
- [x] Add `delete_note` tool allowing agents to permanently delete specific cache IDs.
- [x] Implement pagination and offset parameters for `search_design` and `get_recent_activity`.
- [x] Add an explicit `tags[]` array column to notes for better categorization.
- [x] Add `get_retention_policies` tool so agents can read active cleanup rules for a project.

## Bugs
- [x] Fix `run_smart_cleanup` to execute a `DELETE` query rather than just a dry-run `SELECT COUNT`.
- [x] Add empty array validation in `summarize_and_cleanup` to prevent Postgres execution errors.
- [x] Ensure `generate_adr_from_cache` actually persists the provided `status` parameter to the cache.
- [x] Prevent LLM context overflow in `export_project_to_markdown` by forcing a file write for massive projects.
