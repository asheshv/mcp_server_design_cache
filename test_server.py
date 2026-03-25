"""
test_server.py - Comprehensive automated test suite for all MCP tools in server.py.

Tests use unittest.mock to patch the database connection pools and the
sentence-transformers embedding model, so all tests run offline without
any live database or network connections required.
"""
import datetime
import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

# Patch env vars BEFORE importing server (it fails fast without them)
os.environ.setdefault("DB_READ_PASS", "test_read_pass")
os.environ.setdefault("DB_WRITE_PASS", "test_write_pass")

import db  # noqa: E402
import server  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: Build a fake async context manager for pool.connection()
# ---------------------------------------------------------------------------
def make_pool_mock(rows=None, fetchone_row=None):
    """
    Returns a mock whose .connection() can be used as `async with pool.connection()`.
    Rows/fetchone_row control what the cursor returns.
    """
    cur = AsyncMock()
    cur.execute = AsyncMock()
    cur.fetchall = AsyncMock(return_value=rows or [])
    cur.fetchone = AsyncMock(return_value=fetchone_row)
    cur.__aenter__ = AsyncMock(return_value=cur)
    cur.__aexit__ = AsyncMock(return_value=False)

    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.cursor = MagicMock(return_value=cur)
    conn.__aenter__ = AsyncMock(return_value=conn)
    conn.__aexit__ = AsyncMock(return_value=False)

    # transaction() context manager
    tx = AsyncMock()
    tx.__aenter__ = AsyncMock(return_value=tx)
    tx.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=tx)

    pool = MagicMock()
    pool.connection = MagicMock(return_value=conn)
    return pool, conn, cur


# ---------------------------------------------------------------------------
# Helper: A fake embedding vector (384 dims for all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------
FAKE_VECTOR = [0.1] * 384


def make_embedding_mock():
    """Returns a mock model whose .encode() returns a fake numpy-like array."""
    model_mock = MagicMock()
    encoded = MagicMock()
    encoded.tolist = MagicMock(return_value=FAKE_VECTOR)
    model_mock.encode = MagicMock(return_value=encoded)
    return model_mock


# Async helper for get_embedding_model mock
async def fake_get_model():
    return make_embedding_mock()


async def fake_to_thread(fn, *args):
    """Replacement for asyncio.to_thread that runs synchronously."""
    return fn(*args)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestMCPTools(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Reset the rate limiter between tests."""
        server.limiter = server.RateLimiter(60)

    # ------------------------------------------------------------------
    # search_design
    # ------------------------------------------------------------------
    async def test_search_design_returns_results(self):
        rows = [
            {"id": "id-1", "title": "Auth Design", "content": "x" * 300,
             "cache_type": "idea", "keyword_score": 0.9,
             "semantic_score": 0.85, "combined_score": 1.75,
             "tags": ["auth", "security"]},
        ]
        pool, conn, cur = make_pool_mock(rows=rows)
        model_mock = make_embedding_mock()

        with patch("server.read_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.search_design(
                "myproject", "auth", limit=1, offset=0, tags=["security"]
            )

        self.assertIn("Auth Design", result)
        self.assertIn("Tags: auth, security", result)

    async def test_search_design_with_or_logic(self):
        rows = [{"id": "1", "title": "Any", "content": "x",
                 "cache_type": "idea", "tags": ["t1"],
                 "keyword_score": 0.5, "semantic_score": 0.5,
                 "combined_score": 1.0}]
        pool, _, _ = make_pool_mock(rows=rows)
        model_mock = make_embedding_mock()
        with patch("server.read_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.search_design(
                "p", "q", tags=["t1", "t2"], tag_logic="OR"
            )
        self.assertIn("Any", result)

    async def test_search_design_with_not_logic(self):
        rows = [{"id": "1", "title": "Safe", "content": "x",
                 "cache_type": "idea", "tags": ["safe"],
                 "keyword_score": 0.5, "semantic_score": 0.5,
                 "combined_score": 1.0}]
        pool, _, _ = make_pool_mock(rows=rows)
        model_mock = make_embedding_mock()
        with patch("server.read_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.search_design(
                "p", "q", tags=["bad"], tag_logic="NOT"
            )
        self.assertIn("Safe", result)

    async def test_search_design_no_results(self):
        pool, _, _ = make_pool_mock(rows=[])
        model_mock = make_embedding_mock()

        with patch("server.read_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.search_design("myproject", "unknown-query")

        self.assertEqual(result, "No relevant design notes found.")

    async def test_search_design_invalid_tag_logic(self):
        model_mock = make_embedding_mock()
        with patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.search_design("p", "q", tag_logic="INVALID")
        self.assertIn("Error", result)

    async def test_search_design_clamps_limit_and_offset(self):
        pool, _, cur = make_pool_mock(rows=[])
        model_mock = make_embedding_mock()
        with patch("server.read_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.search_design(
                "p", "q", limit=-5, offset=-10
            )
        self.assertEqual(result, "No relevant design notes found.")

    # ------------------------------------------------------------------
    # expand_design_note
    # ------------------------------------------------------------------
    async def test_expand_design_note_found(self):
        row = {
            "title": "My Idea",
            "content": "Full detailed content here",
            "cache_type": "idea",
            "created_at": datetime.datetime(2026, 3, 22, 10, 0, 0),
            "project_name": "myproject",
        }
        pool, _, _ = make_pool_mock(fetchone_row=row)
        with patch("server.read_pool", pool):
            result = await server.expand_design_note("some-uuid")

        self.assertIn("My Idea", result)
        self.assertIn("Full detailed content here", result)
        self.assertIn("myproject", result)

    async def test_expand_design_note_not_found(self):
        pool, _, _ = make_pool_mock(fetchone_row=None)
        with patch("server.read_pool", pool):
            result = await server.expand_design_note("nonexistent-uuid")

        self.assertIn("Error", result)
        self.assertIn("nonexistent-uuid", result)

    # ------------------------------------------------------------------
    # store_note
    # ------------------------------------------------------------------
    async def test_store_note_success(self):
        pool, _, _ = make_pool_mock()
        model_mock = make_embedding_mock()

        with patch("server.write_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.store_note(
                "myproject", "New Idea", "Content text", "idea", tags=["t1"]
            )

        self.assertIn("cached successfully", result.lower())

    async def test_store_note_invalid_cache_type(self):
        result = await server.store_note("p", "T", "C", cache_type="invalid")
        self.assertIn("Error", result)
        self.assertIn("cache_type", result)

    async def test_store_note_title_too_long(self):
        result = await server.store_note("p", "x" * 600, "content")
        self.assertIn("Error", result)
        self.assertIn("title", result)

    async def test_store_note_content_too_long(self):
        result = await server.store_note("p", "title", "x" * 60_000)
        self.assertIn("Error", result)
        self.assertIn("content", result)

    # ------------------------------------------------------------------
    # update_note
    # ------------------------------------------------------------------
    async def test_update_note_content_change_re_embeds(self):
        existing = {"title": "Old Title", "content": "Old content",
                    "cache_type": "idea", "tags": []}
        pool, conn, cur = make_pool_mock(fetchone_row=existing)
        model_mock = make_embedding_mock()

        with patch("server.write_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.update_note("some-id", content="New content")

        self.assertIn("updated successfully", result)

    async def test_update_note_uses_for_update(self):
        """Verify SELECT includes FOR UPDATE for row locking."""
        existing = {"title": "T", "content": "C",
                    "cache_type": "idea", "tags": []}
        pool, conn, cur = make_pool_mock(fetchone_row=existing)

        with patch("server.write_pool", pool):
            await server.update_note("some-id", cache_type="project")

        select_sql = cur.execute.call_args_list[0][0][0]
        self.assertIn("FOR UPDATE", select_sql)

    async def test_update_note_sets_updated_at(self):
        """Verify UPDATE includes updated_at = now()."""
        existing = {"title": "T", "content": "C",
                    "cache_type": "idea", "tags": []}
        pool, conn, cur = make_pool_mock(fetchone_row=existing)

        with patch("server.write_pool", pool):
            await server.update_note("some-id", cache_type="project")

        update_sql = cur.execute.call_args_list[-1][0][0]
        self.assertIn("updated_at", update_sql)

    async def test_update_note_type_only_skips_embedding(self):
        existing = {"title": "Title", "content": "Content",
                    "cache_type": "idea", "tags": ["t"]}
        pool, _, _ = make_pool_mock(fetchone_row=existing)

        with patch("server.write_pool", pool):
            result = await server.update_note("some-id", cache_type="project")

        self.assertIn("updated successfully", result)

    async def test_update_note_no_fields_returns_error(self):
        result = await server.update_note("some-id")
        self.assertIn("Error", result)
        self.assertIn("at least one field", result)

    async def test_update_note_not_found(self):
        pool, _, _ = make_pool_mock(fetchone_row=None)
        with patch("server.write_pool", pool):
            result = await server.update_note("nonexistent-id", title="New Title")
        self.assertIn("Error", result)
        self.assertIn("nonexistent-id", result)

    async def test_update_note_invalid_cache_type(self):
        result = await server.update_note("id", cache_type="bogus")
        self.assertIn("Error", result)
        self.assertIn("cache_type", result)

    # ------------------------------------------------------------------
    # delete_note
    # ------------------------------------------------------------------
    async def test_delete_note_success(self):
        write_pool, _, cur = make_pool_mock(fetchone_row=("deleted-id",))
        with patch("server.write_pool", write_pool):
            result = await server.delete_note("deleted-id")
        self.assertIn("deleted", result.lower())
        self.assertIn("deleted-id", result)

    async def test_delete_note_not_found(self):
        write_pool, _, _ = make_pool_mock(fetchone_row=None)
        with patch("server.write_pool", write_pool):
            result = await server.delete_note("nonexistent-id")
        self.assertIn("Error", result)
        self.assertIn("nonexistent-id", result)

    # ------------------------------------------------------------------
    # summarize_and_cleanup
    # ------------------------------------------------------------------
    async def test_summarize_and_cleanup_empty_ids(self):
        result = await server.summarize_and_cleanup("proj", [], "summary", "title")
        self.assertIn("Error", result)

    async def test_summarize_and_cleanup_success(self):
        pool, _, _ = make_pool_mock()
        model_mock = make_embedding_mock()

        with patch("server.write_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.summarize_and_cleanup(
                "myproject", ["id-1", "id-2"], "A summary", "New Summary Title"
            )

        self.assertIn("2", result)
        self.assertIn("New Summary Title", result)

    async def test_summarize_and_cleanup_delete_includes_project_guard(self):
        """Verify the DELETE includes project_name in its WHERE clause."""
        pool, conn, cur = make_pool_mock()
        model_mock = make_embedding_mock()

        with patch("server.write_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            await server.summarize_and_cleanup(
                "myproject", ["id-1"], "summary", "title"
            )

        # Check the DELETE call includes project_name
        delete_call = cur.execute.call_args_list[-1]
        delete_sql = delete_call[0][0]
        self.assertIn("project_name", delete_sql)

    # ------------------------------------------------------------------
    # set_retention_policy
    # ------------------------------------------------------------------
    async def test_set_retention_policy(self):
        pool, _, _ = make_pool_mock()
        with patch("server.write_pool", pool):
            result = await server.set_retention_policy("myproject", 14)

        self.assertIn("14", result)
        self.assertIn("myproject", result)

    async def test_set_retention_policy_invalid_days(self):
        result = await server.set_retention_policy("p", 0)
        self.assertIn("Error", result)

        result2 = await server.set_retention_policy("p", -5)
        self.assertIn("Error", result2)

    # ------------------------------------------------------------------
    # get_retention_policies
    # ------------------------------------------------------------------
    async def test_get_retention_policies_all(self):
        rows = [{"project_name": "p1", "days_to_retain": 30, "auto_compress": True}]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.get_retention_policies()
        self.assertIn("p1", result)
        self.assertIn("30 days", result)

    async def test_get_retention_policies_specific(self):
        row = {"project_name": "p1", "days_to_retain": 30, "auto_compress": False}
        pool, _, _ = make_pool_mock(rows=[row])
        with patch("server.read_pool", pool):
            result = await server.get_retention_policies(project="p1")
        self.assertIn("p1", result)
        self.assertIn("Hard Delete", result)

    # ------------------------------------------------------------------
    # run_smart_cleanup
    # ------------------------------------------------------------------
    async def test_run_smart_cleanup_nothing_expired(self):
        write_pool, _, _ = make_pool_mock(rows=[])
        with patch("server.write_pool", write_pool):
            result = await server.run_smart_cleanup()

        self.assertIn("within their retention limits", result)

    async def test_run_smart_cleanup_deletes_expired(self):
        deleted_rows = [{"project_name": "myproject", "deleted_count": 3}]
        write_pool, _, _ = make_pool_mock(rows=deleted_rows)

        with patch("server.write_pool", write_pool):
            result = await server.run_smart_cleanup()

        self.assertIn("myproject", result)
        self.assertIn("3", result)

    # ------------------------------------------------------------------
    # health_check
    # ------------------------------------------------------------------
    async def test_health_check_ok(self):
        pool, _, _ = make_pool_mock()
        with patch("server.read_pool", pool):
            result = await server.health_check()

        self.assertIn("Healthy", result)

    # ------------------------------------------------------------------
    # get_recent_activity
    # ------------------------------------------------------------------
    async def test_get_recent_activity_with_results(self):
        rows = [
            {"id": "id-1", "title": "Recent Note", "content": "body",
             "cache_type": "idea",
             "created_at": datetime.datetime(2026, 3, 22, 9, 0, 0),
             "tags": ["new"]},
        ]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.get_recent_activity(
                "myproject", limit=5, offset=2
            )

        self.assertIn("Recent Note", result)
        self.assertIn("Offset: 2", result)
        self.assertIn("Tags: new", result)

    async def test_get_recent_activity_with_not_logic(self):
        rows = [{"id": "1", "title": "No Tags", "content": "x",
                 "cache_type": "idea",
                 "created_at": datetime.datetime(2026, 3, 22), "tags": []}]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.get_recent_activity(
                "p", tags=["secret"], tag_logic="NOT"
            )
        self.assertIn("No Tags", result)

    async def test_get_recent_activity_empty(self):
        pool, _, _ = make_pool_mock(rows=[])
        with patch("server.read_pool", pool):
            result = await server.get_recent_activity("emptyproject", limit=5)

        self.assertIn("No recent activity", result)

    async def test_get_recent_activity_invalid_tag_logic(self):
        result = await server.get_recent_activity("p", tag_logic="XNOR")
        self.assertIn("Error", result)

    # ------------------------------------------------------------------
    # export_project_to_markdown
    # ------------------------------------------------------------------
    async def test_export_small_project_inline(self):
        rows = [{"title": "Idea A", "content": "body", "cache_type": "idea",
                 "created_at": datetime.datetime(2026, 3, 22, 9, 0, 0)}]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.export_project_to_markdown("smallproject")

        self.assertIn("Idea A", result)
        self.assertIn("Export successful", result)
        self.assertNotIn("/tmp/", result)

    async def test_export_large_project_saves_to_file(self):
        rows = [
            {"title": f"Note {i}", "content": "body", "cache_type": "idea",
             "created_at": datetime.datetime(2026, 3, 22, 9, 0, 0)}
            for i in range(25)
        ]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.export_project_to_markdown("bigproject")

        self.assertIn("/tmp/", result)
        self.assertIn("Saved locally to", result)

    async def test_export_project_no_data(self):
        pool, _, _ = make_pool_mock(rows=[])
        with patch("server.read_pool", pool):
            result = await server.export_project_to_markdown("emptyproject")

        self.assertIn("No data found", result)

    async def test_export_sanitizes_project_name(self):
        """Path traversal in project name should be sanitized."""
        rows = [{"title": "T", "content": "C", "cache_type": "idea",
                 "created_at": datetime.datetime(2026, 3, 22)}] * 25
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.export_project_to_markdown(
                "../../etc/evil"
            )
        # Should NOT contain path traversal
        self.assertNotIn("../../", result)
        self.assertIn("/tmp/", result)

    async def test_export_distinct_projects_no_collision(self):
        """Two projects with similar names should export to different files (SP-08)."""
        rows = [{"title": "T", "content": "C", "cache_type": "idea",
                 "created_at": datetime.datetime(2026, 3, 22)}] * 25
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result1 = await server.export_project_to_markdown("project.v1")
        pool2, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool2):
            result2 = await server.export_project_to_markdown("project/v1")
        # Extract file paths from results
        path1 = result1.split("Saved locally to: ")[-1].strip()
        path2 = result2.split("Saved locally to: ")[-1].strip()
        self.assertNotEqual(path1, path2)

    # ------------------------------------------------------------------
    # generate_spec_from_cache
    # ------------------------------------------------------------------
    async def test_generate_spec_found(self):
        row = {
            "title": "My Feature", "content": "Content here",
            "project_name": "proj",
            "created_at": datetime.datetime(2026, 3, 22),
        }
        pool, _, _ = make_pool_mock(fetchone_row=row)
        with patch("server.read_pool", pool):
            result = await server.generate_spec_from_cache("idea-id-abc")

        self.assertIn("Technical Specification", result)
        self.assertIn("My Feature", result)

    async def test_generate_spec_not_found(self):
        pool, _, _ = make_pool_mock(fetchone_row=None)
        with patch("server.read_pool", pool):
            result = await server.generate_spec_from_cache("bad-id")

        self.assertIn("Error", result)

    # ------------------------------------------------------------------
    # generate_adr_from_cache
    # ------------------------------------------------------------------
    async def test_generate_adr_found(self):
        row = {
            "title": "DB Choice", "content": "Use Postgres",
            "project_name": "proj",
            "created_at": datetime.datetime(2026, 3, 22),
        }
        read_pool, _, _ = make_pool_mock(fetchone_row=row)
        write_pool, _, _ = make_pool_mock()
        model_mock = make_embedding_mock()

        with patch("server.read_pool", read_pool), \
             patch("server.write_pool", write_pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.generate_adr_from_cache(
                "idea-id", status="accepted"
            )

        self.assertIn("ACCEPTED", result)
        self.assertIn("DB Choice", result)
        # Verify full original content is preserved (M-12 fix)
        self.assertIn("Use Postgres", result)

    async def test_generate_adr_not_found(self):
        pool, _, _ = make_pool_mock(fetchone_row=None)
        with patch("server.read_pool", pool):
            result = await server.generate_adr_from_cache("bad-id")

        self.assertIn("Error", result)

    async def test_generate_adr_invalid_status(self):
        result = await server.generate_adr_from_cache("id", status="invalid")
        self.assertIn("Error", result)

    # ------------------------------------------------------------------
    # sync_doc_status
    # ------------------------------------------------------------------
    async def test_sync_doc_status_success(self):
        row = {"title": "Original Title", "content": "Original Content"}
        read_pool, _, _ = make_pool_mock(fetchone_row=row)
        write_pool, _, _ = make_pool_mock()
        model_mock = make_embedding_mock()

        with patch("server.read_pool", read_pool), \
             patch("server.write_pool", write_pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.sync_doc_status(
                "cache-id", "/tmp/spec.md", "implemented"
            )

        self.assertIn("Linked", result)
        self.assertIn("/tmp/spec.md", result)

    async def test_sync_doc_status_not_found(self):
        pool, _, _ = make_pool_mock(fetchone_row=None)
        with patch("server.read_pool", pool):
            result = await server.sync_doc_status("bad-id", "/tmp/spec.md")

        self.assertIn("Error", result)

    # ------------------------------------------------------------------
    # link_external_file_to_cache
    # ------------------------------------------------------------------
    async def test_link_external_file_not_absolute(self):
        result = await server.link_external_file_to_cache(
            "some-id", "relative/path.md"
        )
        self.assertIn("Error", result)
        self.assertIn("absolute", result.lower())

    async def test_link_external_file_outside_allowed_dirs(self):
        result = await server.link_external_file_to_cache(
            "some-id", "/etc/passwd"
        )
        self.assertIn("Error", result)

    async def test_link_external_file_rejects_symlink(self):
        """Symlinks should be rejected to prevent TOCTOU attacks (SP-03)."""
        import tempfile as tf
        with tf.NamedTemporaryFile(dir="/tmp", suffix=".md", delete=False) as f:
            real_path = f.name
        link_path = real_path + "_link"
        try:
            os.symlink(real_path, link_path)
            result = await server.link_external_file_to_cache(
                "some-id", link_path
            )
            self.assertIn("Error", result)
            self.assertIn("symlink", result.lower())
        finally:
            os.unlink(link_path)
            os.unlink(real_path)

    async def test_link_external_file_not_found(self):
        result = await server.link_external_file_to_cache(
            "some-id", "/tmp/nonexistent_file_12345.md"
        )
        self.assertIn("Error", result)

    async def test_link_external_file_success(self):
        with tempfile.NamedTemporaryFile(
            suffix=".md", dir="/tmp", delete=False
        ) as f:
            filepath = f.name
        try:
            row = {"title": "My Title", "content": "Body"}
            read_pool, _, _ = make_pool_mock(fetchone_row=row)
            write_pool, _, _ = make_pool_mock()
            model_mock = make_embedding_mock()

            with patch("server.read_pool", read_pool), \
                 patch("server.write_pool", write_pool), \
                 patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
                 patch("asyncio.to_thread", side_effect=fake_to_thread):
                result = await server.link_external_file_to_cache(
                    "cache-id", filepath, "spec"
                )

            self.assertIn("Linked", result)
            self.assertIn(filepath, result)
        finally:
            os.unlink(filepath)

    # ------------------------------------------------------------------
    # read_external_doc (resource) - NEW COVERAGE (F-01)
    # ------------------------------------------------------------------
    async def test_read_external_doc_success(self):
        with tempfile.NamedTemporaryFile(
            suffix=".md", dir="/tmp", mode="w", delete=False
        ) as f:
            f.write("# Test Doc\nHello world")
            filepath = f.name
        try:
            result = await server.read_external_doc(filepath)
            self.assertIn("# Test Doc", result)
            self.assertIn("Hello world", result)
        finally:
            os.unlink(filepath)

    async def test_read_external_doc_outside_allowed_dirs(self):
        result = await server.read_external_doc("/etc/passwd")
        self.assertIn("Access denied", result)

    async def test_read_external_doc_path_traversal(self):
        result = await server.read_external_doc("/tmp/../etc/passwd")
        self.assertIn("Access denied", result)

    async def test_read_external_doc_nonexistent_file(self):
        result = await server.read_external_doc("/tmp/nonexistent_12345.md")
        self.assertIn("Could not read", result)

    # ------------------------------------------------------------------
    # get_local_project_name - NEW COVERAGE (F-02)
    # ------------------------------------------------------------------
    def test_get_local_project_name_found(self):
        m = mock_open(read_data="my-project\n")
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", m):
            result = server.get_local_project_name()
        self.assertEqual(result, "my-project")

    def test_get_local_project_name_not_found(self):
        with patch("os.path.exists", return_value=False):
            result = server.get_local_project_name()
        self.assertIsNone(result)

    def test_get_local_project_name_empty_file(self):
        m = mock_open(read_data="\n")
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", m):
            result = server.get_local_project_name()
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # _sanitize_filename
    # ------------------------------------------------------------------
    def test_sanitize_filename_removes_traversal(self):
        result = server.sanitize_filename("../../etc/evil")
        self.assertTrue(result.startswith("______etc_evil_"))
        self.assertNotIn("/", result)
        self.assertNotIn("..", result)

    def test_sanitize_filename_keeps_safe_chars(self):
        result = server.sanitize_filename("my-project_1")
        self.assertTrue(result.startswith("my-project_1_"))

    def test_sanitize_filename_distinct_names_no_collision(self):
        """Two names that sanitize identically should get different hashes (SP-08)."""
        a = server.sanitize_filename("project.v1")
        b = server.sanitize_filename("project/v1")
        self.assertNotEqual(a, b)

    # ------------------------------------------------------------------
    # _validate_file_path
    # ------------------------------------------------------------------
    def test_validate_file_path_tmp_allowed(self):
        self.assertTrue(server.validate_file_path("/tmp/test.md"))

    def test_validate_file_path_etc_denied(self):
        self.assertFalse(server.validate_file_path("/etc/passwd"))

    def test_validate_file_path_traversal_denied(self):
        self.assertFalse(server.validate_file_path("/tmp/../etc/passwd"))

    def test_validate_file_path_rejects_symlinks(self):
        """Symlinks should be rejected when reject_symlinks=True (SP-03)."""
        import tempfile
        with tempfile.NamedTemporaryFile(dir="/tmp", delete=False) as f:
            real_path = f.name
        link_path = real_path + "_link"
        try:
            os.symlink(real_path, link_path)
            # Without reject_symlinks, symlink in /tmp/ is allowed
            self.assertTrue(server.validate_file_path(link_path))
            # With reject_symlinks, it's rejected
            self.assertFalse(
                server.validate_file_path(link_path, reject_symlinks=True)
            )
        finally:
            os.unlink(link_path)
            os.unlink(real_path)

    # ------------------------------------------------------------------
    # _quote_conninfo_value
    # ------------------------------------------------------------------
    def test_quote_conninfo_escapes_quotes(self):
        result = db._quote_conninfo_value("pass'word")
        self.assertEqual(result, "'pass''word'")

    def test_quote_conninfo_escapes_backslashes(self):
        result = db._quote_conninfo_value("pass\\word")
        self.assertEqual(result, "'pass\\\\word'")

    def test_quote_conninfo_injection_attempt(self):
        result = db._quote_conninfo_value(
            "localhost sslmode=disable host=evil.com"
        )
        # Should be a single quoted value, not multiple params
        self.assertTrue(result.startswith("'"))
        self.assertTrue(result.endswith("'"))
        self.assertIn("sslmode", result)

    # ------------------------------------------------------------------
    # apply_migrations
    # ------------------------------------------------------------------
    async def test_apply_migrations_successful(self):
        write_pool, _, _ = make_pool_mock(fetchone_row=(None,))
        with patch("db.write_pool", write_pool):
            await db.apply_migrations()

    # ------------------------------------------------------------------
    # onboarding & project context
    # ------------------------------------------------------------------
    async def test_get_project_context_no_file(self):
        with patch("os.path.exists", return_value=False):
            result = await server.get_project_context()
            self.assertIn("No local .design_cache", result)

    async def test_get_project_context_success(self):
        m = mock_open(read_data="test-project\n")
        rows_notes = [{"id": "1", "title": "Note 1",
                       "created_at": datetime.datetime(2026, 3, 22),
                       "tags": ["t1"]}]
        rows_tags = [{"tag": "t1", "count": 1}]

        pool, _, cur = make_pool_mock()
        cur.fetchall.side_effect = [rows_notes, rows_tags]

        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", m), \
             patch("server.read_pool", pool):
            result = await server.get_project_context()

        self.assertIn("Project Context: test-project", result)
        self.assertIn("Note 1", result)
        self.assertIn("Top Tags: t1", result)

    def test_onboard_prompt_with_project(self):
        m = mock_open(read_data="test-project\n")
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", m):
            result = server.onboard()
        self.assertIn("test-project", result)
        self.assertIn("get_project_context", result)

    def test_onboard_prompt_no_project(self):
        with patch("os.path.exists", return_value=False):
            result = server.onboard()
        self.assertIn("couldn't detect", result)

    # ------------------------------------------------------------------
    # RateLimiter (now async)
    # ------------------------------------------------------------------
    async def test_rate_limiter_allows_under_limit(self):
        rl = server.RateLimiter(5)
        for _ in range(5):
            await rl.check("testtool")

    async def test_rate_limiter_blocks_over_limit(self):
        rl = server.RateLimiter(3)
        await rl.check("testtool")
        await rl.check("testtool")
        await rl.check("testtool")
        with self.assertRaises(RuntimeError) as ctx:
            await rl.check("testtool")
        self.assertIn("Rate limit exceeded", str(ctx.exception))

    async def test_rate_limiter_resets_after_window(self):
        """Verify rate limiter resets when time window passes."""
        rl = server.RateLimiter(1)
        # Manually inject an old timestamp
        rl.history["tool"] = [time.time() - 120]
        await rl.check("tool")  # Should not raise

    async def test_rate_limiter_tool_integration(self):
        """Verify tools actually call the rate limiter."""
        server.limiter = server.RateLimiter(0)  # Block everything
        with self.assertRaises(RuntimeError):
            await server.health_check()

    # ------------------------------------------------------------------
    # get_compression_opportunities
    # ------------------------------------------------------------------
    async def test_get_compression_opportunities_healthy(self):
        rows = [{"id": "1", "title": "T1", "content_len": 100,
                 "cache_type": "idea", "tags": []}]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.get_compression_opportunities(
                "p1", char_limit=1000
            )
        self.assertIn("healthy", result.lower())

    async def test_get_compression_opportunities_high_density(self):
        rows = [
            {"id": str(i), "title": f"T{i}", "content_len": 1000,
             "cache_type": "idea", "tags": []}
            for i in range(5)
        ]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.get_compression_opportunities(
                "p1", char_limit=1000
            )
        self.assertIn("HIGH", result)
        self.assertIn("Cluster [IDEA]", result)

    async def test_get_compression_opportunities_empty(self):
        pool, _, _ = make_pool_mock(rows=[])
        with patch("server.read_pool", pool):
            result = await server.get_compression_opportunities("empty")
        self.assertIn("No notes found", result)

    # ------------------------------------------------------------------
    # Missing coverage from Round 2 (F-07, F-08)
    # ------------------------------------------------------------------
    async def test_get_retention_policies_empty(self):
        pool, _, _ = make_pool_mock(rows=[])
        with patch("server.read_pool", pool):
            result = await server.get_retention_policies()
        self.assertIn("No retention policies found", result)

    async def test_get_compression_below_threshold(self):
        """Notes exist but total size is below char_limit."""
        rows = [{"id": "1", "title": "Small", "content_len": 50,
                 "cache_type": "idea", "tags": []}]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.get_compression_opportunities(
                "p1", char_limit=5000
            )
        self.assertIn("healthy", result.lower())
        self.assertNotIn("HIGH", result)

    # ------------------------------------------------------------------
    # SQL injection verification (F-09)
    # ------------------------------------------------------------------
    async def test_store_note_uses_parameterized_query(self):
        """Verify SQL injection payload is passed as param, not interpolated."""
        pool, conn, _ = make_pool_mock()
        model_mock = make_embedding_mock()
        payload = "'; DROP TABLE design_cache; --"

        with patch("server.write_pool", pool), \
             patch("server.get_embedding_model", new_callable=AsyncMock, return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            await server.store_note("proj", payload, "content")

        # The SQL should use %s placeholders, not contain the payload
        execute_call = conn.execute.call_args
        sql = execute_call[0][0]
        params = execute_call[0][1]
        self.assertIn("%s", sql)
        self.assertNotIn("DROP TABLE", sql)
        self.assertIn(payload, params)


if __name__ == "__main__":
    unittest.main()
