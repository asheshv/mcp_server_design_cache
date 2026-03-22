"""
test_server.py - Comprehensive automated test suite for all MCP tools in server.py.

This test suite uses unittest.mock to patch the database connection pools
and the sentence-transformers embedding model, so all tests run offline
without any live database or network connections required.
"""
import asyncio
import datetime
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import server


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


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestMCPTools(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Reset the rate limiter between tests to avoid 60 RPM limit contamination."""
        server.limiter = server.RateLimiter(60)

    # ------------------------------------------------------------------
    # search_design
    # ------------------------------------------------------------------
    async def test_search_design_returns_results(self):
        rows = [
            {"id": "id-1", "title": "Auth Design", "content": "x" * 300, "cache_type": "idea",
             "keyword_score": 0.9, "semantic_score": 0.85},
        ]
        pool, conn, cur = make_pool_mock(rows=rows)
        model_mock = make_embedding_mock()

        with patch("server.read_pool", pool), \
             patch("server.get_embedding_model", return_value=model_mock), \
             patch("asyncio.to_thread", new_callable=AsyncMock, return_value=model_mock.encode("")):
            # encode returns a MagicMock, patch tolist on it
            async def fake_to_thread(fn, *args):
                return model_mock.encode(*args)
            with patch("asyncio.to_thread", side_effect=fake_to_thread):
                result = await server.search_design("myproject", "auth")

        self.assertIn("Auth Design", result)
        self.assertIn("Abstract:", result)

    async def test_search_design_no_results(self):
        pool, _, _ = make_pool_mock(rows=[])
        model_mock = make_embedding_mock()

        async def fake_to_thread(fn, *args):
            return model_mock.encode(*args)

        with patch("server.read_pool", pool), \
             patch("server.get_embedding_model", return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.search_design("myproject", "unknown-query")

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

        async def fake_to_thread(fn, *args):
            return model_mock.encode(*args)

        with patch("server.write_pool", pool), \
             patch("server.get_embedding_model", return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.store_note("myproject", "New Idea", "Content text", "idea")

        self.assertIn("cached successfully", result.lower())

    # ------------------------------------------------------------------
    # summarize_and_cleanup
    # ------------------------------------------------------------------
    async def test_summarize_and_cleanup_empty_ids(self):
        result = await server.summarize_and_cleanup("proj", [], "summary", "title")
        self.assertIn("Error", result)

    async def test_summarize_and_cleanup_success(self):
        pool, _, _ = make_pool_mock()
        model_mock = make_embedding_mock()

        async def fake_to_thread(fn, *args):
            return model_mock.encode(*args)

        with patch("server.write_pool", pool), \
             patch("server.get_embedding_model", return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.summarize_and_cleanup(
                "myproject", ["id-1", "id-2"], "A summary", "New Summary Title"
            )

        self.assertIn("2", result)
        self.assertIn("New Summary Title", result)

    # ------------------------------------------------------------------
    # set_retention_policy
    # ------------------------------------------------------------------
    async def test_set_retention_policy(self):
        pool, _, _ = make_pool_mock()
        with patch("server.write_pool", pool):
            result = await server.set_retention_policy("myproject", 14)

        self.assertIn("14", result)
        self.assertIn("myproject", result)

    # ------------------------------------------------------------------
    # run_smart_cleanup
    # ------------------------------------------------------------------
    async def test_run_smart_cleanup_nothing_expired(self):
        read_pool, _, _ = make_pool_mock(rows=[])
        with patch("server.read_pool", read_pool):
            result = await server.run_smart_cleanup()

        self.assertIn("within their retention limits", result)

    async def test_run_smart_cleanup_deletes_expired(self):
        expired_rows = [{"project_name": "myproject", "expired_count": 3}]
        read_pool, _, _ = make_pool_mock(rows=expired_rows)
        write_pool, _, _ = make_pool_mock()

        with patch("server.read_pool", read_pool), \
             patch("server.write_pool", write_pool):
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
             "cache_type": "idea", "created_at": datetime.datetime(2026, 3, 22, 9, 0, 0)},
        ]
        pool, _, _ = make_pool_mock(rows=rows)
        with patch("server.read_pool", pool):
            result = await server.get_recent_activity("myproject", limit=5)

        self.assertIn("Recent Note", result)

    async def test_get_recent_activity_empty(self):
        pool, _, _ = make_pool_mock(rows=[])
        with patch("server.read_pool", pool):
            result = await server.get_recent_activity("emptyproject", limit=5)

        self.assertIn("No recent activity", result)

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
        self.assertNotIn("/tmp/", result)  # Should return inline for < 20 rows

    async def test_export_large_project_saves_to_file(self):
        rows = [
            {"title": f"Note {i}", "content": "body", "cache_type": "idea",
             "created_at": datetime.datetime(2026, 3, 22, 9, 0, 0)}
            for i in range(25)  # > 20 threshold
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

    # ------------------------------------------------------------------
    # generate_spec_from_cache
    # ------------------------------------------------------------------
    async def test_generate_spec_found(self):
        row = {
            "title": "My Feature", "content": "Content here",
            "project_name": "proj", "created_at": datetime.datetime(2026, 3, 22),
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
            "project_name": "proj", "created_at": datetime.datetime(2026, 3, 22),
        }
        read_pool, _, _ = make_pool_mock(fetchone_row=row)
        write_pool, _, _ = make_pool_mock()
        model_mock = make_embedding_mock()

        async def fake_to_thread(fn, *args):
            return model_mock.encode(*args)

        with patch("server.read_pool", read_pool), \
             patch("server.write_pool", write_pool), \
             patch("server.get_embedding_model", return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.generate_adr_from_cache("idea-id", status="accepted")

        self.assertIn("ACCEPTED", result)
        self.assertIn("DB Choice", result)

    async def test_generate_adr_not_found(self):
        pool, _, _ = make_pool_mock(fetchone_row=None)
        with patch("server.read_pool", pool):
            result = await server.generate_adr_from_cache("bad-id")

        self.assertIn("Error", result)

    # ------------------------------------------------------------------
    # sync_doc_status
    # ------------------------------------------------------------------
    async def test_sync_doc_status_success(self):
        row = {"title": "Original Title", "content": "Original Content"}
        read_pool, _, _ = make_pool_mock(fetchone_row=row)
        write_pool, _, _ = make_pool_mock()
        model_mock = make_embedding_mock()

        async def fake_to_thread(fn, *args):
            return model_mock.encode(*args)

        with patch("server.read_pool", read_pool), \
             patch("server.write_pool", write_pool), \
             patch("server.get_embedding_model", return_value=model_mock), \
             patch("asyncio.to_thread", side_effect=fake_to_thread):
            result = await server.sync_doc_status("cache-id", "/tmp/spec.md", "implemented")

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
        result = await server.link_external_file_to_cache("some-id", "relative/path.md")
        self.assertIn("Error", result)
        self.assertIn("absolute", result.lower())

    async def test_link_external_file_not_found(self):
        result = await server.link_external_file_to_cache("some-id", "/nonexistent/path/file.md")
        self.assertIn("Error", result)
        self.assertIn("not found", result.lower())

    async def test_link_external_file_success(self, tmp_path=None):
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            filepath = f.name
        try:
            row = {"title": "My Title", "content": "Body"}
            read_pool, _, _ = make_pool_mock(fetchone_row=row)
            write_pool, _, _ = make_pool_mock()
            model_mock = make_embedding_mock()

            async def fake_to_thread(fn, *args):
                return model_mock.encode(*args)

            with patch("server.read_pool", read_pool), \
                 patch("server.write_pool", write_pool), \
                 patch("server.get_embedding_model", return_value=model_mock), \
                 patch("asyncio.to_thread", side_effect=fake_to_thread):
                result = await server.link_external_file_to_cache("cache-id", filepath, "spec")

            self.assertIn("Linked", result)
            self.assertIn(filepath, result)
        finally:
            os.unlink(filepath)

    # ------------------------------------------------------------------
    # RateLimiter
    # ------------------------------------------------------------------
    def test_rate_limiter_allows_under_limit(self):
        limiter = server.RateLimiter(5)
        for _ in range(5):
            limiter.check("testtool")  # Should not raise

    def test_rate_limiter_blocks_over_limit(self):
        limiter = server.RateLimiter(3)
        limiter.check("testtool")
        limiter.check("testtool")
        limiter.check("testtool")
        with self.assertRaises(RuntimeError) as ctx:
            limiter.check("testtool")
        self.assertIn("Rate limit exceeded", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
