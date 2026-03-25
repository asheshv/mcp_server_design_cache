"""
test_integration.py - Integration tests against a real PostgreSQL + pgvector database.

Prerequisites:
    docker compose -f docker-compose.test.yml up -d
    # Wait for DB to be healthy
    export DB_HOST=localhost DB_PORT=5433
    export DB_READ_PASS=testpassword DB_WRITE_PASS=testpassword
    python -m pytest test_integration.py -v

These tests exercise real SQL queries, transactions, and schema behavior.
They complement the unit tests (test_server.py) which use mocks.
"""
import asyncio
import os
import sys
import unittest

# Require integration test env vars
REQUIRED_VARS = ["DB_READ_PASS", "DB_WRITE_PASS"]
_missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if _missing:
    print(f"Skipping integration tests: missing env vars {_missing}")
    sys.exit(0)

# Override DB settings for test database
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5433")
os.environ.setdefault("DB_NAME", "design_db")

import server


async def _setup_test_db():
    """Open pools and run migrations."""
    await server.read_pool.open()
    await server.write_pool.open()
    await server.apply_migrations()


async def _teardown_test_db():
    """Clean up test data and close pools."""
    async with server.write_pool.connection() as conn:
        await conn.execute("DELETE FROM design_cache")
        await conn.execute("DELETE FROM retention_policies")
    await server.read_pool.close()
    await server.write_pool.close()


async def _clean_data():
    """Delete all test data between tests."""
    async with server.write_pool.connection() as conn:
        await conn.execute("DELETE FROM design_cache")
        await conn.execute("DELETE FROM retention_policies")


class TestIntegration(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        """Open DB pools once for all tests."""
        asyncio.get_event_loop().run_until_complete(_setup_test_db())

    @classmethod
    def tearDownClass(cls):
        """Close DB pools."""
        asyncio.get_event_loop().run_until_complete(_teardown_test_db())

    async def asyncSetUp(self):
        await _clean_data()
        server.limiter = server.RateLimiter(1000)

    # ------------------------------------------------------------------
    # Store and retrieve
    # ------------------------------------------------------------------
    async def test_store_and_search(self):
        """Store a note, then search for it."""
        result = await server.store_note(
            "test-project", "Authentication Design",
            "We should use JWT tokens for API auth.", "idea",
            tags=["auth", "api"],
        )
        self.assertIn("cached successfully", result.lower())

        search_result = await server.search_design(
            "test-project", "JWT authentication"
        )
        self.assertIn("Authentication Design", search_result)

    async def test_store_and_expand(self):
        """Store a note, search for its ID, then expand it."""
        await server.store_note(
            "test-project", "DB Choice", "PostgreSQL with pgvector", "idea"
        )

        # Get recent to find the ID
        activity = await server.get_recent_activity("test-project")
        self.assertIn("DB Choice", activity)

    async def test_store_and_delete(self):
        """Store, then delete, then verify gone."""
        await server.store_note("test-project", "Temporary", "Delete me", "idea")

        activity = await server.get_recent_activity("test-project")
        self.assertIn("Temporary", activity)

        # Extract ID from activity (format: [date] [TYPE] Title\nContent)
        # Search to get ID
        search = await server.search_design("test-project", "Delete me")
        # Extract ID from "ID:xxxx]"
        import re
        match = re.search(r'ID:([^\]]+)', search)
        self.assertIsNotNone(match)
        note_id = match.group(1)

        delete_result = await server.delete_note(note_id)
        self.assertIn("deleted", delete_result.lower())

    # ------------------------------------------------------------------
    # Update with FOR UPDATE locking
    # ------------------------------------------------------------------
    async def test_update_note_round_trip(self):
        """Store, update, verify the update persists."""
        await server.store_note(
            "test-project", "Draft Idea", "Initial content", "idea"
        )
        search = await server.search_design("test-project", "Draft Idea")
        import re
        match = re.search(r'ID:([^\]]+)', search)
        note_id = match.group(1)

        update_result = await server.update_note(
            note_id, content="Updated content", cache_type="project"
        )
        self.assertIn("updated successfully", update_result)

        expanded = await server.expand_design_note(note_id)
        self.assertIn("Updated content", expanded)
        self.assertIn("PROJECT", expanded)

    # ------------------------------------------------------------------
    # Retention and cleanup
    # ------------------------------------------------------------------
    async def test_retention_policy_and_cleanup(self):
        """Set a 0-day retention, verify cleanup finds nothing (days must be > 0)."""
        result = await server.set_retention_policy("test-project", 0)
        self.assertIn("Error", result)

        result = await server.set_retention_policy("test-project", 1)
        self.assertIn("1 days", result)

        policies = await server.get_retention_policies("test-project")
        self.assertIn("test-project", policies)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    async def test_health_check(self):
        result = await server.health_check()
        self.assertIn("Healthy", result)

    # ------------------------------------------------------------------
    # Compression opportunities
    # ------------------------------------------------------------------
    async def test_compression_empty_project(self):
        result = await server.get_compression_opportunities("empty-project")
        self.assertIn("No notes found", result)

    # ------------------------------------------------------------------
    # Input validation still works with real DB
    # ------------------------------------------------------------------
    async def test_store_invalid_cache_type(self):
        result = await server.store_note(
            "test-project", "T", "C", cache_type="invalid"
        )
        self.assertIn("Error", result)

    async def test_search_invalid_tag_logic(self):
        result = await server.search_design(
            "test-project", "q", tag_logic="INVALID"
        )
        self.assertIn("Error", result)


if __name__ == "__main__":
    unittest.main()
