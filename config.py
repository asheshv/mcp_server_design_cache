"""Configuration constants and environment validation."""
import os


# --- Database ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "design_db")
DB_READ_PASS = os.getenv("DB_READ_PASS")
DB_WRITE_PASS = os.getenv("DB_WRITE_PASS")

if not DB_READ_PASS or not DB_WRITE_PASS:
    raise EnvironmentError(
        "DB_READ_PASS and DB_WRITE_PASS environment variables must be set. "
        "See .env.example for reference."
    )

# --- Connection pool ---
MAX_LIFETIME = 300  # 5 minutes
MAX_POOL_SIZE = 10
MIN_POOL_SIZE = 2

# --- File security ---
ALLOWED_FILE_PREFIXES = ("/tmp/", os.path.expanduser("~/"))

# --- Content limits ---
MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 50_000  # ~50KB

# --- Validation sets ---
VALID_TAG_LOGIC = {"AND", "OR", "NOT"}
VALID_CACHE_TYPES = {"project", "idea"}
VALID_ADR_STATUSES = {"proposed", "accepted", "superseded", "deprecated"}
