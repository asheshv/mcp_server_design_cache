"""Shared utilities: file validation, rate limiting, project detection."""
import asyncio
import hashlib
import os
import re
import time
from collections import defaultdict

from config import ALLOWED_FILE_PREFIXES


def sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use in file paths.

    Appends a short hash to prevent collisions between distinct names
    that map to the same sanitized form.
    """
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    short_hash = hashlib.md5(name.encode()).hexdigest()[:8]
    return f"{safe}_{short_hash}"


def validate_file_path(path: str, reject_symlinks: bool = False) -> bool:
    """Check that a resolved path is within allowed directories."""
    if reject_symlinks and os.path.islink(path):
        return False
    resolved = os.path.realpath(path)
    return any(resolved.startswith(prefix) for prefix in ALLOWED_FILE_PREFIXES)


def get_local_project_name() -> str | None:
    """Read the project name from a local .design_cache file."""
    try:
        path = os.path.join(os.getcwd(), ".design_cache")
        if os.path.exists(path):
            with open(path, "r") as f:
                name = f.readline().strip()
                if name:
                    return name
    except Exception:
        pass
    return None


class RateLimiter:
    """Async-safe rate limiter with per-tool RPM tracking."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self.history = defaultdict(list)
        self.lock = asyncio.Lock()

    async def check(self, tool: str):
        async with self.lock:
            now = time.time()
            self.history[tool] = [t for t in self.history[tool] if now - t < 60]
            if not self.history[tool]:
                del self.history[tool]
            if len(self.history[tool]) >= self.rpm:
                raise RuntimeError(f"Rate limit exceeded (Max {self.rpm} RPM)")
            self.history[tool].append(now)
