"""Embedding model loading with async-safe lazy initialization."""

import asyncio
import sys

from sentence_transformers import SentenceTransformer


_embedding_model = None
_model_lock: asyncio.Lock | None = None


def _get_model_lock() -> asyncio.Lock:
    """Lazily create the lock to avoid binding to wrong event loop on Python 3.9."""
    global _model_lock
    if _model_lock is None:
        _model_lock = asyncio.Lock()
    return _model_lock


async def get_embedding_model():
    """Load embedding model lazily with async-safe double-checked locking."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    async with _get_model_lock():
        if _embedding_model is not None:
            return _embedding_model
        _stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            _embedding_model = await asyncio.to_thread(
                SentenceTransformer, 'all-MiniLM-L6-v2'
            )
        finally:
            sys.stdout = _stdout
        return _embedding_model
