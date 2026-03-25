"""Embedding model loading with async-safe lazy initialization.

Supports two backends:
- ONNX Runtime (lightweight, ~80MB) — preferred for deployment
- sentence-transformers + PyTorch (~2GB) — fallback / development
"""

import asyncio
import os
import sys

import numpy as np

# ONNX Runtime is preferred (lightweight, ~80MB).
# Falls back to sentence-transformers + PyTorch (~2GB) if ONNX model not available.
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "")

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    _HAS_ONNX = True
except ImportError:
    _HAS_ONNX = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

if not _HAS_ONNX and not _HAS_ST:
    raise ImportError(
        "Either onnxruntime+tokenizers or sentence-transformers must be installed."
    )


class OnnxEmbedder:
    """Lightweight embedding using ONNX Runtime + HuggingFace tokenizer."""

    def __init__(self, model_dir: str):
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(
            os.path.join(model_dir, "tokenizer.json")
        )
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=256)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector (same interface as SentenceTransformer)."""
        encoded = self.tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        outputs = self.session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        # Mean pooling over token embeddings
        token_embeddings = outputs[0]  # (1, seq_len, hidden_dim)
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counted = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return (summed / counted).squeeze(0)


_embedding_model = None
_model_lock: asyncio.Lock | None = None


def _get_model_lock() -> asyncio.Lock:
    """Lazily create the lock to avoid binding to wrong event loop on Python 3.9."""
    global _model_lock
    if _model_lock is None:
        _model_lock = asyncio.Lock()
    return _model_lock


def _load_model():
    """Load the appropriate embedding model (ONNX preferred, ST fallback)."""
    if _HAS_ONNX and ONNX_MODEL_PATH and os.path.isdir(ONNX_MODEL_PATH):
        onnx_file = os.path.join(ONNX_MODEL_PATH, "model.onnx")
        if os.path.exists(onnx_file):
            print("Loading ONNX embedding model...", file=sys.stderr)
            return OnnxEmbedder(ONNX_MODEL_PATH)

    if _HAS_ST:
        print("Loading SentenceTransformer model (fallback)...", file=sys.stderr)
        return SentenceTransformer('all-MiniLM-L6-v2')

    raise RuntimeError(
        "No embedding backend available. Set ONNX_MODEL_PATH or install "
        "sentence-transformers."
    )


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
            _embedding_model = await asyncio.to_thread(_load_model)
        finally:
            sys.stdout = _stdout
        return _embedding_model
