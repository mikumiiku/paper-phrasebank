"""Embedding: lazy-loaded sentence-transformers model singleton."""
from __future__ import annotations

import threading

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """Return a module-level SentenceTransformer singleton, loaded once and
    shared by index and search."""
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model
        _model = SentenceTransformer(MODEL_NAME)
        return _model


def encode(texts: list[str]) -> list[list[float]]:
    """Encode a list of strings into dense vectors (as plain float lists)."""
    if not texts:
        return []
    model = get_model()
    # Pass the input list positionally: sentence-transformers renamed the first
    # kwarg across versions (`texts` -> `inputs`); positional keeps this working.
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    # ``vectors`` is an ndarray; normalize -> unit vectors, 1 - cosine == score.
    return [v.tolist() for v in vectors]
