"""Vector submodule: shared Chroma client and collection name."""
from __future__ import annotations

import threading
from pathlib import Path

import chromadb

import phrasebank.config as _config

_lock = threading.Lock()
_client: chromadb.PersistentClient | None = None

COLLECTION_NAME = "phrasebank_sentences"


def _data_chroma_path() -> Path:
    path: Path = _config.data_dir() / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_client() -> chromadb.PersistentClient:
    """Return a module-level PersistentClient singleton rooted at the
    project data dir (``<data_dir>/chroma``).

    If ``data_dir`` changed since the last call (common in tests swapping
    isolated tmp dirs), rebuild against the new path. Comparison uses the
    resolved string form so a fresh isolated data dir always rebuilds."""
    global _client
    if _client is not None and getattr(_client, "_ppb_path", None) == str(_data_chroma_path()):
        return _client

    with _lock:
        target = str(_data_chroma_path())
        if _client is not None and getattr(_client, "_ppb_path", None) == target:
            return _client
        _client = chromadb.PersistentClient(path=target)
        _client._ppb_path = target  # type: ignore[attr-defined]
        return _client


def reset_client() -> None:
    """Drop the cached client (used by tests)."""
    global _client
    _client = None


def get_collection_name() -> str:
    return COLLECTION_NAME
