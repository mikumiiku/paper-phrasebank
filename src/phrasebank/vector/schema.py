"""Metadata schema and id generation for the sentence vector collection."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from importlib.metadata import version as _pkg_version

# ── Field constants (single source of truth, aligned with requirements §4) ──
SENTENCE = "sentence"
LANGUAGE = "language"
PAPER_TITLE = "paper_title"
PAPER_AUTHORS = "paper_authors"
PAPER_YEAR = "paper_year"
SOURCE_FILE_HASH = "source_file_hash"
FUNCTION_CATEGORY = "function_category"
TAGS = "tags"
USAGE_NOTE = "usage_note"
REVIEWED = "reviewed"
CREATED_AT = "created_at"

# Fields whose string<->Python conversion is non-trivial.
_TAGS_SEP = ","


def make_id() -> str:
    return uuid.uuid4().hex


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_identifier(model_name: str) -> str:
    """Deterministic fingerprint of the *embedding model* used to build the
    collection, without loading weights. Combines the model repo name with the
    sentence-transformers library version so a library upgrade (or a different
    model) invalidates stale vectors.

    Stored as ``model_identifier`` in collection metadata and rechecked on read.
    """
    st_version = _pkg_version("sentence-transformers")
    return f"sentence-transformers@{st_version}:{model_name}"


def tags_to_str(tags: list[str] | None) -> str:
    if not tags:
        return ""
    return _TAGS_SEP.join(t for t in tags if t)


def tags_from_str(s: str | None) -> list[str]:
    if not s:
        return []
    return [t for t in s.split(_TAGS_SEP) if t]


def to_metadata(
    *,
    sentence: str,
    paper_title: str,
    source_file_hash: str,
    function_category: str,
    language: str,
    paper_authors: str = "",
    paper_year: str = "",
    tags: list[str] | None = None,
    usage_note: str = "",
    reviewed: bool = True,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a Chroma metadata dict from structured record fields.

    ``tags`` (list) is stored as a comma-separated string; ``reviewed`` is a
    bool; ``created_at`` defaults to now (ISO UTC).
    """
    return {
        SENTENCE: sentence,
        LANGUAGE: language,
        PAPER_TITLE: paper_title,
        PAPER_AUTHORS: paper_authors,
        PAPER_YEAR: paper_year,
        SOURCE_FILE_HASH: source_file_hash,
        FUNCTION_CATEGORY: function_category,
        TAGS: tags_to_str(tags),
        USAGE_NOTE: usage_note,
        REVIEWED: reviewed,
        CREATED_AT: created_at or now_iso(),
    }


def from_metadata(d: dict[str, Any]) -> dict[str, Any]:
    """Reverse of :func:`to_metadata` — tags back to list, reviewed to bool.

    Unknown/extra keys are preserved verbatim so callers don't lose data.
    """
    out = dict(d)
    out[TAGS] = tags_from_str(d.get(TAGS))
    out[REVIEWED] = bool(d.get(REVIEWED, False))
    return out
