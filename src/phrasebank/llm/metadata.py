"""Decoupled metadata extraction: title / authors / year from the first page."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .client import LLMClient
from .prompts import SYSTEM_METADATA, USER_METADATA

FRONT_PAGE_CHARS = 2000


class MetadataExtractionError(Exception):
    """Metadata could not be extracted (first-fault)."""


@dataclass
class Metadata:
    paper_title: str = ""
    paper_authors: str = ""
    paper_year: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_metadata(client: LLMClient, first_page_text: str) -> Metadata:
    """Extract paper metadata from at most the first ``FRONT_PAGE_CHARS`` chars.

    Called exactly once per paper; the caller is responsible for truncating and
    for caching the result across chunks.
    """
    clipped = first_page_text[:FRONT_PAGE_CHARS]
    item = client.call_object(SYSTEM_METADATA, USER_METADATA.format(text=clipped))
    return _to_metadata(item)


def _to_metadata(item: Any) -> Metadata:
    if not isinstance(item, dict):
        raise MetadataExtractionError(
            f"Expected dict for metadata, got {type(item).__name__}"
        )
    return Metadata(
        paper_title=_safe_str(item.get("paper_title")),
        paper_authors=_safe_str(item.get("paper_authors")),
        paper_year=_safe_str(item.get("paper_year")),
    )


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()
