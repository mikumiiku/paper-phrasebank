"""Tests for ``llm.metadata.extract_metadata``."""
from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest

from phrasebank.llm.client import LLMClient, LLMError
from phrasebank.llm.metadata import Metadata, MetadataExtractionError, extract_metadata


def _client() -> LLMClient:
    return LLMClient(client=MagicMock(spec=openai.OpenAI), model_name="test")


def test_extract_metadata_success():
    c = _client()
    c.call_object = MagicMock(
        return_value={"paper_title": "Deep Learning", "paper_authors": "A. B, C. D", "paper_year": "2023"}
    )
    md = extract_metadata(c, "Deep Learning\nA. B, C. D\nAbstract ...")
    assert md == Metadata(
        paper_title="Deep Learning", paper_authors="A. B, C. D", paper_year="2023"
    )


def test_extract_metadata_only_first_page_chars():
    c = _client()
    c.call_object = MagicMock(return_value={"paper_title": "X", "paper_authors": "Y", "paper_year": "Z"})
    long_text = "a" * 5000
    extract_metadata(c, long_text)
    sent = c.call_object.call_args.args[1]
    # USER_METADATA injects text into the prompt; ensure prompt is clipped.
    assert "a" * 5000 not in sent
    # The prompt template markers contain only a few stray "a" chars
    # (inside "Extract metadata"), so the bulk must be the 2000 clipped chars.
    assert 1995 <= sent.count("a") <= 2005


def test_extract_metadata_null_fields_become_empty():
    c = _client()
    c.call_object = MagicMock(
        return_value={"paper_title": "X", "paper_authors": None, "paper_year": None}
    )
    md = extract_metadata(c, "text")
    assert md.paper_authors == ""
    assert md.paper_year == ""


def test_extract_metadata_unexpected_type_raises():
    c = _client()
    c.call_object = MagicMock(return_value="not a dict")
    with pytest.raises(MetadataExtractionError):
        extract_metadata(c, "text")
