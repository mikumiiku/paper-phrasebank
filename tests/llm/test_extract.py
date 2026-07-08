"""Tests for ``llm.extract.extract_sentences`` + ``retry_failed``."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import openai
import pytest

from phrasebank.llm.client import LLMClient, LLMError
from phrasebank.llm.extract import (
    CandidateSentence,
    extract_sentences,
    retry_failed,
    write_failures,
)
from phrasebank.llm.metadata import Metadata


def _client() -> LLMClient:
    return LLMClient(client=MagicMock(spec=openai.OpenAI), model_name="test")


SENTENCE_PAYLOAD = [
    {
        "sentence": "We propose a new method.",
        "language": "en",
        "function_category": "方法论述句",
        "tags": ["method"],
        "usage_note": "useful for methods section",
    }
]


def _md() -> Metadata:
    return Metadata(paper_title="T", paper_authors="A", paper_year="2024")


def test_extract_success():
    c = _client()
    c.call_json = MagicMock(return_value=SENTENCE_PAYLOAD)
    candidates, failed = extract_sentences(c, ["chunk text"], _md())
    assert failed == []
    assert len(candidates) == 1
    s = candidates[0]
    assert isinstance(s, CandidateSentence)
    assert s.sentence == "We propose a new method."
    assert s.function_category == "方法论述句"
    assert s.tags == ["method"]
    assert s.paper_title == "T"
    assert s.paper_year == "2024"


def test_extract_empty_chunk_skips():
    c = _client()
    c.call_json = MagicMock(return_value=[])
    candidates, failed = extract_sentences(c, ["chunk"], _md())
    assert candidates == []
    assert failed == []


def test_extract_partial_failure_returns_failed_indices():
    c = _client()

    def side_effect(system, user):
        # Fail on second chunk.
        if "chunk2" in user:
            raise LLMError("boom")
        return SENTENCE_PAYLOAD

    c.call_json = MagicMock(side_effect=side_effect)
    chunks = ["chunk1: long enough", "chunk2: trigger fail", "chunk3: another"]
    candidates, failed = extract_sentences(c, chunks, _md())
    assert failed == [1]
    assert len(candidates) == 2  # chunk1 + chunk3


def test_extract_fail_hard_raises():
    c = _client()
    c.call_json = MagicMock(side_effect=LLMError("boom"))
    with pytest.raises(LLMError):
        extract_sentences(c, ["chunk"], _md(), fail_hard=True)


def test_extract_propagates_metadata():
    c = _client()
    c.call_json = MagicMock(return_value=SENTENCE_PAYLOAD)
    candidates, failed = extract_sentences(
        c, ["chunk one", "chunk two"], _md(), fail_hard=False
    )
    assert failed == []
    assert len(candidates) == 2
    for cand in candidates:
        assert cand.paper_title == "T"
        assert cand.paper_authors == "A"
        assert cand.paper_year == "2024"


# ──────────────────────────────────────────────────────────────────────────────
# write_failures + retry_failed
# ──────────────────────────────────────────────────────────────────────────────

def test_write_and_retry_failures(isolated_config):
    cfg_dir, data_dir = isolated_config
    chunks = ["c1: a", "c2: b", "c3: c"]
    c = _client()
    # Only the first retry (index 1) succeeds.
    c.call_json = MagicMock(return_value=SENTENCE_PAYLOAD)

    write_failures("deadbeef", [1])
    retry_candidates, still_failed = retry_failed(c, "deadbeef", chunks, _md())
    assert still_failed == []
    assert len(retry_candidates) == 1
    assert c._c.chat.completions.create.call_count == 0  # call_json mocked
    assert c.call_json.call_count == 1


def test_retry_missing_file_returns_empty(isolated_config):
    cfg_dir, data_dir = isolated_config
    c = _client()
    candidates, failed = retry_failed(c, "no_such_hash", ["c1"], _md())
    assert candidates == []
    assert failed == []


def test_retry_failed_inner_failure_propagates(isolated_config):
    cfg_dir, data_dir = isolated_config
    chunks = ["c1"]
    c = _client()
    c.call_json = MagicMock(side_effect=LLMError("boom"))
    write_failures("face", [0])
    retry_candidates, still_failed = retry_failed(c, "face", chunks, _md())
    assert still_failed == [0]
    assert retry_candidates == []
