"""Tests for the JSON sandwich parser + retry behaviour in ``llm.client``."""
from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest

from phrasebank.llm.client import (
    LLMClient,
    LLMError,
    _parse_as_array,
    _parse_as_object,
    _sandwich,
)


def _client(model: str = "test-model") -> LLMClient:
    return LLMClient(client=MagicMock(spec=openai.OpenAI), model_name=model)


# ──────────────────────────────────────────────────────────────────────────────
# Sandwich parser — array mode
# ──────────────────────────────────────────────────────────────────────────────

def test_sandwich_bare_array():
    text = '[{"sentence": "hello."}]'
    assert _parse_as_array(text) == [{"sentence": "hello."}]


def test_sandwich_markdown_fence():
    text = 'Here is the result:\n```json\n[{"sentence": "a."}]\n```\nDone.'
    assert _parse_as_array(text) == [{"sentence": "a."}]


def test_sandwich_dict_wrapping_array():
    # Some models return {"sentences": [...]}.
    text = 'Sure! {"sentences": [{"sentence": "x."}, {"sentence": "y."}]}'
    assert _parse_as_array(text) == [
        {"sentence": "x."},
        {"sentence": "y."},
    ]


def test_sandwich_leading_trailing_garbage():
    text = 'blah blah [{"a": 1}] trailing junk'
    assert _parse_as_array(text) == [{"a": 1}]


def test_sandwich_no_brackets_raises():
    with pytest.raises(LLMError):
        _parse_as_array("no json here at all")


def test_sandwich_invalid_json_raises():
    with pytest.raises(LLMError):
        _parse_as_array("prefix [not json] suffix")


def test_sandwich_dict_without_list_raises():
    # A dict with no list-valued field cannot be unwrapped into an array.
    with pytest.raises(LLMError):
        _parse_as_array('{"foo": "bar"}')


# ──────────────────────────────────────────────────────────────────────────────
# Sandwich parser — object mode (metadata)
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_as_object_bare():
    text = '{"paper_title": "T", "paper_authors": "A", "paper_year": "2024"}'
    out = _parse_as_object(text)
    assert out["paper_title"] == "T"


def test_parse_as_object_markdown_wrapped():
    text = '```json\n{"paper_title": "T"}\n```'
    assert _parse_as_object(text)["paper_title"] == "T"


def test_parse_as_object_list_of_one_unwraps():
    text = '[{"paper_title": "T"}]'
    assert _parse_as_object(text)["paper_title"] == "T"


def test_parse_as_object_no_structure_raises():
    with pytest.raises(LLMError):
        _parse_as_object("nothing")


# ──────────────────────────────────────────────────────────────────────────────
# LLMClient.call_json — end-to-end with mocked SDK
# ──────────────────────────────────────────────────────────────────────────────

def _mock_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def test_call_json_success():
    c = _client()
    c._c.chat.completions.create.return_value = _mock_response(
        '[{"sentence": "a.", "language": "en", "function_category": "结果陈述句", "tags": [], "usage_note": "n"}]'
    )
    out = c.call_json("sys", "usr")
    assert out == [
        {
            "sentence": "a.",
            "language": "en",
            "function_category": "结果陈述句",
            "tags": [],
            "usage_note": "n",
        }
    ]


def test_call_json_dict_wrap_fallback():
    c = _client()
    c._c.chat.completions.create.return_value = _mock_response(
        '{"sentences": [{"sentence": "b."}]}'
    )
    out = c.call_json("sys", "usr")
    assert out == [{"sentence": "b."}]


def test_call_json_invalid_raises_first_fault():
    c = _client()
    c._c.chat.completions.create.return_value = _mock_response("totally not json")
    with pytest.raises(LLMError):
        c.call_json("sys", "usr")


def test_call_object_success():
    c = _client()
    c._c.chat.completions.create.return_value = _mock_response(
        '{"paper_title": "T", "paper_authors": "A", "paper_year": "2024"}'
    )
    out = c.call_object("sys", "usr")
    assert out == {"paper_title": "T", "paper_authors": "A", "paper_year": "2024"}


# ──────────────────────────────────────────────────────────────────────────────
# Retry: 5xx retries, 4xx does not
# ──────────────────────────────────────────────────────────────────────────────

def _status_error(code: int) -> openai.APIStatusError:
    req = MagicMock()
    resp = MagicMock()
    resp.status_code = code
    body = {"error": {"message": "boom"}}
    return openai.APIStatusError("boom", response=resp, body=body)


def test_retry_on_500(monkeypatch):
    c = _client()
    calls = {"n": 0}

    def fake_create(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _status_error(500)
        return _mock_response('[{"sentence": "ok."}]')

    c._c.chat.completions.create.side_effect = fake_create
    out = c.call_json("sys", "usr")
    assert out == [{"sentence": "ok."}]
    assert calls["n"] == 2


def test_no_retry_on_400(monkeypatch):
    c = _client()
    c._c.chat.completions.create.side_effect = _status_error(400)
    with pytest.raises(LLMError):
        c.call_json("sys", "usr")
    assert c._c.chat.completions.create.call_count == 1
