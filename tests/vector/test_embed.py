"""Tests for vector.embed: lazy singleton + encode shape."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from phrasebank.vector import embed


def _fake_encode(self, texts, **kwargs):
    # mimic ST: returns an np.ndarray of shape (len(texts), dim)
    return np.ones((len(texts), 384), dtype="float32")


def test_get_model_is_singleton():
    """get_model loads the SentenceTransformer once and returns the same object."""
    fake_instance = MagicMock()
    with patch.object(embed, "_model", None), patch(
        "phrasebank.vector.embed.SentenceTransformer",
        return_value=fake_instance,
    ) as ctor:
        first = embed.get_model()
        second = embed.get_model()
    assert first is fake_instance
    assert first is second
    ctor.assert_called_once_with(embed.MODEL_NAME)


def test_encode_returns_correct_shape():
    """encode calls the model and returns a list of float-lists with right dim."""
    fake = MagicMock()
    fake.encode.side_effect = _fake_encode.__get__(fake, MagicMock)
    texts = ["hello world", "another sentence"]
    with patch.object(embed, "_model", fake):
        out = embed.encode(texts)
    # called exactly once with the input list
    fake.encode.assert_called_once()
    # Mock records the call via ``call_args``; its ``.args``/``.kwargs``
    # expose the positional and keyword arguments respectively.
    call = fake.encode.call_args
    passed = call.kwargs.get("texts") or (call.args[0] if call.args else None)
    assert passed == texts, f"texts mismatch: passed={passed!r}"
    assert call.kwargs.get("normalize_embeddings") is True
    # output shape
    assert isinstance(out, list)
    assert len(out) == len(texts)
    assert all(isinstance(v, list) for v in out)
    assert all(len(v) == 384 for v in out)
    assert all(all(isinstance(x, float) for x in v) for v in out)


def test_encode_empty_input():
    """encode on an empty list returns [] and never touches the model."""
    fake = MagicMock()
    with patch.object(embed, "_model", fake):
        out = embed.encode([])
    assert out == []
    fake.encode.assert_not_called()


def test_model_name_constant():
    assert embed.MODEL_NAME == "BAAI/bge-m3"
