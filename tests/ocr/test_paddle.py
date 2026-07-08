"""PaddleOCR backend unit tests (HTTP fully mocked)."""
from __future__ import annotations

import httpx
import pytest
import respx

from phrasebank.ocr import get_backend
from phrasebank.ocr.paddle import PaddleBackend

BASE = "http://localhost:8866"


@respx.mock
def test_paddle_dict_text():
    respx.post(BASE + "/ocr").mock(
        return_value=httpx.Response(200, json={"text": "line1\nline2"})
    )
    be = PaddleBackend(api_key="k", base_url=BASE)
    assert be.recognize(1, b"img") == "line1\nline2"


@respx.mock
def test_paddle_results_with_rec_texts():
    """Real PaddleOCR OCR result shape under ``rec_texts``."""
    respx.post(BASE + "/ocr").mock(
        return_value=httpx.Response(
            200,
            json={"prunedResult": {"rec_texts": ["x", "y"], "rec_scores": [0.9, 0.8]}},
        )
    )
    be = PaddleBackend(api_key="k", base_url=BASE)
    assert be.recognize(1, b"img") == "x\ny"


def test_paddle_uses_official_cloud_by_default():
    """PaddleBackend defaults to the official PaddleOCR cloud — no self-host."""
    be = PaddleBackend(api_key="k", base_url="")
    assert "paddleocr.aistudio-app.com" in str(be._client.base_url)


def test_paddle_override_base_url_is_honoured():
    """But an explicit override (self-host / different port) is still used."""
    be = PaddleBackend(api_key="k", base_url="http://ocr.example.com:9999")
    assert "ocr.example.com:9999" in str(be._client.base_url)


def test_registry_wires_paddle():
    be = get_backend("paddle", api_key="k", base_url=BASE)
    assert isinstance(be, PaddleBackend)


def test_unknown_backend_raises():
    from phrasebank.config import ConfigError

    with pytest.raises(ConfigError):
        get_backend("nonexistent")
