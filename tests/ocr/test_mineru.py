"""MinerU backend unit tests (HTTP fully mocked)."""
from __future__ import annotations

import httpx
import pytest
import respx

from phrasebank.ocr import get_backend
from phrasebank.ocr.mineru import MinerUBackend

BASE = "https://api.mineru.net/v4"


@respx.mock
def test_mineru_top_level_text():
    route = respx.post(BASE + "/extract").mock(
        return_value=httpx.Response(200, json={"text": "Hello page 1"})
    )
    be = MinerUBackend(api_key="sk-xzy", base_url=BASE)
    assert be.recognize(1, b"\x89PNG...") == "Hello page 1"
    assert route.called


@respx.mock
def test_mineru_blocks():
    route = respx.post(BASE + "/extract").mock(
        return_value=httpx.Response(200, json={"blocks": [{"text": "a"}, {"text": "b"}]})
    )
    be = MinerUBackend(api_key="sk-xzy", base_url=BASE)
    assert be.recognize(1, b"img") == "a\nb"
    assert route.called


@respx.mock
def test_mineru_non_2xx_raises_first_fault():
    respx.post(BASE + "/extract").mock(return_value=httpx.Response(401))
    be = MinerUBackend(api_key="bad-key", base_url=BASE)
    with pytest.raises(httpx.HTTPStatusError):
        be.recognize(1, b"img")


def test_registry_wires_mineru():
    be = get_backend("mineru", api_key="k", base_url=BASE)
    assert isinstance(be, MinerUBackend)
