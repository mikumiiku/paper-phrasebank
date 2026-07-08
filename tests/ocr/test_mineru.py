"""MinerU backend tests (HTTP fully mocked).

Covers: local /file_parse sync path, response shape variants (md / text /
data-blocks), non-2xx first-fault, and the HMAC signer for the (undocumented)
cloud mode.
"""
from __future__ import annotations

import httpx
import pytest
import respx

from phrasebank.ocr import get_backend
from phrasebank.ocr.mineru import MinerUBackend


def _be(base_url: str = "http://localhost:8000", api_key: str = "") -> MinerUBackend:
    return MinerUBackend(api_key=api_key, base_url=base_url)


@respx.mock
def test_local_plain_text_response():
    route = respx.post("http://localhost:8000/file_parse").mock(
        return_value=httpx.Response(200, json={"text": "Page content here."})
    )
    be = _be()
    assert be.recognize(1, b"\x89PNG...") == "Page content here."
    assert route.called


@respx.mock
def test_local_markdown_response():
    respx.post("http://localhost:8000/file_parse").mock(
        return_value=httpx.Response(200, json={"md": "# Title\n\nBody."})
    )
    be = _be()
    assert be.recognize(1, b"\x89PNG...") == "# Title\n\nBody."


@respx.mock
def test_local_blocks_list_response():
    respx.post("http://localhost:8000/file_parse").mock(
        return_value=httpx.Response(
            200, json={"data": [{"text": "a"}, {"text": "b"}]}
        )
    )
    be = _be()
    assert be.recognize(1, b"img") == "a\nb"


@respx.mock
def test_local_non_2xx_raises_first_fault():
    respx.post("http://localhost:8000/file_parse").mock(
        return_value=httpx.Response(401)
    )
    be = _be()
    with pytest.raises(httpx.HTTPStatusError):
        be.recognize(1, b"img")


@respx.mock
def test_local_uses_file_form_field():
    route = respx.post("http://localhost:8000/file_parse")
    route.mock(return_value=httpx.Response(200, json={"text": "ok"}))
    _be().recognize(3, b"stuff")
    # The multipart form field must be named "file".
    body = route.calls.last.request.content
    assert b'name="file"' in body


def test_cloud_requires_upload_url_even_though_recognize_is_blocked():
    """In cloud mode, recognize() (single image upload) is unsupported."""
    be = MinerUBackend(api_key="appid:secret", base_url="https://api.mineru.net")
    with pytest.raises(RuntimeError, match="本地模式"):
        be.recognize(1, b"img")


def test_hmac_signer_deterministic():
    be = MinerUBackend(api_key="ak:S3Cr3t", base_url="https://api.mineru.net")
    s1 = be._cloud_sign("{}[]:,. xyz", "1234567890")
    s2 = be._cloud_sign("{}[]:,. xyz", "1234567890")
    assert s1 == s2
    assert len(s1) > 0


def test_cloud_submit_success_returns_job_id():
    with respx.mock:
        route = respx.post("https://api.mineru.net/api/v4/file_url/extract-db")
        route.mock(
            return_value=httpx.Response(
                200, json={"code": 0, "data": {"job_id": "job-abc"}}
            )
        )
        be = MinerUBackend(api_key="ak:S3Cr3t", base_url="https://api.mineru.net")
        assert be._cloud_submit(["https://x/y.pdf"], is_ocr=True) == "job-abc"
        assert route.called


def test_registry_produces_mineru_local_by_default():
    be = get_backend("mineru")  # no base_url → local default
    assert isinstance(be, MinerUBackend)
    assert "localhost:8000" in str(be._client.base_url)
