"""MinerU OCR backend.

Calls MinerU v4 API to extract text from an image/PDF page. The exact request
shape here is based on the public MinerU API contract documented at the time
of writing; if MinerU's schema drifts, adjust this single file — consumers only
depend on the ``OcrBackend`` protocol, not these internals.
"""
from __future__ import annotations

import httpx

from . import OcrBackend, http_client, register


@register("mineru")
class MinerUBackend:
    """MinerU v4 <https://api.mineru.net/v4>.

    Auth via ``Authorization: Bearer <api_key>`` header. We POST the image to
    ``/extract`` and pull the joined ``text`` from the response.
    """

    name = "mineru"

    def __init__(self, *, api_key: str = "", base_url: str = "https://api.mineru.net/v4") -> None:
        if not base_url:
            base_url = "https://api.mineru.net/v4"
        self._api_key = api_key
        self._client: httpx.Client = http_client(base_url)

    def recognize(self, page_num: int, image_bytes: bytes) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        files = {"file": (f"page_{page_num}.png", image_bytes, "image/png")}
        resp = self._client.post("/extract", files=files, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Accept either top-level "text" or list of {text: ...} blocks.
        if isinstance(data, dict):
            if "text" in data:
                return str(data["text"])
            blocks = data.get("blocks") or data.get("data") or []
        else:
            blocks = data
        chunks: list[str] = []
        for b in blocks:
            if isinstance(b, dict) and "text" in b:
                chunks.append(str(b["text"]))
            elif isinstance(b, str):
                chunks.append(b)
        return "\n".join(c for c in chunks if c.strip())
