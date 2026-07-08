"""MinerU OCR backend.

MinerU provides a hosted OCR-as-a-service API. The official current base URL
is ``https://api.mineru.net/v4`` (MinerU 4.x line; see
https://opendatalab.com/OpenDataLab/MinerU/docs for the registry's API
documentation). We default to that so the user never needs to fill in a URL.

The user may still set ``ocr_base_url`` in config to self-host / use a
corporate proxy — ``get_backend`` honours that override.
"""
from __future__ import annotations

import httpx

from . import OcrBackend, http_client, register


@register("mineru")
class MinerUBackend:
    """MinerU v4 hosted OCR: https://api.mineru.net/v4.

    Auth via ``Authorization: Bearer <api_key>`` header. We POST the page image
    to ``/extract`` and pull the joined ``text`` from the response.
    """

    # Official hosted API base URL — users should NOT learn this.
    DEFAULT_BASE_URL = "https://api.mineru.net/v4"
    name = "mineru"

    def __init__(self, *, api_key: str = "", base_url: str = "") -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
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
