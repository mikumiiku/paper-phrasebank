"""PaddleOCR backend — local self-host.

PaddleOCR does **not** provide a hosted public OCR API. The standard way to use
it with ``ppb`` is to run the PaddleOCR HTTP service locally:

.. code-block:: bash

    docker run -d --name paddleocr \\
        -p 8866:8866 \\
        paddlepaddle/paddleocr:latest \\
        paddleocr serving start --port 8866

Then point ``ocr_base_url`` at ``http://localhost:8866`` (the default). We
expose that conventional address as ``DEFAULT_BASE_URL`` so the config UI can
autofill it — the user only changes it if they chose a different host/port.
"""
from __future__ import annotations

import httpx

from . import OcrBackend, http_client, register


@register("paddle")
class PaddleBackend:
    """PaddleOCR-compatible HTTP service (typically local)."""

    # Conventional local deployment address — the user only overrides when
    # they chose a different host/port. PaddleOCR has no official hosted API.
    DEFAULT_BASE_URL = "http://localhost:8866"
    name = "paddle"

    def __init__(self, *, api_key: str = "", base_url: str = "") -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        self._api_key = api_key
        self._client: httpx.Client = http_client(base_url)

    def recognize(self, page_num: int, image_bytes: bytes) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        files = {"image": (f"page_{page_num}.png", image_bytes, "image/png")}
        resp = self._client.post("/ocr", files=files, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if "text" in data:
                return str(data["text"])
            lines = data.get("results") or data.get("data") or []
        else:
            lines = data
        chunks: list[str] = []
        for line in lines:
            if isinstance(line, dict):
                chunks.append(str(line.get("text", "")))
            elif isinstance(line, (list, tuple)) and line:
                # PaddleOCR default shape: [[[box], (text, score)], ...]
                chunks.append(str(line[-1][0]) if isinstance(line[-1], (list, tuple)) else str(line[0]))
            elif isinstance(line, str):
                chunks.append(line)
        return "\n".join(c for c in chunks if c.strip())
