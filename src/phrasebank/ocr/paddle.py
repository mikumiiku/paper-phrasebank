"""PaddleOCR API backend.

Generic contract for an HTTP-exposed PaddleOCR-compatible service: POST raw
image bytes to ``/ocr``, receive ``{text: "..."}`` (or list of recognised
lines). If your deployment uses a different route/shape, adjust here.
"""
from __future__ import annotations

import httpx

from . import OcrBackend, http_client, register


@register("paddle")
class PaddleBackend:
    """Generic PaddleOCR-compatible HTTP service."""

    name = "paddle"

    def __init__(self, *, api_key: str = "", base_url: str = "") -> None:
        if not base_url:
            raise ValueError("PaddleOCR 需要提供 base_url")
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
