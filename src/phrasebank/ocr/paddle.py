"""PaddleOCR backend.

Uses the PaddleOCR official cloud REST API (undocumented in upstream except
for the source's own default) plus, optionally, a local PaddleOCR serving
endpoint. Picking between the two is based on ``base_url``:

  - **Official cloud (DEFAULT_BASE_URL)** — job-submission pattern:
    ``POST /api/v2/ocr/jobs`` → returns a ``jobId``, then
    ``GET /api/v2/ocr/jobs/{job_id}`` until the job completes. Bearer-token
    auth via ``PADDLEOCR_ACCESS_TOKEN``. Free tier is 20 000 pages/day. This
    is the recommended default because users do not need to self-host; they
    only need an access token from
    https://aistudio.baidu.com/account/accessToken.

  - **Local (opt-in)** — any base URL not equal to the cloud address ends up
    in simple ``POST /ocr`` multipart mode (field ``image``) for a quick
    self-hosted PaddleOCR HTTP service."""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from . import OcrBackend, http_client, register


@register("paddle")
class PaddleBackend:
    """PaddleOCR — official cloud (default) or local self-host."""

    DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com"
    API_PATH = "/api/v2/ocr/jobs"
    _POLL_DONE = {"DONE", "FAILED", "CANCELLED"}
    name = "paddle"

    def __init__(self, *, api_key: str = "", base_url: str = "") -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        self._token = api_key
        self._cloud_mode = base_url.rstrip("/") == self.DEFAULT_BASE_URL.rstrip("/")
        self._client = http_client(base_url)

    # ── public API ────────────────────────────────────────────────────────

    def recognize(self, page_num: int, image_bytes: bytes) -> str:
        """Submit a single page image to the backend and return OCR text.

        On the cloud path the image is uploaded to Paddle's Object Storage
        first; for a single page this is overkill but we want the code path to
        be honest. To keep usage simple and avoid a dependency on Paddle's
        signed-upload flow, cloud mode currently dispatches into a single-job
        document-parse request with an inline base64 payload. Local mode
        posts directly to ``/ocr``.
        """
        if self._cloud_mode:
            return self._cloud_recognize(page_num, image_bytes)
        return self._local_recognize(page_num, image_bytes)

    # ── local mode ───────────────────────────────────────────────────────

    def _local_recognize(self, page_num: int, image_bytes: bytes) -> str:
        files = {"image": (f"page_{page_num}.png", image_bytes, "image/png")}
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        resp = self._client.post("/ocr", files=files, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return _extract_text_from_paddle_response(data)

    # ── cloud mode ───────────────────────────────────────────────────────

    def _cloud_recognize(self, page_num: int, image_bytes: bytes) -> str:
        """Upload the page to the official API and poll for the result."""
        encoded = "/9j/4AAQSkZJRg==" if False else None  # placeholder
        del encoded
        submit_payload = {
            "file": _b64_first_page(image_bytes),
            "fileName": f"page_{page_num}.png",
            "model": "PP-StructureV3",
            "pageRanges": str(page_num),
        }
        resp = self._client.post(
            self.API_PATH,
            json=submit_payload,
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        job = resp.json()
        if job.get("code") != 0:
            raise RuntimeError(f"PaddleOCR cloud submit failed: {job.get('msg')}")
        job_id = (job.get("data") or {}).get("jobId") or job.get("jobId")
        if not job_id:
            raise RuntimeError(f"PaddleOCR cloud no jobId: {job}")
        return self._cloud_poll_results(str(job_id))

    def _cloud_poll_results(self, job_id: str, *, timeout: float = 300.0,
                            poll_interval: float = 2.0) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._client.get(
                f"{self.API_PATH}/{job_id}",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            info = data.get("data") or {}
            status = info.get("status")
            if status == "DONE":
                return _cloud_assemble(info)
            if status in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"PaddleOCR cloud job {job_id} failed: {info}")
            time.sleep(poll_interval)
        raise TimeoutError(f"PaddleOCR cloud job {job_id} did not finish in {timeout}s")

    # ── helpers ───────────────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        if self._token:
            return {"Authorization": f"Bearer {self._token}", "Client-Platform": "ppb"}
        return {}


def _b64_first_page(image_bytes: bytes) -> str:
    import base64
    return base64.b64encode(image_bytes).decode()


def _extract_text_from_paddle_response(data: Any) -> str:
    """Recover OCR text from a variety of (undocumented) response shapes."""
    if isinstance(data, dict):
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        pruned = data.get("prunedResult") or data.get("results") or {}
        if isinstance(pruned, dict):
            texts = pruned.get("rec_texts") or pruned.get("texts") or []
            if isinstance(texts, list):
                return "\n".join(str(t) for t in texts if t)
        items = data.get("data") or data.get("blocks") or []
    elif isinstance(data, list):
        items = data
    else:
        return ""
    chunks: list[str] = []
    for b in items:
        if isinstance(b, dict):
            for k in ("text", "rec_texts", "content", "md", "markdown"):
                v = b.get(k)
                if isinstance(v, str):
                    chunks.append(v)
                    break
                if isinstance(v, list):
                    chunks.extend(str(x) for x in v if x)
                    break
        elif isinstance(b, str):
            chunks.append(b)
    return "\n".join(c for c in chunks if c.strip())


def _cloud_assemble(info: dict[str, Any]) -> str:
    """Unpack a DONE job into a single text blob."""
    pages = info.get("pages") or info.get("result", {}).get("pages") or []
    texts: list[str] = []
    for p in pages:
        md = p.get("markdownText")
        if md:
            texts.append(md)
            continue
        pruned = p.get("prunedResult") or {}
        rec = pruned.get("rec_texts") or []
        if isinstance(rec, list) and rec:
            texts.extend(str(t) for t in rec)
    return "\n\n".join(t for t in texts if t.strip())
