"""MinerU OCR backend.

MinerU offers two distinct ways to call OCR. This module supports **both** and
picks one based on what the user configured.

1. **Local (default)** — a `mineru-api` FastAPI service running on
   ``http://localhost:8000`` (the canonical ``pip install "mineru[all]" &&
   mineru-api --port 8000`` deployment). Multipart POST to ``/file_parse`` with
   field name ``file``; synchronous text return. This is the **recommended**
   option because the cloud API's signing spec is not published and subject to
   drift.

2. **Cloud (optional, opt-in)** — the mineru.net hosted API at
   ``https://api.mineru_net``. The user must obtain an ``app_id`` /
   ``secret_key`` pair from the mineru.net console, set the cloud base URL, and
   provide the credentials via the ``api_key`` field (encoded as
   ``"<app_id>:<secret_key>"``). The cloud API is URL-only: pass
   ``file_urls=[pdf_url]`` and it returns a job id; we poll the result.

Local is the default to avoid forcing users onto an undocumented signing
contract they do not control.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import httpx

from . import OcrBackend, http_client, register


@register("mineru")
class MinerUBackend:
    """MinerU OCR backend — local or cloud, chosen by ``base_url``.

    ``api_key`` semantics: ignored for local; for cloud, the string must be
    ``"<app_id>:<secret_key>"`` (colon-separated pair) so the HMAC signer can
    derive both values.
    """

    DEFAULT_BASE_URL = "http://localhost:8000"
    name = "mineru"

    # Cloud constants (undocumented by upstream — used only when the user
    # explicitly sets ``Settings.ocr_base_url`` to the cloud address).
    _CLOUD_BASE_URL = "https://api.mineru.net"
    _CLOUD_SUBMIT = "/api/v4/file_url/extract-db"
    _CLOUD_POLL_TPL = "/api/v4/file_url/extract-db/{job_id}"

    def __init__(self, *, api_key: str = "", base_url: str = "") -> None:
        if not base_url:
            base_url = self.DEFAULT_BASE_URL
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._cloud_mode = self._base_url == self._CLOUD_BASE_URL
        self._client = http_client(self._base_url)

    # ── public API ────────────────────────────────────────────────────────

    def recognize(self, page_num: int, image_bytes: bytes) -> str:
        if self._cloud_mode:
            raise RuntimeError(
                "MinerU 云模式不接受单页图片上传，必须使用本地模式 "
                "(mineru-api --port 8000) 或将 ocr_base_url 清空。"
            )
        return self._local_recognize(page_num, image_bytes)

    # ── local mode ───────────────────────────────────────────────────────

    def _local_recognize(self, page_num: int, image_bytes: bytes) -> str:
        files = {"file": (f"page_{page_num}.png", image_bytes, "image/png")}
        resp = self._client.post("/file_parse", files=files)
        resp.raise_for_status()
        data = resp.json()
        # Compatible with the two known response shapes:
        #   { "md": "..." }          (markdown dump)
        #   { "text": "..." }        (plain text)
        #   { "data": [{"text": ...}]}
        if isinstance(data, dict):
            for k in ("md", "text", "content", "markdown"):
                if k in data and isinstance(data[k], str):
                    return data[k]
            for key in ("data", "blocks"):
                blocks = data.get(key)
                if isinstance(blocks, list):
                    return _flatten_blocks(blocks)
        elif isinstance(data, list):
            return _flatten_blocks(data)
        return ""

    # ── cloud mode (async URL submission + polling) ───────────────────────

    def submit_and_poll(self, *, file_urls: list[str], is_ocr: bool = True,
                        timeout: float = 120.0, poll_interval: float = 2.0) -> str:
        """Full cloud job lifecycle — used by user-facing wrappers that already
        have a hosted PDF URL. Returns the assembled markdown/text."""
        job_id = self._cloud_submit(file_urls, is_ocr)
        return self._cloud_poll(job_id, timeout=timeout, poll_interval=poll_interval)

    def _cloud_submit(self, file_urls: list[str], is_ocr: bool) -> str:
        payload = json.dumps({
            "file_urls": file_urls,
            "is_ocr": is_ocr,
        }, ensure_ascii=False)
        ts = str(int(time.time()))
        sig = self._cloud_sign(payload, ts)
        app_id = (self._api_key or "").split(":")[0] if self._api_key else ""
        headers = {
            "Content-Type": "application/json",
            "X-App-Id": app_id,
            "X-Timestamp": ts,
            "X-Signature": sig,
        }
        resp = self._client.post(
            self._CLOUD_SUBMIT, content=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"MinerU cloud submit failed: {data.get('msg')}")
        job_id = (data.get("data") or {}).get("job_id") or data.get("job_id")
        if not job_id:
            raise RuntimeError(f"MinerU cloud submit returned no job_id: {data}")
        return str(job_id)

    def _cloud_poll(self, job_id: str, *, timeout: float,
                    poll_interval: float) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._client.get(self._CLOUD_POLL_TPL.format(job_id=job_id))
            resp.raise_for_status()
            data = resp.json()
            status = (data.get("data") or {}).get("status")
            if status == "done":
                d = data.get("data") or {}
                return "\n".join(
                    filter(None, (d.get("markdown"), d.get("content"), d.get("md")))
                )
            if status == "failed":
                raise RuntimeError(f"MinerU cloud job failed: {data}")
            time.sleep(poll_interval)
        raise TimeoutError(f"MinerU cloud job {job_id} did not finish in {timeout}s")

    # ── helpers ───────────────────────────────────────────────────────────

    def _cloud_sign(self, payload: str, timestamp: str) -> str:
        if not self._api_key or ":" not in self._api_key:
            raise RuntimeError("MinerU 云模式 api_key 格式：<app_id>:<secret_key>")
        secret_key = self._api_key.split(":", 1)[1]
        msg = f"{timestamp}.{payload}".encode()
        sig = hmac.new(secret_key.encode(), msg, hashlib.sha256).digest()
        return base64.b64encode(sig).decode()


def _flatten_blocks(blocks: list[Any]) -> str:
    chunks: list[str] = []
    for b in blocks:
        if isinstance(b, dict):
            for k in ("text", "content", "md", "markdown"):
                v = b.get(k)
                if isinstance(v, str):
                    chunks.append(v)
                    break
        elif isinstance(b, str):
            chunks.append(b)
    return "\n".join(c for c in chunks if c.strip())
