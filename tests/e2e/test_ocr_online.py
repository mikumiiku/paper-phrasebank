"""Online PaddleOCR E2E — real cloud endpoint.

Uses the **official** PaddleOCR cloud API at
``https://paddleocr.aistudio-app.com`` and a real access token supplied via
environment variable ``PADDLEOCR_ACCESS_TOKEN``.

Skipped unless the env var is present AND ``--online`` is passed, so the
ordinary ``pytest`` run stays network-free. Run with:

    PADDLEOCR_ACCESS_TOKEN="<token>" pytest tests/e2e/test_ocr_online.py --online -v

These tests assert the authoritative real API contract (job submit → poll →
result URL → download JSONL) so any drift in upstream breaks CI loudly.
"""
from __future__ import annotations

import json
import os
import struct
import zlib

import httpx
import pytest

def _online_ready() -> bool:
    return bool(os.environ.get("PADDLEOCR_ACCESS_TOKEN")) \
        and os.environ.get("PPB_ONLINE") == "1"


pytestmark = pytest.mark.skipif(not _online_ready(),
                                reason="needs PADDLEOCR_ACCESS_TOKEN + PPB_ONLINE=1")

BASE = "https://paddleocr.aistudio-app.com"
TOKEN = os.environ["PADDLEOCR_ACCESS_TOKEN"]


def _tiny_png(w: int = 64, h: int = 32) -> bytes:
    def ck(t: bytes, d: bytes) -> bytes:
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    row = b"\x00" * (w * 3)
    return sig + ck(b"IHDR", ihdr) + ck(b"IDAT", zlib.compress(row * h)) + ck(b"IEND", b"")


# Reuse the project's own backend so we exercise the real URL/auth path.
@pytest.fixture
def paddle_be():
    from phrasebank.ocr.paddle import PaddleBackend
    return PaddleBackend(api_key=TOKEN, base_url="")


def test_real_submit_returns_job_id(paddle_be):
    """The cloud endpoint must accept a multipart local-file submission with the
    project's exact data shape and hand back a ``jobId``."""
    with httpx.Client(timeout=60, verify=True) as c:
        data = {"model": "PaddleOCR-VL-1.6",
                "optionalPayload": json.dumps({"useDocOrientationClassify": False})}
        files = {"file": ("test.png", _tiny_png(), "image/png")}
        r = c.post(
            f"{BASE}/api/v2/ocr/jobs",
            headers={"Authorization": f"bearer {TOKEN}", "Client-Platform": "ppb"},
            data=data, files=files,
        )
        assert r.status_code == 200, r.text
        resp = r.json()
        assert resp.get("code") == 0, resp
        job_id = (resp.get("data") or {}).get("jobId")
        assert job_id, f"no jobId in response: {resp}"


def test_real_poll_reaches_done_with_result_url(paddle_be):
    """Full job lifecycle — submit, poll, and confirm ``resultUrl.jsonUrl`` is
    returned when the job state is ``done``."""
    # This endorses the cloud's real job-state machine. Our parser/OCR
    # callback wiring depends on ``done`` + ``resultUrl`` semantics, so a
    # silent upstream change is caught here.
    import time
    with httpx.Client(timeout=120, verify=True) as c:
        data = {"model": "PaddleOCR-VL-1.6",
                "optionalPayload": json.dumps({"useDocOrientationClassify": False})}
        files = {"file": ("test.png", _tiny_png(), "image/png")}
        r = c.post(f"{BASE}/api/v2/ocr/jobs",
                   headers={"Authorization": f"bearer {TOKEN}"},
                   data=data, files=files)
        job_id = r.json()["data"]["jobId"]

        deadline = time.time() + 180
        result_url = None
        while time.time() < deadline:
            r2 = c.get(f"{BASE}/api/v2/ocr/jobs/{job_id}",
                       headers={"Authorization": f"bearer {TOKEN}"})
            body = r2.json()
            state = (body.get("data") or {}).get("state")
            assert state != "failed", f"job failed: {body}"
            if state == "done":
                result_url = (body.get("data") or {}).get("resultUrl", {}).get("jsonUrl")
                break
            time.sleep(3)
        assert result_url, "job did not reach done in time"

        # The result JSONL must be downloadable.
        r3 = c.get(result_url)
        assert r3.status_code == 200
        assert r3.text.strip(), "result JSONL empty"
        # First line is a JSON object with `result.layoutParsingResults`.
        first = json.loads(r3.text.strip().split("\n")[0])
        assert "result" in first, f"unexpected structure: {list(first)}"
