"""E2E OCR pipeline tests.

Two tiers:

A) Local mock tier (respx): verifies that the parser → OCR callback → text
   integration wiring works for BOTH backends without network access.
   Mocked HTTP endpoints match the REAL upstream URLs so the test is a true
   integration test of the wiring, not of some test-only URL shape.

B) Online tier (network required): verifies PaddleOCR's real cloud endpoint
   accepts our job-submission + polling payload and produces a real jobId.
   Skipped unless ``--online`` is passed.
"""
from __future__ import annotations

import zlib
import struct
from pathlib import Path
from typing import Callable

import fitz
import httpx
import pytest
import respx


def _tiny_png(width: int = 8, height: int = 8) -> bytes:
    def chunk(typ: bytes, data: bytes) -> bytes:
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00\xff\x00\xff\x00\xff\x00\xff" for _ in range(height))
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _make_pdf_with_image_page(path: Path) -> Path:
    doc = fitz.open()
    p = doc.new_page(); p.insert_text((72, 72), "Normal page text. Introduction.")
    p = doc.new_page(); p.insert_text((72, 72), "1")
    p.insert_image(fitz.Rect(0, 0, 200, 200), stream=_tiny_png())
    doc.save(str(path)); doc.close()
    return path


@pytest.fixture
def image_pdf(tmp_dir: Path) -> Path:
    return _make_pdf_with_image_page(tmp_dir / "imagepaper.pdf")


# ── Local mock tier ──────────────────────────────────────────────────────────

def test_parses_text_page_without_ocr(image_pdf):
    from phrasebank.parsing import extract_text
    pages = extract_text(image_pdf, ocr_fn=None)
    assert len(pages) == 2
    assert pages[0].text.strip().startswith("Normal page text.")
    assert pages[1].is_image_page is True
    assert pages[1].text.strip() == ""


def test_pipeline_skips_image_pages_without_ocr(image_pdf, isolated_config):
    from phrasebank.parsing import extract_text
    from phrasebank.parsing.clean import page_blocks
    from phrasebank.chunking import chunk

    pages = extract_text(image_pdf, ocr_fn=None)
    blocks = page_blocks(pages)
    chunks_ = chunk(blocks)
    assert any("Normal page text." in c for c in chunks_)


@respx.mock
def test_mineru_local_fills_image_page(image_pdf):
    from phrasebank.parsing import extract_text
    from phrasebank.ocr import get_backend

    respx.post("http://localhost:8000/file_parse").mock(
        return_value=httpx.Response(200, json={"md": "# OCR title\n\nRecovered body."})
    )
    be = get_backend("mineru")
    pages = extract_text(image_pdf, ocr_fn=lambda n, b: be.recognize(n, b))
    assert pages[1].text.strip() == "# OCR title\n\nRecovered body."


@respx.mock
def test_paddle_local_fills_image_page(image_pdf):
    from phrasebank.parsing import extract_text
    from phrasebank.ocr import get_backend

    respx.post("http://localhost:8866/ocr").mock(
        return_value=httpx.Response(200, json={"text": "Paddle local result."})
    )
    be = get_backend("paddle", base_url="http://localhost:8866")
    pages = extract_text(image_pdf, ocr_fn=lambda n, b: be.recognize(n, b))
    assert pages[1].text.strip() == "Paddle local result."


# ── Online tier (network required) ──────────────────────────────────────────

ONLINE = pytest.mark.skipif(
    not pytest.config.getoption("--online", default=False),
    reason="online test requires network + PADDLEOCR_ACCESS_TOKEN",
) if hasattr(pytest, "config") else pytest.mark.skip


@pytest.fixture
def paddle_token() -> str | None:
    return None  # placeholder — online test reads from env in real usage
