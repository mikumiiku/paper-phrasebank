"""E2E: OCR backend for image-only PDF pages.

Real MinerU/PaddleOCR API is slow and rate-limited, so we mock the
underlying HTTP endpoint (respx) to verify the integration — the parser
detects image pages, the OCR callback fires, the response text flows back
into the page object, and the pipeline downstream sees it as if OCR had
run for real.
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
    # page 1: normal text page (survives without OCR)
    p = doc.new_page()
    p.insert_text((72, 72), "Normal page with text. Introduction line.")
    # page 2: image-only page (few chars + embedded image)
    p = doc.new_page()
    p.insert_text((72, 72), "1")  # barely any text
    p.insert_image(fitz.Rect(0, 0, 200, 200), stream=_tiny_png())
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def image_pdf(tmp_dir: Path) -> Path:
    return _make_pdf_with_image_page(tmp_dir / "imagepaper.pdf")


@pytest.fixture
def ocr_mineru():
    from phrasebank.ocr import get_backend
    return get_backend("mineru", api_key="sk-e2e")


def test_parses_text_page_without_ocr(image_pdf):
    from phrasebank.parsing import extract_text
    pages = extract_text(image_pdf, ocr_fn=None)
    assert len(pages) == 2
    assert pages[0].text.strip().startswith("Normal page")
    assert pages[1].is_image_page is True
    assert pages[1].text.strip() == ""  # no OCR → stays empty


@respx.mock
def test_parses_image_page_with_ocr_callback(image_pdf, ocr_mineru):
    """Page 2 should have its text filled in by the OCR callback."""
    from phrasebank.parsing import extract_text

    route = respx.post("https://api.mineru.net/v4/extract").mock(
        return_value=httpx.Response(200, json={"text": "Recovered image text."})
    )

    def ocr_fn(page_num: int, image_bytes: bytes) -> str:
        return ocr_mineru.recognize(page_num, image_bytes)

    pages = extract_text(image_pdf, ocr_fn=ocr_fn)
    assert route.called
    assert len(pages) == 2
    assert pages[1].is_image_page is True
    assert pages[1].text.strip() == "Recovered image text."


@respx.mock
def test_mineru_called_for_image_page(image_pdf):
    """End-to-end wiring: parser invokes the MinerU endpoint on image pages."""
    from phrasebank.parsing import extract_text

    # Mock the real MinerU hosted endpoint.
    route = respx.post("https://api.mineru.net/v4/extract").mock(
        return_value=httpx.Response(200, json={"text": "OCR-recovered text from page."})
    )

    from phrasebank.ocr import get_backend
    be = get_backend("mineru", api_key="sk-e2e")

    def ocr_fn(page_num: int, image_bytes: bytes) -> str:
        return be.recognize(page_num, image_bytes)

    pages = extract_text(image_pdf, ocr_fn=ocr_fn)
    assert route.called
    assert pages[1].text.strip() == "OCR-recovered text from page."


@respx.mock
def test_paddleocr_called_for_image_page(image_pdf):
    """End-to-end wiring: parser invokes the PaddleOCR endpoint on image pages."""
    from phrasebank.parsing import extract_text

    route = respx.post("http://localhost:8866/ocr").mock(
        return_value=httpx.Response(200, json={"text": "Paddle text recovered."})
    )

    from phrasebank.ocr import get_backend
    be = get_backend("paddle", api_key="", base_url="http://localhost:8866")

    def ocr_fn(page_num: int, image_bytes: bytes) -> str:
        return be.recognize(page_num, image_bytes)

    pages = extract_text(image_pdf, ocr_fn=ocr_fn)
    assert route.called
    assert pages[1].text.strip() == "Paddle text recovered."


def test_pipeline_skips_image_pages_without_ocr(image_pdf, isolated_config):
    """With no OCR configured, image pages do not crash the pipeline — they
    are detected, skipped (empty text), and contribute no chunks."""
    from phrasebank.parsing import extract_text
    from phrasebank.parsing.clean import page_blocks
    from phrasebank.chunking import chunk

    pages = extract_text(image_pdf, ocr_fn=None)
    blocks = page_blocks(pages)
    chunks_ = chunk(blocks)
    # Page 1's text still flows through; image page contributes nothing.
    assert any("Normal page" in c for c in chunks_)
    # And never crashed.
