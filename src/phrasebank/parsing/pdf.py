"""PDF extraction branch (PyMuPDF).

Per page: pull the text layer, detect image-only pages (few chars + has
images), and optionally run the injected OCR callback to fill those pages.
"""
from __future__ import annotations

from pathlib import Path

import fitz

from . import OcrFn, TextPage

# Below this char count, a page with images is treated as image-only.
IMAGE_PAGE_TEXT_THRESHOLD = 50


def _normalize(text: str) -> str:
    """Strip per-line trailing whitespace; pymupdf inserts stray newlines."""
    return "\n".join(ln.rstrip() for ln in text.splitlines()).rstrip()


def extract_pdf(path: Path, *, ocr_fn: OcrFn | None = None) -> list[TextPage]:
    doc = fitz.open(path)
    try:
        pages: list[TextPage] = []
        for idx, page in enumerate(doc):
            text = _normalize(page.get_text())
            has_images = bool(page.get_images())
            is_image = len(text.strip()) < IMAGE_PAGE_TEXT_THRESHOLD and has_images
            if is_image:
                text = _ocr_page(page, idx + 1, ocr_fn) if ocr_fn is not None else ""
            pages.append(TextPage(page_num=idx + 1, text=text, is_image_page=is_image))
        return pages
    finally:
        doc.close()


def _ocr_page(page: fitz.Page, page_num: int, ocr_fn: OcrFn) -> str:
    """Render page to PNG bytes and hand it to the OCR callback."""
    pix = page.get_pixmap()
    try:
        return ocr_fn(page_num, pix.tobytes("png"))
    except Exception as exc:
        raise RuntimeError(f"OCR 回调在第 {page_num} 页失败") from exc
