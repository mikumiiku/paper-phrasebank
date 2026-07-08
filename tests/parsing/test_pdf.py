"""PDF extraction: text pages, image-page detection, OCR injection."""
from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from phrasebank.parsing import TextPage, extract_text


def test_extracts_text_pages(make_pdf) -> None:
    path = make_pdf(
        [
            {"text": "Page one content.\nSecond line."},
            {"text": "Page two content here."},
        ]
    )
    pages = extract_text(path)

    assert len(pages) == 2
    assert all(isinstance(p, TextPage) for p in pages)
    assert pages[0].page_num == 1 and pages[1].page_num == 2
    assert not pages[0].is_image_page
    assert "Page one content." in pages[0].text
    assert "Page two content here." in pages[1].text


def test_detects_image_page_with_low_text(make_pdf) -> None:
    path = make_pdf(
        [
            {"text": "Normal page with plenty of readable text on it.", "image": False},
            {"text": "ab", "image": True},  # < 50 chars + has image
        ]
    )
    pages = extract_text(path)

    assert pages[0].is_image_page is False
    assert pages[1].is_image_page is True
    # No OCR injected -> text left empty
    assert pages[1].text == ""


def test_short_text_without_image_is_not_image_page(make_pdf) -> None:
    path = make_pdf([{"text": "Hi", "image": False}])
    pages = extract_text(path)

    assert pages[0].is_image_page is False
    assert pages[0].text == "Hi"


def test_ocr_callback_fills_image_page(make_pdf) -> None:
    captured: list[tuple[int, int]] = []

    def fake_ocr(page_num: int, page_bytes: bytes) -> str:
        captured.append((page_num, len(page_bytes)))
        return f"OCR result for page {page_num}"

    path = make_pdf(
        [
            {"text": "Normal page with plenty of readable text on it.", "image": False},
            {"text": "ab", "image": True},
        ]
    )
    pages = extract_text(path, ocr_fn=fake_ocr)

    assert pages[1].is_image_page is True
    assert pages[1].text == "OCR result for page 2"
    assert captured == [(2, captured[0][1])]
    assert captured[0][1] > 0  # real PNG bytes handed to OCR


def test_ocr_failure_raises(make_pdf) -> None:
    def bad_ocr(page_num: int, page_bytes: bytes) -> str:
        raise RuntimeError("boom")

    path = make_pdf([{"text": "ab", "image": True}])

    with pytest.raises(RuntimeError, match="OCR 回调在第 1 页失败"):
        extract_text(path, ocr_fn=bad_ocr)


def test_rejects_unknown_suffix(tmp_dir: Path) -> None:
    path = tmp_dir / "foo.txt"
    path.write_text("hi")
    with pytest.raises(ValueError, match="不支持的文件格式"):
        extract_text(path)
