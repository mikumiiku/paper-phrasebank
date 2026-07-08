"""Parsing: PDF/DOCX -> list[TextPage].

Public entry :func:`extract_text` dispatches by file suffix.
OCR is injected via ``ocr_fn`` so the OCR backend (Phase D) plugs in
without this module knowing its concrete type.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

OcrFn = Callable[[int, bytes], str]


@dataclass
class TextPage:
    """One logical page of extracted text."""

    page_num: int
    text: str
    is_image_page: bool


def extract_text(path: str | Path, ocr_fn: OcrFn | None = None) -> list[TextPage]:
    """Extract pages from a PDF or DOCX file.

    Args:
        path: input file; dispatch is by suffix (``.pdf`` / ``.docx``).
        ocr_fn: optional ``(page_num, page_bytes) -> str`` used on image-only
            PDF pages. When ``None``, image pages are flagged with empty text.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        from .pdf import extract_pdf

        return extract_pdf(p, ocr_fn=ocr_fn)
    if suffix == ".docx":
        from .docx import extract_docx

        return extract_docx(p)
    raise ValueError(f"不支持的文件格式: {suffix}（仅支持 .pdf / .docx）")
