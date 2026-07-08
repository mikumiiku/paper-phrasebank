"""DOCX extraction branch (python-docx).

Emits one ``TextPage`` per section. A section starts at each paragraph whose
style name begins with ``Heading``; the heading text is prepended as a plain
line so the chunker's section regex can find it. When no heading style is
used anywhere, fall back to one page per non-empty paragraph.
"""
from __future__ import annotations

from pathlib import Path

from docx import Document

from . import TextPage


def extract_docx(path: Path) -> list[TextPage]:
    doc = Document(path)
    paragraphs = [p for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return []

    section_idxs = [i for i, p in enumerate(paragraphs) if _is_section_heading(p)]
    if section_idxs:
        # Discard front matter (Title/authors) before the first real section.
        return _by_sections(paragraphs[section_idxs[0] :])
    if any(_is_heading(p) for p in paragraphs):
        # Only Title/Subtitle present — treat whole doc as one block.
        return [TextPage(page_num=1, text=_join(paragraphs), is_image_page=False)]
    return _by_paragraphs(paragraphs)


def _style_level(p) -> int:
    """Return heading level (1..9) or 0 if not a heading."""
    name = (p.style.name or "").lower()
    if name.startswith("heading"):
        return int(name.split()[-1]) if name.split()[-1].isdigit() else 1
    return 0


def _is_section_heading(p) -> bool:
    """A section anchor: a Heading style at level >= 1 (not Title)."""
    return _style_level(p) >= 1


def _is_heading(p) -> bool:
    name = (p.style.name or "").lower()
    return _style_level(p) >= 1 or name == "title"


def _join(paragraphs) -> str:
    return "\n\n".join(p.text.strip() for p in paragraphs).strip()


def _by_sections(paragraphs) -> list[TextPage]:
    pages: list[TextPage] = []
    title: str | None = None
    body: list[str] = []
    seq = 0

    def flush() -> None:
        nonlocal title, body, seq
        if title is None and not body:
            return
        text = "\n\n".join(([title] if title else []) + body).strip()
        if text:
            seq += 1
            pages.append(TextPage(page_num=seq, text=text, is_image_page=False))
        title, body = None, []

    for p in paragraphs:
        if _is_section_heading(p):
            flush()
            title = p.text.strip()
        else:
            body.append(p.text.strip())
    flush()
    return pages


def _by_paragraphs(paragraphs) -> list[TextPage]:
    return [
        TextPage(page_num=i + 1, text=p.text.strip(), is_image_page=False)
        for i, p in enumerate(paragraphs)
    ]
