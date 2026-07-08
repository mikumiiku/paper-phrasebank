"""DOCX extraction: heading-driven sections vs paragraph fallback."""
from __future__ import annotations

from phrasebank.parsing import TextPage, extract_text


def test_heading_sections(make_docx) -> None:
    path = make_docx(
        [
            ("Title", "My Paper"),
            ("Heading 1", "Introduction"),
            ("Normal", "Intro paragraph one."),
            ("Normal", "Intro paragraph two."),
            ("Heading 1", "Method"),
            ("Normal", "We propose a new method."),
        ]
    )
    pages = extract_text(path)

    assert len(pages) == 2
    intro, method = pages[0], pages[1]
    assert "Introduction" in intro.text
    assert "Intro paragraph one." in intro.text
    assert "Intro paragraph two." in intro.text
    assert "Method" in method.text
    assert "We propose a new method." in method.text
    assert all(not p.is_image_page for p in pages)


def test_paragraph_fallback_without_headings(make_docx) -> None:
    path = make_docx(
        [
            ("Normal", "First paragraph."),
            ("Normal", "Second paragraph."),
            ("Normal", "Third paragraph."),
        ]
    )
    pages = extract_text(path)

    assert len(pages) == 3
    assert pages[0].text == "First paragraph."
    assert pages[2].text == "Third paragraph."
    assert [p.page_num for p in pages] == [1, 2, 3]


def test_empty_docx(make_docx) -> None:
    path = make_docx([])
    assert extract_text(path) == []


def test_nested_heading_levels(make_docx) -> None:
    path = make_docx(
        [
            ("Heading 1", "Results"),
            ("Normal", "Top-level result."),
            ("Heading 2", "Ablation"),
            ("Normal", "Ablation details."),
        ]
    )
    pages = extract_text(path)
    # Heading 1 and Heading 2 both start new sections
    assert len(pages) == 2
    assert "Results" in pages[0].text
    assert "Ablation" in pages[1].text
