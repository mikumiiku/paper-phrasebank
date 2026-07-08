"""Clean: header/footer stripping, page numbers, citation superscripts."""
from __future__ import annotations

from phrasebank.parsing import TextPage
from phrasebank.parsing.clean import page_blocks


def _pages(texts: list[str]) -> list[TextPage]:
    return [TextPage(page_num=i + 1, text=t, is_image_page=False) for i, t in enumerate(texts)]


def test_strips_repeated_header_footer() -> None:
    # header "Journal of X" and footer "2024 ACM" appear on all 4 pages (>30%)
    pages = _pages([
        "Journal of X\n\nReal content one.\n\n2024 ACM",
        "Journal of X\n\nReal content two.\n\n2024 ACM",
        "Journal of X\n\nReal content three.\n\n2024 ACM",
        "Journal of X\n\nReal content four is longer text here.\n\n2024 ACM",
    ])
    blocks = page_blocks(pages)

    assert len(blocks) == 4
    for b in blocks:
        assert "Journal of X" not in b
        assert "2024 ACM" not in b
    assert "Real content one." in blocks[0]


def test_does_not_strip_rare_repeated_line() -> None:
    # repeated on only 1 of 4 pages (25% < 30%) -> keep
    pages = _pages([
        "Rare line\n\nBody A.",
        "Other head\n\nBody B.",
        "Third head\n\nBody C.",
        "Fourth head\n\nBody D.",
    ])
    blocks = page_blocks(pages)
    assert "Rare line" in blocks[0]


def test_strips_isolated_page_number() -> None:
    pages = _pages([
        "42\n\nImportant text here.",
        "Important text continued.\n\n7",
    ])
    blocks = page_blocks(pages)

    assert "42" not in blocks[0]
    assert "7" not in blocks[1]
    assert "Important text" in blocks[0]


def test_strips_citation_superscripts() -> None:
    pages = _pages([
        "Prior work established this [1] and extended it [12, 13] widely.",
    ])
    blocks = page_blocks(pages)

    assert "[1]" not in blocks[0]
    assert "[12, 13]" not in blocks[0]
    assert "Prior work established this" in blocks[0]
    assert "and extended it" in blocks[0]
    assert "widely." in blocks[0]


def test_empty_input_returns_empty() -> None:
    assert page_blocks([]) == []


def test_whitespace_only_page_dropped() -> None:
    pages = _pages(["   \n  \n", "Real content."])
    blocks = page_blocks(pages)
    assert blocks == ["Real content."]
