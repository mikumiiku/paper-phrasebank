"""Regex cleaning of parsed pages into paragraph blocks.

Removes:
  * running headers / footers (top/bottom lines repeated across >=30% pages),
  * isolated page numbers (a lone integer at the top or bottom of a page),
  * citation superscripts like ``[1]`` / ``[12, 13]``.
"""
from __future__ import annotations

import re
from collections import Counter

from . import TextPage

_HEADER_FOOTER_RATIO = 0.30
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_ISOLATED_PAGE_NUM_RE = re.compile(r"^\s*\-?\d+\s*$")


def page_blocks(pages: list[TextPage]) -> list[str]:
    if not pages:
        return []

    n = len(pages)
    # A header/footer must repeat on at least ~30% of pages, and repeating on
    # a single page is never enough — hence the floor of 2.
    threshold = max(2, int(n * _HEADER_FOOTER_RATIO))

    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()
    split_lines: list[list[str]] = []

    for pg in pages:
        lines = [ln.strip() for ln in pg.text.splitlines() if ln.strip()]
        split_lines.append(lines)
        if lines:
            top_counter[lines[0]] += 1
            bottom_counter[lines[-1]] += 1

    headers = {ln for ln, c in top_counter.items() if c >= threshold}
    footers = {ln for ln, c in bottom_counter.items() if c >= threshold}

    blocks: list[str] = []
    for lines in split_lines:
        cleaned = _strip_noise(lines, headers, footers)
        text = "\n".join(cleaned)
        text = _CITATION_RE.sub("", text)
        text = _collapse_blank_lines(text)
        if text:
            blocks.append(text)
    return blocks


def _strip_noise(
    lines: list[str], headers: set[str], footers: set[str]
) -> list[str]:
    """Drop header/footer lines and isolated page numbers at the edges."""
    while lines and (lines[0] in headers or _ISOLATED_PAGE_NUM_RE.match(lines[0])):
        lines = lines[1:]
    while lines and (lines[-1] in footers or _ISOLATED_PAGE_NUM_RE.match(lines[-1])):
        lines = lines[:-1]
    return lines


def _collapse_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
