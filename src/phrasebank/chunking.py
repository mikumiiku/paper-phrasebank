"""Three-level degradable text chunking.

1. Section split — regex on common academic headings (DOCX headings already
   appear as plain text lines from the parser).
2. Natural paragraph split — double newline.
3. Punctuation fallback — from ``max_chunk_size`` forward to the nearest
   sentence terminator; never hard-cut at a character boundary.
"""
from __future__ import annotations

import re

_SECTION_RE = re.compile(
    r"^(?:Art\.\s*)?(?:\d+(?:\.\d+)*\.?\s+)?(?:Abstract|Introduction"
    r"|Related\s+Work|Background|Method(?:s|ology)?|Results?"
    r"|Discussion|Conclusions?|References|Experiment(?:s)?|Evaluation"
    r"|Acknowledg(?:ement)?s?)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Sentence terminators, ASCII forms expect trailing space.
_SENTENCE_END_RE = re.compile(r"[.?!;]\s|[。？；]")


def chunk(text_blocks: list[str], max_chunk_size: int = 2000) -> list[str]:
    """Split cleaned paragraph blocks into LLM-friendly chunks.

    Split on double newlines, then merge an isolated heading line into the
    paragraph that follows it — headings anchor their body, never stand alone.
    Oversized paragraphs fall back to sentence boundaries, never hard-cut.
    """
    merged: list[str] = []
    for block in text_blocks:
        pieces = [p for p in block.split("\n\n") if p.strip()]
        merged.extend(_merge_headings(pieces))

    out: list[str] = []
    for paragraph in merged:
        out.extend(_split_long(paragraph, max_chunk_size))
    return out


def _merge_headings(pieces: list[str]) -> list[str]:
    """Attach a lone heading line forward to the paragraph that follows it."""
    merged: list[str] = []
    pending: str | None = None
    for piece in pieces:
        if _SECTION_RE.match(piece):
            # A heading belongs to the next body paragraph — carry it.
            if pending is None:
                pending = piece
            else:
                # Two headings in a row: emit the orphan, keep the new one.
                merged.append(pending)
                pending = piece
            continue
        if pending is not None:
            merged.append(pending + "\n\n" + piece)
            pending = None
        else:
            merged.append(piece)
    if pending is not None:
        merged.append(pending)
    return merged


def _split_long(text: str, max_chunk_size: int) -> list[str]:
    """Break oversized paragraphs only at sentence boundaries."""
    if len(text) <= max_chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        rest = n - start
        if rest <= max_chunk_size:
            chunks.append(text[start:])
            break
        cut = _find_sentence_end(text, start + max_chunk_size, n)
        if cut is None:
            # No terminator ahead -> return the rest intact, never hard-cut.
            chunks.append(text[start:])
            break
        chunks.append(text[start:cut])
        start = cut
    return chunks


def _find_sentence_end(text: str, from_pos: int, limit: int) -> int | None:
    """Return the end position of the first sentence terminator at/after ``from_pos``.

    Returns ``None`` when no terminator exists in ``text[from_pos:limit]``.
    """
    if from_pos >= limit:
        return None
    window = text[from_pos:limit]
    m = _SENTENCE_END_RE.search(window)
    if m is None:
        return None
    return from_pos + m.end()
