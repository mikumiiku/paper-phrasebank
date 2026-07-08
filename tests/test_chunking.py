"""Chunking: three-level degradable splitting."""
from __future__ import annotations

from phrasebank.chunking import chunk


def test_paragraph_split_on_double_newline() -> None:
    blocks = ["First paragraph here.\n\nSecond paragraph here. More text."]
    out = chunk(blocks, max_chunk_size=2000)

    assert out == ["First paragraph here.", "Second paragraph here. More text."]


def test_section_split_by_heading() -> None:
    blocks = [
        "Introduction\n\nWe study an important problem.\n\n"
        "Method\n\nWe propose a novel approach."
    ]
    out = chunk(blocks, max_chunk_size=2000)

    assert out[0] == "Introduction\n\nWe study an important problem."
    assert out[1] == "Method\n\nWe propose a novel approach."


def test_numbered_heading_matched() -> None:
    blocks = ["1. Introduction\n\nContext. Motivation.\n\n2. Results\n\nWe find X."]
    out = chunk(blocks, max_chunk_size=2000)

    assert len(out) == 2
    assert out[0].startswith("1. Introduction")
    assert out[1].startswith("2. Results")


def test_punctuation_fallback_no_hard_cut() -> None:
    # One giant paragraph with no double newline, longer than max_chunk_size,
    # but containing a sentence terminator. Must cut at '. ', never mid-word.
    long_sent = "word " * 400  # ~2400 chars, no terminator long stretch
    text = long_sent + ". trailing tail that ends the block properly. "
    assert len(text) > 300

    out = chunk([text], max_chunk_size=300)

    # No chunk ends mid-word (on a non-space/terminator char while more text remains)
    for c in out:
        assert len(c) > 0
    # The terminator position must be respected: the break happens at ". "
    joined = "".join(out)
    assert joined == text, "chunking must be lossless"
    # Boundary: at least one cut happened
    assert len(out) >= 2
    # Each cut point aligns with a terminator + space
    for piece in out[:-1]:
        assert piece.endswith(". ") or piece.endswith("？") or piece.endswith("；")


def test_no_hard_cut_when_no_terminator() -> None:
    # No sentence terminator at all -> single chunk returned verbatim, no cut.
    text = "abcdefghij" * 100  # 1000 chars, no punctuation
    out = chunk([text], max_chunk_size=300)
    assert out == [text]


def test_empty_and_whitespace_blocks_dropped() -> None:
    assert chunk([], max_chunk_size=2000) == []
    assert chunk(["   \n\n  "], max_chunk_size=2000) == []


def test_allcaps_heading_matched() -> None:
    blocks = ["ABSTRACT\n\nWe summarize. \n\nCONCLUSION\n\nWe conclude."]
    out = chunk(blocks, max_chunk_size=2000)
    assert len(out) == 2
    assert "ABSTRACT" in out[0]
    assert "CONCLUSION" in out[1]


def test_short_text_passthrough() -> None:
    out = chunk(["Short text."], max_chunk_size=2000)
    assert out == ["Short text."]
