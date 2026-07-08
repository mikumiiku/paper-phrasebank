"""Review queue: enqueue, mark, resume, mark-all."""
from __future__ import annotations

import pytest

from phrasebank.config import data_dir
from phrasebank.review import queue


def _sample_entry(sentence="A baseline shows X.", category="方法论述句"):
    return {
        "sentence": sentence,
        "language": "en",
        "function_category": category,
        "tags": ["baseline"],
        "usage_note": "用于方法比较",
        "paper_title": "Paper",
        "paper_authors": "A",
        "paper_year": "2024",
    }


def test_enqueue_and_summary(isolated_config):
    fh = "abc123"
    state = queue.enqueue(fh, "p.pdf", [_sample_entry()], force=False)
    assert queue.summary(state) == {"pending": 1, "keep": 0, "drop": 0}
    assert queue.queue_exists(fh)


def test_resume_after_partial_review(isolated_config, monkeypatch):
    """Simulate Ctrl+C after keeping 1, dropping 1 of 3: resume hits #3."""
    fh = "resume_hash"
    entries = [
        _sample_entry("s1"),
        _sample_entry("s2"),
        _sample_entry("s3"),
    ]
    queue.enqueue(fh, "p.pdf", entries, force=False)
    # mark #1 keep, #2 drop (don't touch #3)
    queue.mark_reviewed(fh, "s1", "keep")
    queue.mark_reviewed(fh, "s2", "drop")
    state = queue.load_queue(fh)
    hit = queue.first_pending(state)
    assert hit is not None
    idx, entry = hit
    assert entry["sentence"] == "s3"


def test_mark_all_keep(isolated_config):
    fh = "keep_all_hash"
    queue.enqueue(fh, "p.pdf", [_sample_entry(f"s{i}") for i in range(3)], force=False)
    n = queue.mark_all(fh, "keep")
    assert n == 3
    assert queue.summary(queue.load_queue(fh)) == {"pending": 0, "keep": 3, "drop": 0}


def test_force_resets_reviewed_to_pending(isolated_config):
    fh = "force_hash"
    queue.enqueue(fh, "p.pdf", [_sample_entry()], force=False)
    queue.mark_reviewed(fh, "A baseline shows X.", "keep")
    assert queue.summary(queue.load_queue(fh))["pending"] == 0
    # force re-extract
    queue.enqueue(fh, "p.pdf", [_sample_entry()], force=True)
    s = queue.load_queue(fh)
    assert queue.summary(s)["pending"] == 1


def test_keep_entries_filters(isolated_config):
    fh = "filter_hash"
    queue.enqueue(
        fh,
        "p.pdf",
        [_sample_entry("keep1"), _sample_entry("drop1", category="研究空白陈述句")],
        force=False,
    )
    queue.mark_reviewed(fh, "keep1", "keep")
    queue.mark_reviewed(fh, "drop1", "drop")
    kept = queue.keep_entries(queue.load_queue(fh))
    assert [e["sentence"] for e in kept] == ["keep1"]


def test_dedupe_by_sentence(isolated_config):
    fh = "dedupe_hash"
    queue.enqueue(fh, "p.pdf", [_sample_entry("dup")], force=False)
    queue.enqueue(fh, "p.pdf", [_sample_entry("dup")], force=False)
    assert len(queue.load_queue(fh)["entries"]) == 1
