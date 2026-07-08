"""End-to-end v1 pipeline.

Pipeline: PDF → parse → clean → chunk → (mocked) LLM → review queue
         → (auto-approve) review → vector store → (mocked or real) search.

Runs under ``isolated_config`` so it never touches the user's real
~/.config / ~/.local/share. LLM is mocked (no OCR needed here since the
synthetic PDF has text layers). Embedding + Chroma are real — but against
the isolated data dir — so this also exercises the model-consistency check
once on a fresh collection.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from phrasebank import config
from phrasebank.llm.extract import CandidateSentence
from phrasebank.llm.metadata import Metadata
from phrasebank.pipeline import run_extract
from phrasebank.review.queue import (
    enqueue,
    file_hash,
    keep_entries,
    load_queue,
    mark_all,
    queue_exists,
    summary,
)


# ── Mock LLM layer ──────────────────────────────────────────────────────────

def _metadata_response(*args, **kwargs):
    return [Metadata(paper_title="Attention Is All You Need",
                     paper_authors="Vaswani et al.", paper_year="2017")]


def _sentence_response(system, user):
    if "metadata" in system.lower() or "paper_title" in system:
        return [Metadata(paper_title="Attention Is All You Need",
                         paper_authors="Vaswani et al.", paper_year="2017")]
    # Sentence-response path: return a hand-crafted candidate.
    return [
        CandidateSentence(
            sentence="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
            language="en",
            function_category="研究背景引入句",
            tags=["baseline", "RNN"],
            usage_note="用于介绍已有序列建模方法",
            paper_title="Attention Is All You Need",
            paper_authors="Vaswani et al.",
            paper_year="2017",
        )
    ]


class _FakeLLMClient:
    """Mimics ``LLMClient`` just enough for the pipeline + extract."""

    def call_object(self, system, user):
        return _metadata_response()[0].to_dict()

    def call_json(self, system, user):
        items = _sentence_response(system, user)
        # return list[dict] matching real LLMClient semantics
        return [c.__dict__ for c in items]


# ── Helper: build a tiny valid PDF ──────────────────────────────────────────

def _make_three_page_pdf(path: Path) -> Path:
    import fitz
    doc = fitz.open()
    # Page 1: title page (for metadata extraction)
    p = doc.new_page()
    p.insert_text(
        (72, 72),
        "Attention Is All You Need\nVaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin\n2017\n\n"
        "We propose a new simple network architecture, the Transformer.",
    )
    # Page 2: a body paragraph that survives cleaning
    p = doc.new_page()
    p.insert_text(
        (72, 72),
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. "
        "The best performing models also connect the encoder and decoder through an attention mechanism.",
    )
    # Page 3: conclusion
    p = doc.new_page()
    p.insert_text(
        (72, 72),
        "We are excited about the future work of applying attention to many other tasks.",
    )
    doc.save(str(path))
    doc.close()
    return path


# ── Tests ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_vector_client(isolated_config):
    """Each E2E test uses a fresh tmp data dir; point the vector client there."""
    from phrasebank.vector import reset_client
    reset_client()
    yield


@pytest.fixture
def paper_pdf(tmp_dir, isolated_config) -> Path:
    return _make_three_page_pdf(tmp_dir / "paper.pdf")


def test_extract_populates_review_queue(paper_pdf, isolated_config):
    settings = config.Settings(
        provider="openai_compatible",
        base_url="https://fake.local/v1",
        api_key="sk-fake",
        model_name="fake",
    )
    with patch("phrasebank.pipeline._build_llm_client", return_value=_FakeLLMClient()):
        n = run_extract(str(paper_pdf), settings=settings)
    # one candidate sentence from one chunk
    assert n >= 1
    fh = file_hash(paper_pdf)
    assert queue_exists(fh)
    s = summary(load_queue(fh))
    assert s["pending"] >= 1


def test_extract_idempotent_without_force(paper_pdf, isolated_config):
    settings = config.Settings(
        provider="openai_compatible", base_url="https://fake.local/v1",
        api_key="sk-fake", model_name="fake",
    )
    with patch("phrasebank.pipeline._build_llm_client", return_value=_FakeLLMClient()):
        first = run_extract(str(paper_pdf), settings=settings)
        second = run_extract(str(paper_pdf), settings=settings)
    assert first >= 1
    assert second == 0  # short-circuited by existing queue


def test_extract_force_re_runs(paper_pdf, isolated_config):
    settings = config.Settings(
        provider="openai_compatible", base_url="https://fake.local/v1",
        api_key="sk-fake", model_name="fake",
    )
    with patch("phrasebank.pipeline._build_llm_client", return_value=_FakeLLMClient()):
        run_extract(str(paper_pdf), settings=settings)
        # pre-existing reviewed state → force resets back to pending
        from phrasebank.review.queue import load_queue, mark_all, reset_status_reviewed
        fh = file_hash(paper_pdf)
        mark_all(fh, "keep")
        assert summary(load_queue(fh))["pending"] == 0
        third = run_extract(str(paper_pdf), settings=settings, force=True)
    assert third >= 1  # got new candidates


def test_review_flow_vectors_and_searches(paper_pdf, isolated_config):
    """Full path: extract (mocked LLM) → approve-all → vector store → search."""
    settings = config.Settings(
        provider="openai_compatible", base_url="https://fake.local/v1",
        api_key="sk-fake", model_name="fake",
    )
    with patch("phrasebank.pipeline._build_llm_client", return_value=_FakeLLMClient()):
        run_extract(str(paper_pdf), settings=settings)

    fh = file_hash(paper_pdf)
    state = load_queue(fh)
    assert summary(state)["pending"] >= 1

    # Auto-approve everything via the queue API (simulates a fast "keep_all")
    mark_all(fh, "keep")
    kept = keep_entries(load_queue(fh))
    assert len(kept) >= 1

    # Real vector store writes (embedding computes on real BGE-M3 — may be slow)
    from phrasebank.review.interactive import _entry_to_vector_rec, _flush_kept
    n = _flush_kept(load_queue(fh))
    assert n >= 1

    # Search using the same real vector store (capture via Console file=)
    from phrasebank.search import run_search
    from contextlib import redirect_stdout
    import sys
    from io import StringIO
    buf = StringIO()
    # run_search builds its own Console() (without file=), so redirect stdout.
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        run_search("序列模型", top_k=5)
    finally:
        sys.stdout = old_stdout
    out = buf.getvalue()
    assert ("dominant" in out) or ("sequence" in out.lower()) or ("Attention" in out) or ("Transformer" in out), out


def test_search_empty_store_shows_hint(isolated_config, tmp_dir):
    """With no data, search should hint the user rather than raise."""
    from phrasebank.search import run_search
    from io import StringIO
    import sys
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        run_search("anything", top_k=3)
    finally:
        sys.stdout = old_stdout
    out = buf.getvalue()
    assert "向量库为空" in out or "extract" in out.lower() or "empty" in out.lower(), out
