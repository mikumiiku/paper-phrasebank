"""Tests for search.run_search orchestration and rich rendering.

``run_search`` imports ``encode`` into its own module namespace, so we patch it
at ``phrasebank.search.encode`` (where used), and ``store.query`` /
``store.get_collection`` on the ``store`` module (accessed as attributes).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from phrasebank import search


# ── helpers ──────────────────────────────────────────────────────────────────

def _candidate(sentence="We propose a novel method.", score=0.9, dist=0.1):
    return {
        "id": "id-1",
        "distance": dist,
        "score": score,
        "metadata": {
            "sentence": sentence,
            "language": "en",
            "paper_title": "A Great Paper",
            "paper_authors": "Smith et al.",
            "paper_year": "2024",
            "source_file_hash": "h",
            "function_category": "方法论述句",
            "tags": "attention,transformer",
            "usage_note": "mechanism intro",
            "reviewed": True,
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    }


def _fake_col(count: int) -> MagicMock:
    col = MagicMock()
    col.count.return_value = count
    return col


# ── run_search ───────────────────────────────────────────────────────────────

class TestRunSearch:
    def test_renders_sentence_and_source(self, capsys):
        col = _fake_col(1)
        q_patch = patch("phrasebank.vector.store.query", return_value=[_candidate()])
        with patch("phrasebank.search.encode", return_value=[[0.1, 0.2, 0.3]]), \
                patch("phrasebank.vector.store.get_collection", return_value=col), \
                q_patch:
            search.run_search("novel attention mechanism", top_k=5)

        out = capsys.readouterr().out
        assert "We propose a novel method." in out
        assert "方法论述句" in out          # function category
        assert "attention" in out          # tag
        assert "A Great Paper" in out      # source paper
        assert "score: 0.9" in out

    def test_encode_then_query_order(self, capsys):
        col = _fake_col(1)
        order: list[str] = []

        def fake_encode(q):  # noqa: ANN001
            order.append("encode")
            return [[0.1, 0.2, 0.3]]

        def fake_query(emb, top_k):  # noqa: ANN001
            order.append("query")
            return [_candidate()]

        with patch("phrasebank.search.encode", side_effect=fake_encode), \
                patch("phrasebank.vector.store.get_collection", return_value=col), \
                patch("phrasebank.vector.store.query", side_effect=fake_query):
            search.run_search("some query", top_k=7)
        assert order == ["encode", "query"]

    def test_top_k_forwarded_to_query(self):
        col = _fake_col(1)
        with patch("phrasebank.search.encode", return_value=[[0.1]]), \
                patch("phrasebank.vector.store.get_collection", return_value=col), \
                patch("phrasebank.vector.store.query", return_value=[_candidate()]) \
                as q_mock:
            search.run_search("q", top_k=7)
        _args, kwargs = q_mock.call_args
        assert kwargs["top_k"] == 7

    def test_empty_store_prints_hint_no_query(self, capsys):
        col = _fake_col(0)
        with patch("phrasebank.search.encode") as enc_mock, \
                patch("phrasebank.vector.store.get_collection", return_value=col), \
                patch("phrasebank.vector.store.query") as q_mock:
            search.run_search("q")
        out = capsys.readouterr().out
        assert "向量库为空" in out
        assert "ppb extract" in out
        q_mock.assert_not_called()
        enc_mock.assert_not_called()

    def test_no_results_prints_notice(self, capsys):
        col = _fake_col(1)
        with patch("phrasebank.search.encode", return_value=[[0.1]]), \
                patch("phrasebank.vector.store.get_collection", return_value=col), \
                patch("phrasebank.vector.store.query", return_value=[]):
            search.run_search("obscure query")
        out = capsys.readouterr().out
        assert "没有匹配的句子" in out

    def test_mismatch_error_becomes_systemexit(self, capsys):
        from phrasebank.vector.store import ModelMismatchError

        col = MagicMock()
        col.count.side_effect = ModelMismatchError("boom mismatch")
        with patch("phrasebank.vector.store.get_collection", return_value=col):
            with pytest.raises(SystemExit) as exc:
                search.run_search("q")
            assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "boom mismatch" in out

    def test_empty_query_prints_hint(self, capsys):
        with patch("phrasebank.vector.store.get_collection") as gc:
            search.run_search("   ")
        out = capsys.readouterr().out
        assert "查询为空" in out
        gc.assert_not_called()
