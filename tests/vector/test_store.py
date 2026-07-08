"""Tests for vector.store: model validation, add, query."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import phrasebank.vector.store as store
from phrasebank.vector import schema as S


# ── fixtures ─────────────────────────────────────────────────────────────────

def _record(**over):
    base = {
        "sentence": "We propose a novel attention mechanism.",
        "language": "en",
        "paper_title": "Attention Is All You Need",
        "paper_authors": "Vaswani et al.",
        "paper_year": "2017",
        "source_file_hash": "abc123",
        "function_category": "方法论述句",
        "tags": ["attention", "transformer"],
        "usage_note": "describing a new mechanism",
        "reviewed": True,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    base.update(over)
    return base


def _make_collection(meta=None, *, count=0, query_result=None):
    """Build a fake chromadb Collection with controllable metadata, count and a
    canned query result (ids/distances/metadatas triple)."""
    col = MagicMock()
    col.metadata = {} if meta is None else dict(meta)
    col.count.return_value = count
    col.modify = MagicMock()

    def _query(**kwargs):  # noqa: ANN001
        if query_result is not None:
            return query_result
        return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    col.query.side_effect = _query
    col.upsert = MagicMock()
    return col


def _mock_get_or_create(collection):
    """Patch store.get_client + relevant class methods so store never touches
    disk and returns our fake collection."""
    client = MagicMock()

    def goc(name=None, **_kw):
        collection._captured_name = name
        return collection

    client.get_or_create_collection.side_effect = goc
    client.get_collection.return_value = collection

    fixed_ident = "sentence-transformers@5.6.0:BAAI/bge-m3"

    return [
        patch("phrasebank.vector.store.get_client", return_value=client),
        patch("phrasebank.vector.store._build_model_identifier", return_value=fixed_ident),
        patch("phrasebank.vector.embed.MODEL_NAME", "BAAI/bge-m3"),
    ]


# ── get_collection ───────────────────────────────────────────────────────────

class TestGetCollection:
    def test_fresh_collection_gets_model_metadata_stamped(self):
        col = _make_collection(meta=None)
        patches = _mock_get_or_create(col)
        with patches[0], patches[1], patches[2]:
            out = store.get_collection(create=True)
        assert out is col
        args, kwargs = col.modify.call_args
        assert kwargs["metadata"]["model_name"] == "BAAI/bge-m3"
        assert kwargs["metadata"]["model_identifier"] == "sentence-transformers@5.6.0:BAAI/bge-m3"
        assert col._captured_name == "phrasebank_sentences"

    def test_matching_model_passes(self):
        col = _make_collection(
            meta={"model_name": "BAAI/bge-m3",
                  "model_identifier": "sentence-transformers@5.6.0:BAAI/bge-m3"}
        )
        patches = _mock_get_or_create(col)
        with patches[0], patches[1], patches[2]:
            assert store.get_collection(create=True) is col

    def test_mismatched_model_name_raises(self):
        # identifier matches but model name differs -> mismatch
        col = _make_collection(
            meta={"model_name": "BAAI/bge-small-en",
                  "model_identifier": "sentence-transformers@5.6.0:BAAI/bge-m3"}
        )
        patches = _mock_get_or_create(col)
        with pytest.raises(store.ModelMismatchError):
            with patches[0], patches[1], patches[2]:
                store.get_collection(create=True)

    def test_mismatched_model_identifier_raises(self):
        # model name matches but identifier (app+st version) differs -> mismatch
        col = _make_collection(
            meta={"model_name": "BAAI/bge-m3",
                  "model_identifier": "sentence-transformers@OLD:BAAI/bge-m3"}
        )
        patches = _mock_get_or_create(col)
        with pytest.raises(store.ModelMismatchError):
            with patches[0], patches[1], patches[2]:
                store.get_collection(create=True)

    def test_no_create_still_validates(self):
        col = _make_collection(
            meta={"model_name": "BAAI/bge-m3",
                  "model_identifier": "sentence-transformers@OLD:BAAI/bge-m3"}
        )
        patches = _mock_get_or_create(col)
        with pytest.raises(store.ModelMismatchError):
            with patches[0], patches[1], patches[2]:
                store.get_collection(create=False)


# ── add_sentences ────────────────────────────────────────────────────────────

def test_add_sentences_encodes_and_upserts_with_metadata():
    col = _make_collection(meta={"model_name": "BAAI/bge-m3",
                                 "model_identifier": "sentence-transformers@5.6.0:BAAI/bge-m3"},
                           count=0)
    recs = [_record(), _record(sentence="Second.", paper_title="Other")]

    def fake_encode(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    patches = _mock_get_or_create(col)
    with patches[0], patches[1], patches[2], patch(
        "phrasebank.vector.store.embed.encode", side_effect=fake_encode
    ):
        ids = store.add_sentences(recs)

    assert len(ids) == 2
    # ids must be non-empty hex strings (uuid4)
    assert all(isinstance(i, str) and i for i in ids)

    args, kwargs = col.upsert.call_args
    assert kwargs["ids"] == ids
    assert len(kwargs["embeddings"]) == 2
    assert len(kwargs["metadatas"]) == 2
    # tags stored as comma string, sentence preserved
    assert kwargs["metadatas"][0]["tags"] == "attention,transformer"
    assert kwargs["metadatas"][0]["sentence"] == recs[0]["sentence"]
    assert kwargs["metadatas"][0]["reviewed"] is True


def test_add_sentences_empty_is_noop():
    col = _make_collection(meta={"model_name": "BAAI/bge-m3",
                                 "model_identifier": "sentence-transformers@5.6.0:BAAI/bge-m3"},
                           count=0)
    patches = _mock_get_or_create(col)
    with patches[0], patches[1], patches[2], patch(
        "phrasebank.vector.store.embed.encode"
    ) as enc:
        ids = store.add_sentences([])
    assert ids == []
    enc.assert_not_called()
    col.upsert.assert_not_called()


# ── query ────────────────────────────────────────────────────────────────────

def test_query_returns_candidates_with_score_and_metadata():
    meta_in_col = {
        "sentence": "We propose a novel attention mechanism.",
        "language": "en",
        "paper_title": "Attention Is All You Need",
        "paper_authors": "Vaswani et al.",
        "paper_year": "2017",
        "source_file_hash": "abc123",
        "function_category": "方法论述句",
        "tags": "attention,transformer",
        "reviewed": True,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    canned = {
        "ids": [["id-1", "id-2"]],
        "distances": [[0.0, 0.5]],
        "metadatas": [[meta_in_col, meta_in_col]],
    }
    col = _make_collection(
        meta={"model_name": "BAAI/bge-m3",
              "model_identifier": "sentence-transformers@5.6.0:BAAI/bge-m3"},
        count=2, query_result=canned,
    )
    patches = _mock_get_or_create(col)
    with patches[0], patches[1], patches[2]:
        cands = store.query([0.1, 0.2, 0.3], top_k=2)

    assert len(cands) == 2
    assert {c["id"] for c in cands} == {"id-1", "id-2"}
    assert cands[0]["distance"] == 0.0 and pytest.approx(cands[0]["score"]) == 1.0
    assert cands[1]["distance"] == 0.5 and pytest.approx(cands[1]["score"]) == 0.5
    # metadata preserved verbatim (tags still comma string)
    assert cands[0]["metadata"]["tags"] == "attention,transformer"

    # n_results and include wired through
    args, kwargs = col.query.call_args
    assert kwargs["n_results"] == 2
    assert kwargs["query_embeddings"] == [[0.1, 0.2, 0.3]]


def test_query_empty_store_returns_empty_list():
    col = _make_collection(
        meta={"model_name": "BAAI/bge-m3",
              "model_identifier": "sentence-transformers@5.6.0:BAAI/bge-m3"},
        count=0,
    )
    patches = _mock_get_or_create(col)
    with patches[0], patches[1], patches[2]:
        cands = store.query([0.1, 0.2, 0.3], top_k=5)
    assert cands == []
    col.query.assert_not_called()


def test_query_passes_where_filter():
    col = _make_collection(
        meta={"model_name": "BAAI/bge-m3",
              "model_identifier": "sentence-transformers@5.6.0:BAAI/bge-m3"},
        count=1, query_result={"ids": [[]], "distances": [[]], "metadatas": [[]]},
    )
    patches = _mock_get_or_create(col)
    with patches[0], patches[1], patches[2]:
        store.query([0.1], top_k=3, where={"reviewed": True})

    args, kwargs = col.query.call_args
    assert kwargs["where"] == {"reviewed": True}
