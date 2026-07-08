"""Chroma collection access: model-consistency check + add / query."""
from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from phrasebank.vector import embed, get_client, get_collection_name
from phrasebank.vector import schema as _s

log = logging.getLogger(__name__)


class ModelMismatchError(Exception):
    """Raised when the on-disk collection was built with a different embedding
    model (or library version) than the one currently configured. Signals the
    user to rebuild or unify the model before searching."""


def _build_model_identifier() -> str:
    return _s.model_identifier(embed.MODEL_NAME)


def get_collection(create: bool = True) -> Collection:
    """Load the sentence collection, validating it was built with the current
    embedding model.

    On first creation the model metadata (``model_name`` + ``model_identifier``)
    is stamped onto the collection. On subsequent loads it is compared to the
    current values; any mismatch raises :class:`ModelMismatchError`
    (first-fault — never swallowed).
    """
    client = get_client()
    name = get_collection_name()
    current_ident = _build_model_identifier()

    if create:
        try:
            col = client.get_or_create_collection(name=name)
        except chromadb.errors.InvalidCollectionException:
            # Race between get and create; fall back to a plain get.
            col = client.get_collection(name=name)
    else:
        col = client.get_collection(name=name)

    meta = col.metadata or {}
    stored_name = meta.get("model_name")
    stored_ident = meta.get("model_identifier")

    if stored_name is None:
        # Fresh collection — stamp model metadata now.
        col.modify(metadata={**meta, "model_name": embed.MODEL_NAME,
                              "model_identifier": current_ident})
        return col

    if stored_name != embed.MODEL_NAME or stored_ident != current_ident:
        raise ModelMismatchError(
            f"向量库模型不一致: 库内使用 {stored_name}:{stored_ident}, "
            f"当前配置 {embed.MODEL_NAME}:{current_ident}。"
            f"请运行 `ppb extract --force` 重建, 或统一到同一模型后再检索。"
        )
    return col


def _encode_or_empty(records: list[dict]) -> list[list[float]]:
    texts = [r.get(_s.SENTENCE, "") for r in records]
    if not texts:
        return []
    return embed.encode(texts)


def add_sentences(records: list[dict]) -> list[str]:
    """Encode + upsert reviewed sentence records into the collection.

    ``records`` are the structured dicts produced by the review stage (the same
    fields consumed by :func:`schema.to_metadata`). Returns the list of ids
    written. Signature is a list of ids for symmetry with Chroma and to keep
    if ``records`` is empty it is a no-op (returns ``[]``).
    """
    if not records:
        return []

    col = get_collection(create=True)
    ids: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for rec in records:
        ids.append(rec.get("id") or _s.make_id())
        metadatas.append(_s.to_metadata(**rec))

    embeddings = _encode_or_empty(records)
    col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    log.info("upserted %d sentence(s) into %s", len(ids), get_collection_name())
    return ids


def query(
    embedding: list[float],
    top_k: int,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Vector search. Returns a list of candidate dicts, each carrying ``id``,
    ``distance``, ``metadata`` (raw Chroma metadata with tags as the stored
    comma-string), and ``score`` (1 - distance, larger is closer).

    Returns an empty list when the collection is empty or no results match
    the ``where`` filter — the caller decides how to report that.
    """
    col = get_collection(create=True)
    if col.count() == 0:
        return []

    kwargs: dict[str, Any] = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["metadatas", "distances"],
    }
    where = where or {}
    if _s.REVIEWED in where:
        # Chroma stores bools, but normalize defensively.
        where[_s.REVIEWED] = bool(where[_s.REVIEWED])
    if where:
        kwargs["where"] = where

    res = col.query(**kwargs)
    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]

    candidates: list[dict[str, Any]] = []
    for sid, dist, meta in zip(ids, dists, metas):
        candidates.append({
            "id": sid,
            "distance": float(dist),
            "score": 1.0 - float(dist),
            "metadata": meta,
        })
    return candidates
