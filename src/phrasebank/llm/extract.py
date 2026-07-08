"""Per-chunk sentence extraction + failure tracking."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from phrasebank import config
from .client import LLMClient, LLMError
from .metadata import Metadata
from .prompts import SENTENCE_SYSTEM, SENTENCE_USER

# Heuristic: infer primary language of a chunk so the prompt can prime the model.
# Purely a hint; each sentence decides on its own language.
_ZH_CHAR_RANGE = (0x4E00, 0x9FFF)


def _infer_language(text: str) -> str:
    sample = text[:200]
    total = max(len(sample), 1)
    zh = sum(1 for ch in sample if _ZH_CHAR_RANGE[0] <= ord(ch) <= _ZH_CHAR_RANGE[1])
    return "zh" if zh / total > 0.2 else "en"


@dataclass
class CandidateSentence:
    sentence: str
    language: str
    function_category: str
    tags: list[str] = field(default_factory=list)
    usage_note: str = ""
    paper_title: str = ""
    paper_authors: str = ""
    paper_year: str = ""


class ExtractionFailure(Exception):
    """Raised only when explicitly requested (see ``fail_hard``)."""


def _build_sentence(item: Any, md: Metadata) -> CandidateSentence:
    if not isinstance(item, dict):
        raise LLMError(f"Expected dict per sentence, got {type(item).__name__}")
    tags_raw = item.get("tags") or []
    # Defensive: some providers serialise ``tags`` as a single comma string
    # instead of a JSON array ("A-MEM, agentic" -> ["A-MEM, agentic"]).
    if isinstance(tags_raw, str):
        tags_raw = [p.strip() for p in tags_raw.split(",") if p.strip()]
    tags = [str(t).strip() for t in tags_raw if t and len(str(t)) < 64]
    return CandidateSentence(
        sentence=_safe_str(item.get("sentence")),
        language=_safe_str(item.get("language")) or "en",
        function_category=_safe_str(item.get("function_category")),
        tags=tags,
        usage_note=_safe_str(item.get("usage_note")),
        paper_title=md.paper_title,
        paper_authors=md.paper_authors,
        paper_year=md.paper_year,
    )


def extract_sentences(
    client: LLMClient,
    chunks: list[str],
    metadata: Metadata,
    *,
    fail_hard: bool = False,
) -> tuple[list[CandidateSentence], list[int]]:
    """Extract candidate sentences from each chunk in order.

    Returns ``(candidates, failed_indices)``.  By default, a single chunk
    failure is recorded in ``failed_indices`` rather than aborting the batch
    ("first-fault" applies per-chunk, not per-batch).  Set ``fail_hard=True``
    to raise immediately on the first failing chunk.
    """
    candidates: list[CandidateSentence] = []
    failed: list[int] = []

    md = metadata
    for idx, chunk in enumerate(chunks):
        language = _infer_language(chunk)
        try:
            items = client.call_json(
                SENTENCE_SYSTEM,
                SENTENCE_USER.format(language=language, chunk=chunk),
            )
        except (LLMError, Exception) as exc:
            if fail_hard:
                raise
            failed.append(idx)
            continue

        if not items:
            continue
        for item in items:
            try:
                candidates.append(_build_sentence(item, md))
            except LLMError:
                if fail_hard:
                    raise

    return candidates, failed


def retry_failed(
    client: LLMClient,
    file_hash: str,
    chunks: list[str],
    metadata: Metadata,
    *,
    fail_hard: bool = False,
) -> tuple[list[CandidateSentence], list[int]]:
    """Re-run extraction only for the chunk indices recorded in a failures file.

    The on-disk path aligns with the review queue (``review_queue``).
    Missing file => nothing to retry => empty result, no error.
    """
    failures_path = _failures_path(file_hash)
    if not failures_path.exists():
        return [], []

    try:
        indices = json.loads(failures_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return [], []

    sub_chunks = [chunks[i] for i in indices if 0 <= i < len(chunks)]
    candidates, inner_failed = extract_sentences(
        client, sub_chunks, metadata, fail_hard=fail_hard
    )
    # Remap inner indices back to original chunk positions.
    failed_original = [indices[i] for i in inner_failed]
    return candidates, failed_original


def write_failures(file_hash: str, failed_indices: list[int]) -> Path:
    """Persist failed chunk indices to the review-queue sidecar JSON."""
    path = _failures_path(file_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(failed_indices, ensure_ascii=False), encoding="utf-8")
    return path


def _failures_path(file_hash: str) -> Path:
    return config.data_dir() / "review_queue" / f"{file_hash}_failures.json"


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()
