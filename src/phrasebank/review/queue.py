"""Review queue: per-paper JSON-backed candidate sentences.

Single-process, sequential-access; no extra storage engine. Each source file
gets one queue file under ``<data_dir>/review_queue/<file_hash>.json``. The
review loop marks each entry ``status: reviewed`` in-place (atomic rewrite)
so ``Ctrl+C`` at any point resumes cleanly.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phrasebank.config import data_dir

QUEUE_ROOT = "review_queue"
queue_lock = threading.Lock()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def queue_path(file_hash_: str) -> Path:
    base = data_dir() / QUEUE_ROOT
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{file_hash_}.json"


def queue_exists(file_hash_: str) -> bool:
    return queue_path(file_hash_).exists()


# ── Read / Write ────────────────────────────────────────────────────────────

def load_queue(file_hash_: str) -> dict[str, Any]:
    p = queue_path(file_hash_)
    if not p.exists():
        return {"file_hash": file_hash_, "entries": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OSError(f"读取审核队列失败：{p}\n{exc}") from exc


def save_queue(state: dict[str, Any]) -> None:
    """Atomic-ish write via temp+rename so a crashed review leaves the
    previous state intact."""
    p = queue_path(state["file_hash"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(p)


# ── Mutations ───────────────────────────────────────────────────────────────

def reset_status_reviewed(state: dict[str, Any]) -> dict[str, Any]:
    """Use when re-extracting (--force): bump any 'reviewed' back to pending
    so the user sees every sentence again."""
    for e in state.get("entries", []):
        if e.get("status") == "reviewed" and e.get("decision") == "keep":
            e["status"] = "pending"
            e.pop("decision", None)
    return state


def enqueue(
    file_hash_: str,
    source_path: str,
    entries: list[dict[str, Any]],
    force: bool = False,
) -> dict[str, Any]:
    """Append candidates to the paper's queue. With ``force``, reset previously
    reviewed entries to pending; otherwise merge keyed by sentence text
    (idempotent re-run)."""
    with queue_lock:
        state = load_queue(file_hash_) if queue_exists(file_hash_) else {
            "file_hash": file_hash_,
            "source_path": source_path,
            "entries": [],
        }
        if force:
            state = reset_status_reviewed(state)

        seen = {e["sentence"] for e in state["entries"]}
        for cand in entries:
            if cand.get("sentence") in seen:
                continue
            state["entries"].append(
                {
                    "sentence": cand["sentence"],
                    "language": cand.get("language", "en"),
                    "function_category": cand.get("function_category", ""),
                    "tags": list(cand.get("tags") or []),
                    "usage_note": cand.get("usage_note", ""),
                    "paper_title": cand.get("paper_title", ""),
                    "paper_authors": cand.get("paper_authors", ""),
                    "paper_year": cand.get("paper_year", ""),
                    "status": "pending",
                }
            )
            seen.add(cand["sentence"])
        state["updated_at"] = now_iso()
        save_queue(state)
        return state


def first_pending(state: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    for i, e in enumerate(state.get("entries", [])):
        if e.get("status") == "pending":
            return i, e
    return None


def mark_reviewed(
    file_hash_: str,
    sentence: str,
    decision: str,           # "keep" | "drop"
    edits: dict | None = None,
) -> None:
    """Mark one sentence reviewed. ``edits`` may carry updated
    ``function_category`` / ``tags`` (user edits during review)."""
    with queue_lock:
        state = load_queue(file_hash_)
        for e in state["entries"]:
            if e["sentence"] == sentence and e["status"] == "pending":
                e["status"] = "reviewed"
                e["decision"] = decision
                e["reviewed_at"] = now_iso()
                if decision == "keep" and edits:
                    if edits.get("function_category"):
                        e["function_category"] = edits["function_category"]
                    if edits.get("tags") is not None:
                        e["tags"] = edits["tags"]
                save_queue(state)
                return


def mark_all(file_hash_: str, decision: str) -> int:
    """Bulk decision for all remaining pending entries. Returns count affected."""
    with queue_lock:
        state = load_queue(file_hash_)
        n = 0
        for e in state["entries"]:
            if e.get("status") == "pending":
                e["status"] = "reviewed"
                e["decision"] = decision
                e["reviewed_at"] = now_iso()
                n += 1
        save_queue(state)
        return n


def keep_entries(state: dict[str, Any]) -> list[dict[str, Any]]:
    """All entries the user kept and reviewed — ready to send to the vector store."""
    return [e for e in state.get("entries", [])
            if e.get("status") == "reviewed" and e.get("decision") == "keep"]


def summary(state: dict[str, Any]) -> dict[str, int]:
    c = {"pending": 0, "keep": 0, "drop": 0}
    for e in state.get("entries", []):
        s = e.get("status")
        if s == "pending":
            c["pending"] += 1
        elif s == "reviewed":
            c[e.get("decision", "drop")] += 1
    return c


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
