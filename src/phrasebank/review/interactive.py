"""Interactive review loop (``ppb review``).

Walks the user through every pending candidate across all queued papers.
Each action rewrites the queue file immediately, so ``Ctrl+C`` at any point
leaves a consistent state and the next run resumes from the same spot.
"""
from __future__ import annotations

from pathlib import Path

import questionary
import typer
from rich.console import Console

from phrasebank import FUNCTION_CATEGORIES
from phrasebank.config import data_dir
from phrasebank.review import queue
from phrasebank.ui.review_render import render_candidate

console = Console()
QUEUE_DIR = "review_queue"


def _queued_papers() -> list[Path]:
    root = data_dir() / QUEUE_DIR
    if not root.exists():
        return []
    return sorted(root.glob("*.json"))


def _edit_flow(entry: dict) -> dict:
    """Allow the user to tweak function_category and tags only (the original
    sentence is immutable — if it's wrong, drop it)."""
    new_cat = questionary.select(
        "修改功能分类：",
        choices=FUNCTION_CATEGORIES,
        default=entry.get("function_category") or FUNCTION_CATEGORIES[0],
    ).ask()
    if new_cat is None:
        return {}
    cur_tags = ", ".join(entry.get("tags") or [])
    tags_text = questionary.text(
        "修改标签（逗号分隔，回车确认）：",
        default=cur_tags,
    ).ask()
    if tags_text is None:
        return {}
    tags = [t.strip() for t in tags_text.split(",") if t.strip()]
    return {"function_category": new_cat, "tags": tags}


def _review_paper(state: dict) -> int:
    """Review one paper's queue. Returns the number of entries kept."""
    fh = state["file_hash"]
    total = len(state["entries"])
    kept = 0
    while True:
        hit = queue.first_pending(state)
        if hit is None:
            break
        idx, entry = hit
        console.print(render_candidate(entry, idx, total))
        action = questionary.select(
            "操作：",
            choices=[
                questionary.Choice("保留", "keep"),
                questionary.Choice("丢弃", "drop"),
                questionary.Choice("编辑（仅分类/标签）", "edit"),
                questionary.Choice("剩余全部保留", "keep_all"),
                questionary.Choice("剩余全部丢弃", "drop_all"),
                questionary.Choice("退出（稍后继续）", "quit"),
            ],
        ).ask()
        if action is None or action == "quit":
            return kept
        if action == "keep":
            queue.mark_reviewed(fh, entry["sentence"], "keep")
            kept += 1
        elif action == "drop":
            queue.mark_reviewed(fh, entry["sentence"], "drop")
        elif action == "edit":
            edits = _edit_flow(entry)
            if edits:
                queue.mark_reviewed(fh, entry["sentence"], "keep", edits)
                kept += 1
            else:
                continue
        elif action == "keep_all":
            n = queue.mark_all(fh, "keep")
            kept += n
            return kept
        elif action == "drop_all":
            queue.mark_all(fh, "drop")
            return kept
        # refresh state for next iteration
        state = queue.load_queue(fh)
    return kept


def _entry_to_vector_rec(e: dict, file_hash_: str) -> dict:
    """Map a review-queue entry's free fields into ``to_metadata`` kwargs."""
    from phrasebank.vector.schema import to_metadata
    return to_metadata(
        sentence=e["sentence"],
        language=e.get("language") or "en",
        paper_title=e.get("paper_title") or "",
        paper_authors=e.get("paper_authors") or "",
        paper_year=e.get("paper_year") or "",
        source_file_hash=file_hash_,
        function_category=e.get("function_category") or "",
        tags=e.get("tags") or [],
        usage_note=e.get("usage_note") or "",
        reviewed=True,
    )


def _flush_kept(state: dict) -> int:
    """Push all kept entries to the vector store. Returns count added."""
    kept = queue.keep_entries(state)
    if not kept:
        return 0
    from phrasebank.vector.store import add_sentences
    file_hash_ = state.get("file_hash", "")
    records = [_entry_to_vector_rec(e, file_hash_) for e in kept]
    add_sentences(records)
    return len(records)


def run_review() -> None:
    papers = _queued_papers()
    if not papers:
        console.print(
            "[yellow]没有待审核的候选句。请先运行 `ppb extract <file>` 抽取。[/yellow]"
        )
        return

    total_kept = 0
    for p in papers:
        state = queue.load_queue(p.stem)
        pending = queue.summary(state)["pending"]
        if pending == 0:
            continue
        console.print(
            f"\n[bold magenta]审核论文：{state.get('source_path')}[/bold magenta]  "
            f"[dim]（待审核 {pending} 条）[/dim]"
        )
        try:
            kept = _review_paper(state)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]已中断，进度已保存。下次运行 `ppb review` 继续。[/yellow]")
            return
        if kept:
            n = _flush_kept(queue.load_queue(p.stem))
            total_kept += n
            console.print(f"[green]✓ 本批入库 {n} 条[/green]")
    console.print(f"\n[bold green]审核完成，共入库 {total_kept} 条。[/bold green]")
