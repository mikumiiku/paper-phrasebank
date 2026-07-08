"""Search orchestration (Phase F): encode -> vector search -> rich display."""
from __future__ import annotations

import logging

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from phrasebank.vector import schema as _s
from phrasebank.vector import store
from phrasebank.vector.embed import encode

log = logging.getLogger(__name__)


def _render_result(candidate: dict, index: int, console: Console) -> None:
    meta = candidate["metadata"]
    sentence = meta.get(_s.SENTENCE, "")
    func = meta.get(_s.FUNCTION_CATEGORY, "—")
    tags = _s.tags_from_str(meta.get(_s.TAGS))
    tags_str = ", ".join(tags) if tags else "—"
    title = meta.get(_s.PAPER_TITLE, "")
    authors = meta.get(_s.PAPER_AUTHORS, "")
    year = meta.get(_s.PAPER_YEAR, "")
    source = " — ".join(part for part in (authors, title, year) if part) or "—"
    score = candidate.get("score", 0.0)

    body = Text()
    body.append(f"{sentence}\n\n", style="bold")
    body.append("分类: ", style="dim")
    body.append(f"{func}\n")
    body.append("标签: ", style="dim")
    body.append(f"{tags_str}\n")
    body.append("来源: ", style="dim")
    body.append(f"{source}\n")
    body.append(f"score: ", style="dim")
    body.append(f"{score:.3f}", style="green")

    console.print(
        Panel(
            body,
            title=f"[cyan]#{index + 1}[/cyan]",
            title_align="left",
            border_style="blue",
        )
    )


def run_search(query: str, top_k: int = 10) -> None:
    """Encode ``query``, search the local vector store, and print results.

    First-fault surfaces: ``ModelMismatchError`` is reported explicitly; an
    empty store prints a friendly hint pointing at the extract + review flow;
    a non-empty store with no matches prints a softer notice (never raised).
    """
    if not query or not query.strip():
        Console().print("[yellow]查询为空, 请输入检索内容。[/yellow]")
        return

    console = Console()

    try:
        # Empty store check uses the same validated collection the query uses,
        # so the model-consistency check runs first (first-fault).
        col = store.get_collection(create=True)
        if col.count() == 0:
            console.print(
                "[yellow]向量库为空, 请先用 `ppb extract` + `ppb review` "
                "入库后再检索。[/yellow]"
            )
            return

        vector = encode([query.strip()])[0]
        candidates = store.query(vector, top_k=top_k)
    except store.ModelMismatchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc

    if not candidates:
        console.print("[yellow]没有匹配的句子, 请换个描述再试。[/yellow]")
        return

    console.print(f"[dim]找到 {len(candidates)} 条匹配 (top-k={top_k}):[/dim]\n")
    for i, cand in enumerate(candidates):
        _render_result(cand, i, console)
