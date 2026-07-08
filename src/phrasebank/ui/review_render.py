"""Rich rendering of a single candidate during review."""
from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from phrasebank import FUNCTION_CATEGORIES


def render_candidate(entry: dict, index: int, total: int) -> Panel:
    cat = entry.get("function_category") or "（未分类）"
    tags = ", ".join(entry.get("tags") or []) or "—"
    meta = f"来源：{entry.get('paper_title') or '—'}"
    if entry.get("paper_year"):
        meta += f"  ·  {entry['paper_year']}"
    title = f"[bold white]#{index + 1}[/bold white]  [dim]({index + 1}/{total})[/dim]  [cyan]{cat}[/cyan]"

    body = Text()
    body.append(f"{entry.get('sentence', '').strip()}\n\n", style="bold")
    body.append(f"[dim]语言：{'英文' if entry.get('language') == 'en' else '中文'}[/dim]\n")
    body.append(f"标签：{tags}\n")
    body.append(f"用途：{entry.get('usage_note') or '—'}\n")
    body.append(meta, style="dim")

    return Panel(body, title=title, border_style="blue")
