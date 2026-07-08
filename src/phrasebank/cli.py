"""CLI entrypoint — registers the four v1 subcommands under a single Typer app.

Each subcommand delegates to its dedicated module; this file only wires the
command tree and the global ``--version`` flag.
"""
from __future__ import annotations

import typer
from rich.console import Console

from phrasebank import __version__
from phrasebank.config import (
    ConfigError,
    Settings,
    display_dict,
    exists,
    load,
    require_llm_configured,
    save,
)
from phrasebank.ui.config_menu import edit_loop, run_wizard

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help=(
        "paper-phrasebank — 把论文里的模板句抽到本地向量库，"
        "写论文时用自然语言检索复用。"
    ),
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"ppb {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        help="显示版本号",
    ),
) -> None:
    if ctx.invoked_subcommand is None:
        app()


# ── config ──────────────────────────────────────────────────────────────────

@app.command("config")
def config_cmd(
    show: bool = typer.Option(False, "--show", help="只读展示当前配置"),
    set_key: str | None = typer.Option(None, "--set", help="快速设置单项的 key"),
    value: str | None = typer.Argument(None, help="快速设置单项时的 value"),
) -> None:
    """交互式查看/修改配置（LLM / OCR 密钥等）。"""
    if set_key is not None:
        if value is None:
            console.print("[red]--set 需要同时提供 value[/red]")
            raise typer.Exit(1)
        s = load()
        if not hasattr(s, set_key):
            console.print(f"[red]未知配置项: {set_key}[/red]")
            raise typer.Exit(1)
        setattr(s, set_key, value)
        save(s)
        console.print(f"[green]已设置 {set_key}[/green]")
        return

    if show:
        s = load()
        for k, v in display_dict(s).items():
            console.print(f"[cyan]{k:<16}[/cyan] {v}")
        return

    edit_loop()


# ── extract ─────────────────────────────────────────────────────────────────

@app.command("extract")
def extract_cmd(
    file: str = typer.Argument(..., help="PDF 或 DOCX 文件路径"),
    force: bool = typer.Option(
        False, "--force", help="已处理过也强制重新抽取"
    ),
) -> None:
    """抽取论文候选句，写入待审核队列。"""
    from phrasebank.pipeline import run_extract

    try:
        s = require_llm_configured()
    except ConfigError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        console.print("[yellow]自动进入配置引导…[/yellow]")
        s = run_wizard()

    run_extract(file, settings=s, force=force)


# ── review ──────────────────────────────────────────────────────────────────

@app.command("review")
def review_cmd() -> None:
    """逐条审核候选句（支持 Ctrl+C 断点恢复），通过的写入向量库。"""
    from phrasebank.review.interactive import run_review

    run_review()


# ── search ──────────────────────────────────────────────────────────────────

@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="自然语言描述使用场景"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回条数"),
) -> None:
    """自然语言检索匹配的句子。"""
    from phrasebank.search import run_search

    run_search(query, top_k=top_k)


# ── upgrade ─────────────────────────────────────────────────────────────────

@app.command("upgrade")
@app.command("update")  # alias
def upgrade_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="跳过确认"),
) -> None:
    """检测是否有新版本并自动升级 ppb。"""
    from phrasebank.upgrade import run_upgrade

    rc = run_upgrade(assume_yes=yes)
    raise SystemExit(rc)
