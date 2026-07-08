"""End-to-end orchestration for ``ppb extract``.

Thin glue layer — each step lives in its own module:

    parse → clean → chunk → LLM (metadata + sentences) → review queue

This module only decides *which* step runs next and how failures are
reported (first-fault at the step level; per-chunk sentence failures are
recorded by ``llm.extract`` so the caller can retry them later).
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from phrasebank import config
from phrasebank.chunking import chunk as chunk_text
from phrasebank.llm import get_client
from phrasebank.llm.extract import extract_sentences, write_failures
from phrasebank.llm.metadata import Metadata, extract_metadata
from phrasebank.llm.client import LLMClient
from phrasebank.ocr import get_backend
from phrasebank.parsing import extract_text
from phrasebank.parsing.clean import page_blocks
from phrasebank.review.queue import (
    enqueue,
    file_hash,
    load_queue as load_review_queue,
    queue_exists,
    summary,
)

console = Console()


def _build_llm_client(settings: config.Settings) -> LLMClient:
    import openai

    client = openai.OpenAI(
        base_url=settings.base_url or None,
        api_key=settings.api_key,
    )
    return LLMClient(client, model_name=settings.model_name)


def _build_ocr_fn(settings: config.Settings):
    """Return an ``ocr_fn`` if a backend is configured, else ``None``.

    Callers must check ``settings.ocr_backend`` — an unconfigured backend means
    image-only pages are skipped with a warning (per requirements §3.2/§3.1.3),
    not an error.
    """
    if not settings.ocr_backend:
        return None
    be = get_backend(
        settings.ocr_backend,
        api_key=settings.ocr_api_key,
        base_url=settings.ocr_base_url or None,
    )

    def _ocr(page_num: int, image_bytes: bytes) -> str:
        return be.recognize(page_num, image_bytes)

    return _ocr


def run_extract(
    file: str,
    settings: config.Settings,
    *,
    force: bool = False,
) -> int:
    """Run the indexing pipeline. Returns the number of candidate sentences
    written to the review queue."""
    path = Path(file)
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件: {path}")

    fh = file_hash(path)
    if not force and queue_exists(fh):
        console.print(
            f"[yellow]此论文已处理过 ({fh[:8]}…)。[/yellow] "
            "[dim]加 --force 重新抽取，或运行 `ppb review` 继续审核之前的候选。[/dim]"
        )
        existing = load_review_queue(fh)
        s = summary(existing)
        console.print(
            f"[dim]队列中现有：待审核 {s['pending']}，已保留 {s['keep']}，已丢弃 {s['drop']}[/dim]"
        )
        return 0

    ocr_fn = _build_ocr_fn(settings)
    if path.suffix.lower() == ".pdf" and ocr_fn is None:
        console.print(
            "[yellow]注意：PDF 可能含图片页但未配置 OCR，这些页将被跳过。"
            "[/yellow]"
        )

    # 1. Parse
    console.print(f"[dim]正在解析：{path.name}[/dim]")
    pages = extract_text(path, ocr_fn=ocr_fn)
    if not pages:
        console.print("[red]解析未产出任何文本，终止。[/red]")
        return 0

    image_pages = sum(1 for p in pages if p.is_image_page and not p.text.strip())
    if image_pages:
        console.print(f"[dim]跳过 {image_pages} 个无法提取的图片页（未配置/OCR 失败）。[/dim]")

    # 2. Clean
    blocks = page_blocks(pages)
    if not blocks:
        console.print("[red]清洗后无文本块，终止。[/red]")
        return 0

    # 3. Chunk
    chunks = chunk_text(blocks)
    if not chunks:
        console.print("[red]分块失败，终止。[/red]")
        return 0
    console.print(f"[dim]文本分块：{len(chunks)} 块[/dim]")

    # 4. Extract metadata once from the first page's text
    client = _build_llm_client(settings)
    first_page_text = next((p.text for p in pages if p.text.strip()), "")
    md: Metadata = Metadata()
    if first_page_text.strip():
        try:
            md = extract_metadata(client, first_page_text)
        except Exception as exc:  # metadata failure is non-fatal
            console.print(f"[yellow]元数据抽取失败（跳过）：{exc}[/yellow]")
    if md.paper_title:
        console.print(f"[dim]论文：{md.paper_title} ({md.paper_year or '—'})[/dim]")

    # 5. Extract sentences per chunk
    console.print(f"[dim]正在逐块抽取候选句…[/dim]")
    candidates, failed = extract_sentences(client, chunks, md)
    if failed:
        write_failures(fh, failed)
        console.print(
            f"[yellow]警告：{len(failed)} 个分块抽取失败（已记入 "
            f"{fh[:8]}…_failures.json，可后续重试）。[/yellow]"
        )

    if not candidates:
        console.print("[yellow]未抽到候选句。[/yellow]")
        return 0

    # 6. Enqueue for review
    state = enqueue(fh, str(path), [c.__dict__ for c in candidates], force=force)
    s = summary(state)
    console.print(
        f"[green]✓ 已写入待审核队列：新增 {len(candidates)} 条，"
        f"共待审核 {s['pending']} 条。[/green]"
    )
    console.print("[dim]下一步：运行 `ppb review` 开始审核。[/dim]")
    return len(candidates)
