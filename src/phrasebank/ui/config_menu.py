"""Interactive configuration flows (first-run wizard + `ppb config` menu).

All interaction happens in-terminal — users never touch the .toml file
directly. ``run_wizard`` returns the resulting ``Settings`` (or raises
``ConfigError`` when the user aborts, so the caller clearly sees the exit).
"""
from __future__ import annotations

import questionary
import typer

from phrasebank.config import (
    PROVIDER_DEFAULTS,
    Settings,
    display_dict,
    save,
)

CHOICE_BACK = "__back__"
CHOICE_DONE = "__done__"


def _ask_password(message: str, allow_empty: bool = False) -> str:
    while True:
        val = questionary.password(message).ask()
        if val is None:  # Ctrl+C → safe exit
            raise typer.Exit(1)
        val = val.strip()
        if val or allow_empty:
            return val
        questionary.print("  该项不能为空，请重新输入。", style="fg:red")


def _ask_text(message: str, default: str = "", allow_empty: bool = False) -> str:
    while True:
        val = questionary.text(message, default=default).ask()
        if val is None:
            raise typer.Exit(1)
        val = val.strip()
        if val or allow_empty:
            return val
        questionary.print("  该项不能为空，请重新输入。", style="fg:red")


def _step_llm() -> Settings:
    """Step 1: LLM provider → base_url / api_key / model_name."""
    questionary.print("\n[bold]第一步：LLM 提供商[/bold]")
    provider = questionary.select(
        "选择 LLM 提供商：",
        choices=[
            questionary.Choice("DeepSeek（官方 API）", "deepseek"),
            questionary.Choice(
                "OpenAI Compatible（通义千问 / Kimi / Ollama 等）",
                "openai_compatible",
            ),
        ],
    ).ask()
    if provider is None:
        raise typer.Exit(1)

    defaults = PROVIDER_DEFAULTS[provider]
    s = Settings(provider=provider)

    if provider == "deepseek":
        s.base_url = defaults["base_url"]
        s.model_name = defaults["model_name"]
        questionary.print(
            f"  已自动绑定 base_url = {s.base_url}，model = {s.model_name}"
        )
        s.api_key = _ask_password("请输入 DeepSeek API Key：")
    else:
        s.base_url = _ask_text(
            "Base URL（如 https://api.moonshot.cn/v1）：", default=""
        )
        s.api_key = _ask_password(
            "API Key（本地模型无密钥可直接回车跳过）：", allow_empty=True
        )
        model_default = defaults["model_name"]
        s.model_name = _ask_text("Model Name：", default=model_default)
    return s


def _step_ocr(s: Settings) -> Settings:
    """Step 2: OCR backend (optional). The official base_url is filled in
    automatically; the user is only asked for the API key (if required)."""
    from phrasebank.ocr import REGISTRY, configured_base_url

    questionary.print("\n[bold]第二步：OCR 后端（可选）[/bold]")
    backend = questionary.select(
        "选择 OCR 后端（用于处理图片型 PDF 页面）：",
        choices=[
            questionary.Choice("暂不启用 OCR", ""),
            questionary.Choice("MinerU API (官方云端)", "mineru"),
            questionary.Choice("PaddleOCR (推荐本地部署, 默认 http://localhost:8866)", "paddle"),
        ],
    ).ask()
    if backend is None:
        raise typer.Exit(1)
    s.ocr_backend = backend
    if backend:
        needs_key = (backend == "mineru")
        if needs_key:
            s.ocr_api_key = _ask_password("MinerU API Key：", allow_empty=False)
        else:
            questionary.print("  [dim]PaddleOCR 默认无认证, API Key 可跳过。[/dim]")
            s.ocr_api_key = _ask_password(
                "PaddleOCR API Key（本地默认无认证, 回车跳过）：", allow_empty=True
            )
        # Official/local-auto base_url (shown to the user for transparency)
        official = configured_base_url(backend)
        if official:
            questionary.print(f"  官方/默认地址: [cyan]{official}[/dim]（使用默认直接回车）")
            override = _ask_text("Base URL (回车使用默认)：", default=official)
            s.ocr_base_url = override
        else:
            s.ocr_base_url = ""
    return s


def _step_confirm(s: Settings) -> Settings:
    """Step 3: review + write."""
    questionary.print("\n[bold]第三步：确认配置[/bold]")
    rows = display_dict(s)
    for k, v in rows.items():
        questionary.print(f"  [cyan]{k:<16}[/cyan] {v}")
    ok = questionary.confirm("确认写入以上配置？", default=True).ask()
    if ok is None or not ok:
        raise typer.Exit(1)
    return s


def run_wizard() -> Settings:
    """Full first-run flow. Caller writes the returned settings."""
    questionary.print("[bold magenta]欢迎使用 ppb！[/bold magenta]")
    questionary.print("首次运行需要做几步配置（每一步都支持 Ctrl+C 安全退出）\n")
    s = _step_llm()
    s = _step_ocr(s)
    s = _step_confirm(s)
    path = save(s)
    questionary.print(f"\n[green]✓ 配置已写入 {path}[/green]")
    return s


def edit_loop(initial: Settings | None = None) -> None:
    """`ppb config` interactive single-item edit loop."""
    s = initial or __import__("phrasebank.config", fromlist=["load"]).load()
    while True:
        questionary.print("\n[bold]当前配置（选择一项修改，Esc 退出）[/bold]")
        rows = display_dict(s)
        choices = [questionary.Choice(f"{k:<16} = {v}", k) for k, v in rows.items()]
        choices += [questionary.Choice("保存并退出", CHOICE_DONE)]
        picked = questionary.select("修改哪一项？", choices=choices).ask()
        if picked is None or picked in (CHOICE_DONE, ""):
            return
        if picked == "provider":
            provider = questionary.select(
                "LLM 提供商：",
                choices=["deepseek", "openai_compatible"],
                default=s.provider,
            ).ask()
            if provider is None:
                continue
            s.provider = provider
            defaults = PROVIDER_DEFAULTS[provider]
            if defaults.get("base_url"):
                s.base_url = defaults["base_url"]
            if defaults.get("model_name"):
                s.model_name = defaults["model_name"]
        elif picked == "base_url":
            s.base_url = _ask_text("Base URL：", default=s.base_url)
        elif picked == "api_key":
            s.api_key = _ask_password("API Key：")
        elif picked == "model_name":
            s.model_name = _ask_text("Model Name：", default=s.model_name)
        elif picked == "ocr_backend":
            backend = questionary.select(
                "OCR 后端：",
                choices=["", "mineru", "paddle"],
                default=s.ocr_backend,
            ).ask()
            if backend is None:
                continue
            s.ocr_backend = backend
        elif picked == "ocr_api_key":
            s.ocr_api_key = _ask_password("OCR API Key：", allow_empty=True)
        elif picked == "ocr_base_url":
            s.ocr_base_url = _ask_text("OCR Base URL：", default=s.ocr_base_url)
