"""OCR backends (pluggable).

Backends follow the ``OcrBackend`` protocol and are selected by name from
``phrasebank.config.Settings.ocr_backend``. Adding a new backend means
(1) implementing the protocol and (2) registering it in ``REGISTRY`` — no
``if/else`` spread across the codebase.
"""
from __future__ import annotations

from typing import Protocol

import httpx


class OcrBackend(Protocol):
    """Common surface for OCR providers.

    ``recognize`` receives a single image page's bytes and returns the
    extracted text. Network failures raise ``httpx.HTTPError`` (first-fault);
    callers decide whether to retry or skip.
    """

    name: str

    def recognize(self, page_num: int, image_bytes: bytes) -> str: ...


# name → cls  (cls 接收 api_key / base_url 关键字参数)
REGISTRY: dict[str, type[OcrBackend]] = {}


def register(name: str) -> callable:
    """Decorator form: ``@register("mineru")``."""
    def deco(cls: type[OcrBackend]) -> type[OcrBackend]:
        REGISTRY[name] = cls
        return cls
    return deco


def get_backend(name: str, *, api_key: str = "", base_url: str = "") -> OcrBackend:
    """Instantiate the requested backend or raise ConfigError."""
    from phrasebank.config import ConfigError

    if not name:
        raise ConfigError("未配置 OCR 后端。")
    cls = REGISTRY.get(name)
    if cls is None:
        raise ConfigError(f"未知 OCR 后端：{name!r}（已注册：{list(REGISTRY)}）")
    return cls(api_key=api_key, base_url=base_url)


def configured_base_url(name: str) -> str:
    if name == "mineru":
        return "https://api.mineru.net/v4"
    return ""


def http_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=60.0)
