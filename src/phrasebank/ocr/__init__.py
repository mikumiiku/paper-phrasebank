"""OCR backends (pluggable).

Backends follow the ``OcrBackend`` protocol and are selected by name from
``phrasebank.config.Settings.ocr_backend``. Adding a new backend means
(1) implementing the protocol and (2) registering it in ``REGISTRY`` — no
``if/else`` spread across the codebase.

Each backend also exposes a ``DEFAULT_BASE_URL`` so the user is **never
asked** to supply an API base URL unless they are self-hosting / using an
alternate endpoint. ``Settings.ocr_base_url`` is still honoured when
non-empty (override semantics) to support self-hosted deployments.
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


# name → cls.  Each cls is also expected to expose a ``DEFAULT_BASE_URL``
# module-level attribute (used unless the user has set an override).
REGISTRY: dict[str, type[OcrBackend]] = {}


def register(name: str) -> callable:
    """Decorator form: ``@register("mineru")``."""
    def deco(cls: type[OcrBackend]) -> type[OcrBackend]:
        REGISTRY[name] = cls
        return cls
    return deco


def get_backend(
    name: str,
    *,
    api_key: str = "",
    base_url: str = "",
) -> OcrBackend:
    """Instantiate the requested backend or raise ``ConfigError``.

    ``base_url`` is the user-supplied override (from ``Settings.ocr_base_url``).
    When empty, the backend's ``DEFAULT_BASE_URL`` is used — so the user never
    needs to learn the official API address.
    """
    from phrasebank.config import ConfigError

    if not name:
        raise ConfigError("未配置 OCR 后端。")
    cls = REGISTRY.get(name)
    if cls is None:
        raise ConfigError(f"未知 OCR 后端：{name!r}（已注册：{list(REGISTRY)}）")

    # Backend-provided default wins over the empty override; override wins
    # when the user has explicitly set one (self-host / corporate proxy).
    effective_base_url = base_url or getattr(cls, "DEFAULT_BASE_URL", "")
    return cls(api_key=api_key, base_url=effective_base_url)


def configured_base_url(name: str) -> str:
    """Base URL shown in the config UI for a backend — the official one."""
    cls = REGISTRY.get(name)
    if cls is None:
        return ""
    return getattr(cls, "DEFAULT_BASE_URL", "")


def http_client(base_url: str) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=120.0)
