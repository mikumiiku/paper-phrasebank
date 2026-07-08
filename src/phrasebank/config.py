"""Configuration layer.

All settings are terminal-interactive via `ppb config`; this module is just the
persistence carrier. Storage path: ``~/.config/ppb/config.toml`` (via
``platformdirs``). Values for the local embedding model are fixed and not
user-configurable — put in ``CONFIG_SCHEMA`` as readonly defaults.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import platformdirs
import tomllib
import tomli_w

APP_NAME = "ppb"
MIN_API_KEY_CHARS = 8


def config_dir() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME))


def data_dir() -> Path:
    return Path(platformdirs.user_data_dir(APP_NAME))


def config_path() -> Path:
    return config_dir() / "config.toml"


# ── Schema ──────────────────────────────────────────────────────────────────

CONFIG_SCHEMA: dict[str, dict[str, Any]] = {
    "provider": {
        "default": "deepseek",
        "choices": ["deepseek", "openai_compatible"],
    },
    "base_url": {"default": ""},
    "api_key": {"default": ""},
    "model_name": {"default": ""},
    "ocr_backend": {
        "default": "",
        "choices": ["", "mineru", "paddle"],
    },
    "ocr_api_key": {"default": ""},
    "ocr_base_url": {"default": ""},
}

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model_name": "deepseek-chat",
    },
    "openai_compatible": {
        "base_url": "",
        "model_name": "",
    },
}


# ── Model ───────────────────────────────────────────────────────────────────

@dataclass
class Settings:
    provider: str = "deepseek"
    base_url: str = ""
    api_key: str = ""
    model_name: str = ""
    ocr_backend: str = ""
    ocr_api_key: str = ""
    ocr_base_url: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class ConfigError(Exception):
    """Raised on first-fault configuration problems (never swallowed)."""


# ── Helpers ──────────────────────────────────────────────────────────────────

_SECRET_KEYS = {"api_key", "ocr_api_key"}


def _mask(value: str) -> str:
    """Mask a secret for terminal display: first 2 + **** + last 2."""
    if not value:
        return "<未设置>"
    if len(value) <= MIN_API_KEY_CHARS:
        return "****"
    return f"{value[:2]}****{value[-2:]}"


# ── Load / Save (thread-safe singleton) ──────────────────────────────────────

_lock = threading.Lock()
_cached: Settings | None = None


def _apply_provider_defaults(s: Settings) -> None:
    if s.base_url and s.model_name:
        return
    defaults = PROVIDER_DEFAULTS.get(s.provider, {})
    if not s.base_url and defaults.get("base_url"):
        s.base_url = defaults["base_url"]
    if not s.model_name and defaults.get("model_name"):
        s.model_name = defaults["model_name"]


def load(force: bool = False) -> Settings:
    """Load settings from disk. Caches the result unless ``force`` is True."""
    global _cached
    if _cached is not None and not force:
        return _cached

    with _lock:
        if _cached is not None and not force:
            return _cached

        path = config_path()
        s = Settings()
        if path.exists():
            try:
                raw = tomllib.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ConfigError(f"配置文件路径损坏无法解析: {path}\n{exc}") from exc
            for k, v in raw.items():
                if hasattr(s, k):
                    setattr(s, k, v)

        _apply_provider_defaults(s)
        _cached = s
        return s


def save(s: Settings) -> Path:
    """Persist settings to disk, creating the parent dir as needed."""
    with _lock:
        _apply_provider_defaults(s)
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in asdict(s).items()}
        with path.open("wb") as f:
            tomli_w.dump(payload, f)
        global _cached
        _cached = s
        return path


def exists() -> bool:
    return config_path().exists()


def require_llm_configured() -> Settings:
    """Return settings or raise ``ConfigError`` when LLM is missing."""
    s = load()
    if not s.api_key:
        raise ConfigError(
            "LLM API Key 未配置。请运行 `ppb config` 完成 LLM 配置后再执行抽取。"
        )
    return s


def display_dict(s: Settings) -> dict[str, str]:
    """Render-ready mapping with secrets masked."""
    out: dict[str, str] = {}
    for k, v in asdict(s).items():
        out[k] = _mask(v) if k in _SECRET_KEYS else v or "<未设置>"
    return out
