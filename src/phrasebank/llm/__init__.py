"""LLM extraction layer: pluggable OpenAI-compatible client + prompt templates."""
from __future__ import annotations

import logging

import openai

import phrasebank.config as pconfig

log = logging.getLogger(__name__)


_cache: openai.OpenAI | None = None


def get_client() -> openai.OpenAI:
    """Return a module-level cached ``openai.OpenAI`` client built from config."""
    global _cache
    if _cache is not None:
        return _cache

    s = pconfig.require_llm_configured()
    _cache = openai.OpenAI(base_url=s.base_url or None, api_key=s.api_key)
    return _cache


def reset_client() -> None:
    """Drop the cached client (used by tests)."""
    global _cache
    _cache = None


from .client import LLMClient  # noqa: E402,F401
from .prompts import SYSTEM_METADATA, USER_METADATA, SENTENCE_SYSTEM, SENTENCE_USER  # noqa: F401
