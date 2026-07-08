"""Config: load / save / mask / provider defaults / require_llm_configured."""
from __future__ import annotations

import pytest

from phrasebank import config


def _fresh_settings(**kw) -> config.Settings:
    s = config.Settings(**kw)
    return s


def test_mask_long_secret():
    assert config._mask("sk-abcdef123456") == "sk****56"


def test_mask_short_secret():
    assert config._mask("short") == "****"


def test_mask_empty_shows_unset():
    assert config._mask("") == "<未设置>"


def test_save_load_round_trip(isolated_config):
    cfg, _ = isolated_config
    s = provider_configured() if False else config.Settings(
        provider="deepseek",
        api_key="sk-abcdefghijklm",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )
    config.save(s)
    assert config.exists()
    loaded = config.load(force=True)
    assert loaded.api_key == s.api_key
    assert loaded.model_name == s.model_name


def test_provider_defaults_applied_on_load(isolated_config):
    s = config.Settings(
        provider="deepseek",
        api_key="k",
        # base_url / model_name intentionally empty
    )
    config.save(s)
    loaded = config.load(force=True)
    assert loaded.base_url == config.PROVIDER_DEFAULTS["deepseek"]["base_url"]
    assert loaded.model_name == config.PROVIDER_DEFAULTS["deepseek"]["model_name"]


def test_require_llm_configured_ok(isolated_config):
    s = config.Settings(provider="openai_compatible", api_key="k", base_url="u", model_name="m")
    config.save(s)
    got = config.require_llm_configured()
    assert got.api_key == "k"


def test_require_llm_configured_unset_raises(isolated_config):
    # reset cache and make sure the config file reports no key
    import phrasebank.config as _cfg

    _cfg._cached = None
    config.save(config.Settings(api_key=""))
    with pytest.raises(config.ConfigError):
        config.require_llm_configured()


def test_display_dict_masks_secrets(isolated_config):
    s = config.Settings(api_key="sk-1234567890", ocr_api_key="ocr-key-long")
    d = config.display_dict(s)
    assert d["api_key"] == "sk****90"
    assert d["ocr_api_key"].endswith("****") or "**" in d["ocr_api_key"]
