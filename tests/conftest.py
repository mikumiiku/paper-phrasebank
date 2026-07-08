"""Root-shared pytest fixtures."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="ppb_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def isolated_config(tmp_dir, monkeypatch):
    """Redirect config + data dirs under a temp path so tests never touch
    real ~/.config/ppb or ~/.local/share/ppb."""
    cfg = tmp_dir / "config"
    dat = tmp_dir / "data"
    cfg.mkdir()
    dat.mkdir()
    monkeypatch.setenv("PPB_TEST_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("PPB_TEST_DATA_DIR", str(dat))
    # the config module reads platformdirs; patch via env for those backends
    monkeypatch.setattr(
        "platformdirs.user_config_dir",
        lambda *a, **k: str(cfg),
    )
    monkeypatch.setattr(
        "platformdirs.user_data_dir",
        lambda *a, **k: str(dat),
    )
    # invalidate cached settings
    import phrasebank.config as _cfg

    _cfg._cached = None  # type: ignore[attr-defined]
    return cfg, dat
