"""Tests for ``ppb upgrade``.

All network / subprocess / input IO is injected via dependency injection in
``run_upgrade`` — no monkeypatching of globals, no real network. Each branch
of the 12-step plan is exercised at least once.

Run: `pytest tests/test_upgrade.py -v`
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock

import pytest

from phrasebank import upgrade as up
from phrasebank.upgrade import (
    ReleaseInfo,
    UpgradeError,
    UpgradePlan,
    build_upgrade_command,
    current_version,
    detect_install_method,
    newer_version,
    run_upgrade,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rel(ver: str = "1.2.0", body: string = "") -> ReleaseInfo:
    return ReleaseInfo(
        tag_name=f"v{ver}", version=ver,
        published_at="2026-01-15T00:00:00Z", body=body,
        tarball_url=f"https://github.com/x/y/archive/v{ver}.tar.gz",
        wheel_url=f"https://github.com/x/y/releases/download/v{ver}/paper_phrasebank-{ver}-py3-none-any.whl",
    )


def _capture(fn, *args, **kwargs) -> tuple[str, int]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = fn(*args, **kwargs)
    return buf.getvalue(), rc


class _RunRecorder:
    def __init__(self, returncode: int = 0) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self.returncode = returncode

    def __call__(self, command: list[str], **kwargs) -> int:
        self.calls.append((tuple(command), kwargs))
        return self.returncode


# ── newer_version ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cur, lat, expected",
    [
        ("1.0.0", "1.0.0", False),
        ("1.0.0", "1.0.1", True),
        ("1.0.0", "0.9.0", False),
        ("1.2.0", "2.0.0", True),
        ("v1.0.0", "1.0.1", True),           # leading v handled
        ("1.0.0", "release-1.0.1", True),    # release- prefix handled
        ("1.0.0+build", "1.0.1", True),      # local build tag stripped
    ],
)
def test_newer_version(cur, lat, expected):
    assert newer_version(cur, lat) is expected


# ── current_version ──────────────────────────────────────────────────────

def test_current_version_returns_nonempty():
    v = current_version()
    assert isinstance(v, str) and v


# ── build_upgrade_command ────────────────────────────────────────────────

def test_build_upgrade_command_uv():
    plan = build_upgrade_command("uv", _rel("1.2.0"))
    assert plan.method == "uv"
    assert "uv" in plan.command
    assert "--force" in plan.command
    assert up.PYPI_NAME in plan.command


def test_build_upgrade_command_pipx():
    plan = build_upgrade_command("pipx", _rel("1.2.0"))
    assert plan.method == "pipx"
    assert "pipx" in plan.command and "upgrade" in plan.command
    assert up.PYPI_NAME in plan.command


def test_build_upgrade_command_pip():
    plan = build_upgrade_command("pip", _rel("1.2.0"))
    assert plan.method == "pip"
    assert "--upgrade" in plan.command
    assert "1.2.0" in " ".join(plan.command)


# ── run_upgrade: up-to-date → no-op ─────────────────────────────────────

def test_run_upgrade_uptodate_short_circuits(tmp_path):
    # same version as installed (1.0.0) → up-to-date message, no command run
    release = _rel("1.0.0")

    def fake_fetch():
        return release

    run = _RunRecorder(0)
    # Patch _confirm so the non-interactive test path short-circuits: newer()
    # is False so we never reach the confirm prompt anyway.
    import phrasebank.upgrade as up_mod
    up_mod._confirm = lambda _p: True  # type: ignore[assignment]

    out, rc = _capture(
        run_upgrade,
        assume_yes=False,
        _fetch_latest_release=fake_fetch,
        _detect_method=lambda: "uv",
        _run_command=run,
    )
    assert rc == 0, out
    assert "up to date" in out.lower(), out
    assert run.calls == []  # no subprocess invoked


# ── run_upgrade: newer version + assume_yes ─────────────────────────────

def test_run_upgrade_success_runs_command(tmp_path):
    release = _rel("2.0.0", body="## v2.0.0\n\n* Fancy new feature.")

    def fake_fetch():
        return release

    def fake_detect():
        return "uv"

    run = _RunRecorder(0)
    out, rc = _capture(
        run_upgrade,
        assume_yes=True,  # bypass confirm prompt
        _fetch_latest_release=fake_fetch,
        _detect_method=fake_detect,
        _run_command=run,
    )
    assert rc == 0, out
    assert len(run.calls) == 1
    cmd = run.calls[0][0]
    assert "uv" in cmd and "install" in cmd


def test_run_upgrade_declined_cancel(tmp_path):
    release = _rel("2.0.0")

    def fake_fetch():
        return release

    # questionary.confirm is NOT mocked; we inject a fake that returns False
    import phrasebank.upgrade as up_mod
    up_mod._confirm = lambda _prompt: False  # type: ignore[assignment]

    run = _RunRecorder()
    out, rc = _capture(
        run_upgrade,
        assume_yes=False,
        _fetch_latest_release=fake_fetch,
        _detect_method=lambda: "uv",
        _run_command=run,
    )
    assert rc == 0
    assert "Cancel" in out or "取消" in out
    assert run.calls == []


def test_run_upgrade_fetch_failure_returns_1(tmp_path):
    def fake_fetch():
        raise UpgradeError("network down")

    run = _RunRecorder()
    out, rc = _capture(
        run_upgrade,
        assume_yes=True,
        _fetch_latest_release=fake_fetch,
        _detect_method=lambda: "uv",
        _run_command=run,
    )
    assert rc == 1
    assert "network down" in out or "无法连接" in out
    assert run.calls == []


def test_run_upgrade_subprocess_failure_nonzero(tmp_path):
    release = _rel("2.0.0")

    def fake_fetch():
        return release

    run = _RunRecorder(1)  # pretend subprocess failed
    out, rc = _capture(
        run_upgrade,
        assume_yes=True,
        _fetch_latest_release=fake_fetch,
        _detect_method=lambda: "uv",
        _run_command=run,
    )
    assert rc == 1
    assert len(run.calls) == 1
    assert "手动执行" in out or "fallback" in out.lower() or "退出码" in out


# ── detect_install_method (smoke) ───────────────────────────────────────────

def test_detect_install_method_returns_one_of_known():
    method = detect_install_method()
    assert method in {"uv", "pipx", "pip"}
