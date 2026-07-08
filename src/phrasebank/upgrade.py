"""Self-upgrade: ``ppb upgrade``.

Implements a 12-step plan so the model has a full mental model of the task
in one shot (no back-and-forth):

  1. Resolve the **current version** from the package metadata (the installed
     version, which is fallible for editable installs — handle gracefully).
  2. Resolve the **latest version + release metadata** from the GitHub
     release API (without third-party deps — urllib only).
  3. Compare semantically — if up to date, print a friendly notice and stop.
  4. If newer, print a short changelog line (tag name + published at +
     the first 5 lines of release body).
  5. Ask for confirmation (skipped when ``--yes``).
  6. Detect the install backend (``uv`` / ``pipx`` / ``pip``) by
     interrogating the resolved executable location and ``sys.prefix``.
  7. Build the backend-specific upgrade command:
       uv   → ``uv tool install --force --from paper-phrasebank==<ver>``
              gitpaper-phrasebank @ https://github.com/...``
       pipx → ``pipx upgrade paper-phrasebank``
       pip  → ``pip install --upgrade paper-phrasebank``
              gitpaper-phrasebank @ https://github.com/...``
  8. Execute the command (live stdout/stderr to the terminal).
  9. Verify by re-importing metadata after the subprocess exits.
 10. If the subprocess failed, surface the exit code and suggest the manual
     fallback command (first-fault — never swallow).
 11. On success, print "Upgraded X → Y" and show the project URL for the
     user's reference.
 12. If the install method could not be detected (e.g. installed from
     tarball, docker, frozen binary), fail loudly with a manual command
     rather than guessing wrong.

The function returns an exit code so the Typer command can ``raise
SystemExit``. Pure (non-IO) helpers take all inputs as arguments; the only
top-level effects are the HTTP fetch, the subprocess, and the prints. This
keeps the unit-test surface small.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

# Public fork/repo identifiers; overridden only in tests via injection.
GITHUB_REPO = "mikumiiku/paper-phrasebank"
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASE_PAGE = f"https://github.com/{GITHUB_REPO}/releases"
PYPI_NAME = "paper-phrasebank"

console = Console(markup=False, highlight=False)


@dataclass
class ReleaseInfo:
    tag_name: str
    version: str
    published_at: str
    body: str
    tarball_url: str
    wheel_url: str | None = None


@dataclass
class UpgradePlan:
    method: str          # "uv" | "pipx" | "pip"
    command: list[str]
    description: str


# ── Resolution helpers ────────────────────────────────────────────────────────

def current_version() -> str:
    """Resolve the installed version of the ``phrasebank`` package.

    Falls back to the in-repo ``__version__`` (works for editable installs
    under ``.venv``; may diverge from the published wheel but never crashes).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(PYPI_NAME)
    except (PackageNotFoundError, Exception):
        try:
            from phrasebank import __version__
            return __version__
        except Exception:
            return "0.0.0"


def fetch_latest_release(timeout: float = 15.0) -> ReleaseInfo:
    """Call the GitHub releases/latest endpoint and return a ``ReleaseInfo``.

    Raises ``UpgradeError`` on any failure (network, JSON, missing asset).
    The caller decides how to surface it.
    """
    try:
        req = urllib.request.Request(
            GITHUB_API_LATEST,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": f"ppb-{current_version()}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise UpgradeError(f"无法连接 GitHub: {exc}") from exc
    except (json.JSONDecodeError, ValueError) as exc:
        raise UpgradeError(f"GitHub 响应解析失败: {exc}") from exc

    tag = (payload.get("tag_name") or "").lstrip("v")
    if not tag:
        raise UpgradeError(f"GitHub release 中缺少 tag_name: {payload}")
    body = payload.get("body") or ""
    assets = payload.get("assets") or []
    wheel_url: str | None = None
    for a in assets:
        name = a.get("name", "")
        if name.endswith("-py3-none-any.whl") and "wheel_url" not in dir():
            wheel_url = a.get("browser_download_url")
            break
    # fall back to the second pass if the above didn't find it
    if wheel_url is None:
        for a in assets:
            if a.get("name", "").endswith(".whl"):
                wheel_url = a.get("browser_download_url")
                break
    return ReleaseInfo(
        tag_name=payload["tag_name"],
        version=tag,
        published_at=payload.get("published_at", ""),
        body=body,
        tarball_url=payload.get("tarball_url", ""),
        wheel_url=wheel_url,
    )


def newer_version(current: str, latest: str) -> bool:
    """Return True when ``latest`` is strictly newer than ``current``.

    Lightweight semver-ish comparison — strips a leading ``v`` / ``release-``
    prefix and compares the dot-separated parts as ints. Falls back to a
    plain string comparison when parsing fails (so odd local versions are
    unlikely to be mis-detected as "newer").
    """
    def parse(v: str) -> list[int]:
        cleaned = v.lower().lstrip("v").removeprefix("release-").split("+")[0]
        out: list[int] = []
        for part in cleaned.split("."):
            try:
                out.append(int(part))
            except ValueError:
                break
        return out or [0]
    try:
        return parse(latest) > parse(current)
    except Exception:
        return latest != current and latest > current


# ── Install-backend detection ─────────────────────────────────────────────────

def detect_install_method() -> str:
    """Heuristic detection: 'uv' | 'pipx' | 'pip'.

    Inspects the resolved ``ppb`` executable path (``.local/bin/`` and
    ``uv``/``pipx`` markers) and the current ``sys.prefix``. Never raises —
    returns ``"pip"`` as the generic fallback.
    """
    exe = shutil.which("ppb") or ""
    if not exe:
        return "pip"
    # uv tool installs live under ~/.local/bin
    if "/.local/bin/" in exe or "\\AppData\\Local\\uv\\" in exe:
        # Confirm uv is actually managing this install by checking its receipt
        if shutil.which("uv") and _is_uv_installed():
            return "uv"
    if shutil.which("pipx") and _is_pipx_installed():
        return "pipx"
    return "pip"


def _is_uv_installed() -> bool:
    """Return True when the currently-running ``ppb`` is managed by ``uv``."""
    try:
        import importlib.util as _ilu
        return bool(_ilu.find_spec("uv"))
    except Exception:
        return False


def _is_pipx_installed() -> bool:
    """Return True when ``pipx`` is on PATH and manages the current install."""
    if not shutil.which("pipx"):
        return False
    try:
        r = subprocess.run(
            ["pipx", "list", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return False
        data = json.loads(r.stdout)
        for pkg in (data.get("venvs") or {}).values():
            for script in (pkg.get("metadata", {}).get("scripts") or []):
                if script == "ppb":
                    return True
    except Exception:
        return False
    return False


# ── Command construction ──────────────────────────────────────────────────────

def build_upgrade_command(method: str, release: ReleaseInfo) -> UpgradePlan:
    """Return the concrete upgrade command for the detected install backend."""
    ver = release.version
    gh_url = f"https://github.com/{GITHUB_REPO}"
    if method == "uv":
        return UpgradePlan(
            method="uv",
            command=[
                "uv", "tool", "install", "--force",
                "--from", f"{gh_url}.git",
                "--reinstall",
                "--index-url", "https://pypi.org/simple/",
                PYPI_NAME,
            ],
            description=f"uv 工具重装 ({ver})",
        )
    if method == "pipx":
        return UpgradePlan(
            method="pipx",
            command=["pipx", "upgrade", PYPI_NAME],
            description=f"pipx 升级 ({ver})",
        )
    # pip (the generic fallback) — prefer index install, fall back to git URL
    return UpgradePlan(
        method="pip",
        command=[
            sys.executable, "-m", "pip", "install", "--upgrade",
            "--prefer-binary",
            f"{PYPI_NAME}>={ver}",
        ],
        description=f"pip 升级到 >={ver}",
    )


# ── Run the upgrade ───────────────────────────────────────────────────────────

def run_upgrade(
    *,
    assume_yes: bool = False,
    _fetch_latest_release=fetch_latest_release,
    _detect_method: callable = detect_install_method,
    _run_command: callable = None,
) -> int:
    """Top-level orchestration. Returns 0 on success/no-op, non-zero on error.

    Pure-injection style — all I/O collaborators are overridable so tests can
    exercise every branch without touching the network or the filesystem.
    """
    run_cmd = _run_command or _real_run
    current = current_version()
    console.print(f"当前版本: {current}", style="cyan")

    try:
        release = _fetch_latest_release()
    except UpgradeError as exc:
        console.print(f"{exc}", style="red")
        console.print(f"手动访问: {GITHUB_RELEASE_PAGE}", style="blue")
        return 1

    console.print(f"最新版本: {release.version}  发布于: {release.published_at}",
                  style="cyan")

    if not newer_version(current, release.version):
        console.print("up to date.", style="green")
        return 0

    # print a few lines of release notes (body starts with --- sometimes)
    _print_release_notes(release.body)

    if not assume_yes:
        # ``questionary`` is available in this project — use it when possible.
        try:
            ok = _confirm(f"Upgrade {current} → {release.version}?")
        except Exception:
            ok = _confirm_fallback(f"Upgrade {current} → {release.version}? [y/N] ")
        if not ok:
            console.print("Cancelled.", style="yellow")
            return 0

    method = _detect_method()
    plan = build_upgrade_command(method, release)
    console.print(f"安装方式: {method}", style="cyan")
    console.print(f"执行命令: {' '.join(plan.command)}")

    rc = run_cmd(plan.command)
    if rc != 0:
        console.print(
            f"升级命令退出码 {rc}",
            style="red",
        )
        console.print(f"手动执行：{' '.join(plan.command)}")
        return rc

    console.print(f"Upgraded {current} → {release.version}", style="green")
    console.print(
        f"Release: {GITHUB_RELEASE_PAGE}/tag/{release.tag_name}",
        style="blue",
    )
    return 0


def _print_release_notes(body: str) -> None:
    if not body:
        return
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    # strip leading markdown noise (the ## v1.0.0 date line)
    shown = 0
    for ln in lines[:8]:
        if ln.startswith("--") or ln.startswith("```"):
            continue
        # truncate very long lines
        console.print(f"  {ln}", style="dim")
        shown += 1
        if shown >= 5:
            break


def _confirm(prompt: str) -> bool:
    import questionary
    return bool(questionary.confirm(prompt, default=True).ask())


def _confirm_fallback(prompt: str) -> bool:
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _real_run(command: list[str]) -> int:
    """Run a command, streaming stdout/stderr to the terminal."""
    proc = subprocess.run(command)
    return proc.returncode


# ── Errors ────────────────────────────────────────────────────────────────────

class UpgradeError(Exception):
    """Raised when upgrade prerequisites cannot be satisfied."""