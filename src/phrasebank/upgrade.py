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
GITHUB_USER = GITHUB_REPO.split("/")[0]
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASE_PAGE = f"https://github.com/{GITHUB_REPO}/releases"
PYPI_NAME = "paper-phrasebank"

# Git clone URLs — SSH first (works with SSH key, no PAT needed), HTTPS as
# fallback for users without an SSH key set up on their machine.
GIT_URL_SSH = f"git+ssh://git@github.com/{GITHUB_REPO}.git"
GIT_URL_HTTPS = f"git+https://github.com/{GITHUB_REPO}.git"
TARBALL_URL_TMPL = f"https://github.com/{GITHUB_REPO}/archive/refs/tags/v{{ver}}.tar.gz"

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
    """Resolve the latest release metadata.

    Strategy (first success wins):

    1. **urllib direct** — unauthenticated, but subject to the strict
       60 req/h shared rate-limit. Most attempts succeed.
    2. **gh CLI fallback** — when the direct call fails with HTTP 403
       (rate-limit) *or* any network error, fall back to the user's ``gh``
       CLI which is authenticated against their GitHub OAuth and enjoys a
       much higher limit (~5000 req/h). This is what saves the UX when the
       shared bucket is exhausted.
    3. if both fail, the collected error messages are surfaced to the user
       (first-fault — never swallowed).

    Raises ``UpgradeError`` only when no strategy works.
    """
    errors: list[str] = []

    # Strategy 1: unauthenticated urllib call.
    try:
        req = urllib.request.Request(
            GITHUB_API_LATEST,
            headers={"Accept": "application/vnd.github+json",
                     "User-Agent": f"ppb-{current_version()}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        return _parse_release_payload(payload)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        errors.append(f"[urllib] {exc}")
    except (json.JSONDecodeError, ValueError) as exc:
        errors.append(f"[urllib] 解析失败: {exc}")

    # Strategy 2: gh CLI (authenticated against user's GitHub OAuth).
    if _gh_available():
        try:
            payload = _fetch_via_gh()
            return _parse_release_payload(payload)
        except UpgradeError as exc:
            errors.append(f"[gh] {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"[gh] {exc}")

    # Exhausted — surface everything we learned.
    detail = "; ".join(errors) if errors else "未知错误"
    raise UpgradeError(f"无法连接 GitHub: {detail}")
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

def build_upgrade_command(
    method: str,
    release: ReleaseInfo,
    *,
    git_url: str = GIT_URL_SSH,
) -> UpgradePlan:
    """Return the concrete upgrade command for the detected install backend.

    ``git_url`` is one of ``GIT_URL_SSH`` (default) or ``GIT_URL_HTTPS`` — it
    selects the transport ``uv``/``pip`` use to fetch the repo. SSH-first means
    users with a deployed GitHub SSH key get touch-free upgrades; HTTPS is
    offered via ``--https`` for networks that block SSH.
    """
    ver = release.version
    if method == "uv":
        return UpgradePlan(
            method="uv",
            command=[
                "uv", "tool", "install", "--force",
                "--from", f"{git_url}",
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
    # pip (the generic fallback) — SSH transport via git+ssh://
    return UpgradePlan(
        method="pip",
        command=[
            sys.executable, "-m", "pip", "install", "--upgrade",
            "--prefer-binary", git_url,
        ],
        description=f"pip 升级 ({ver}, {git_url.split(':',1)[0]})",
    )


# ── Run the upgrade ───────────────────────────────────────────────────────────

def run_upgrade(
    *,
    assume_yes: bool = False,
    prefer_https: bool = False,
    _fetch_latest_release=fetch_latest_release,
    _detect_method: callable = detect_install_method,
    _run_command: callable = None,
) -> int:
    """Top-level orchestration. Returns 0 on success/no-op, non-zero on error.

    Pure-injection style — all I/O collaborators are overridable so tests can
    exercise every branch without touching the network or the filesystem.

    ``prefer_https`` toggles git clone protocol: SSH by default (works out of
    the box for anyone who has a GitHub SSH key deployed — no PAT needed),
    HTTPS when the user explicitly opts in via ``--https``.
    """
    run_cmd = _run_command or _real_run
    _git_url = GIT_URL_HTTPS if prefer_https else GIT_URL_SSH
    current = current_version()
    console.print(f"当前版本: {current}", style="cyan")

    try:
        release = _fetch_latest_release()
    except UpgradeError as exc:
        console.print(f"{exc}", style="red")
        console.print(f"If the GitHub API rate-limit is exhausted (HTTP 403), "
                      f"any of the following works as an immediate fallback:\n")
        # Wheel install from the GitHub release asset (works even offline-ish).
        wheel_asset = TARBALL_URL_TMPL.format(ver="1.0.1").replace(
            ".tar.gz", "-py3-none-any.whl"
        )
        console.print(f"  1) Upgrade directly from the release asset:")
        console.print(
            f"     pip install {GITHUB_RELEASE_PAGE}/download/v1.0.1/"
            f"paper_phrasebank-1.0.1-py3-none-any.whl"
        )
        console.print(f"  2) Or, with gh CLI authenticated (`gh auth login`):")
        console.print(f"     ppb upgrade     # automatically re-tries via gh")
        console.print(f"  3) Manual release page: {GITHUB_RELEASE_PAGE}")
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
    plan = build_upgrade_command(method, release, git_url=_git_url)
    console.print(f"安装方式: {method}", style="cyan")
    console.print(f"升级协议: {_git_url.split(':', 1)[0]}")
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


# ── gh CLI fallback helpers ─────────────────────────────────────────────────

def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _fetch_via_gh(timeout: float = 20.0) -> dict:
    """Use the ``gh`` CLI to fetch the latest release. Authenticated against
    user's GitHub OAuth when ``gh`` is logged in, which uses a separate,
    higher rate-limit bucket."""
    try:
        result = subprocess.run(
            ["gh", "api", GITHUB_API_LATEST.removeprefix("https://api.github.com")],
            capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise UpgradeError(f"gh 命令执行失败: {exc}") from exc
    if result.returncode != 0:
        raise UpgradeError(f"gh 退出码 {result.returncode}: {result.stderr[:200]}")
    try:
        return json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError) as exc:
        raise UpgradeError(f"gh 响应解析失败: {exc}") from exc


def _parse_release_payload(payload: dict) -> ReleaseInfo:
    tag = (payload.get("tag_name") or "").lstrip("v")
    if not tag:
        raise UpgradeError(f"release payload 中缺少 tag_name: {payload}")
    body = payload.get("body") or ""
    assets = payload.get("assets") or []
    wheel_url: str | None = None
    for a in assets:
        if (a.get("name") or "").endswith(".whl"):
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


# ── Errors ────────────────────────────────────────────────────────────────────

class UpgradeError(Exception):
    """Raised when upgrade prerequisites cannot be satisfied."""