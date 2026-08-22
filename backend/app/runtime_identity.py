"""Deterministic identity for the code actually loaded by a research worker."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


RUNTIME_IDENTITY_SCHEMA = "factorfactory.runtime-identity/v1"
_APP_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_ROOT.parent.parent
_CRITICAL_FILES = (
    "backend/app/config.py",
    "backend/app/orchestrator.py",
    "backend/app/search_pool.py",
    "backend/app/miner/agent.py",
    "backend/app/meta/agent.py",
    "backend/app/eval/harness.py",
    "backend/app/feedback.py",
    "backend/app/factors/diversity.py",
    "backend/app/factors/return_source_governance.py",
)


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=_PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _capture_runtime_identity() -> dict:
    digest = hashlib.sha256()
    included: list[str] = []
    for relative in _CRITICAL_FILES:
        path = _PROJECT_ROOT / relative
        if not path.is_file():
            continue
        content = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        included.append(relative)
    return {
        "schema_version": RUNTIME_IDENTITY_SCHEMA,
        "code_sha256": digest.hexdigest(),
        "code_short": digest.hexdigest()[:12],
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "critical_files": included,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


LOADED_RUNTIME_IDENTITY = _capture_runtime_identity()


def runtime_identity() -> dict:
    """Return the identity frozen when this module was imported."""
    return {
        **LOADED_RUNTIME_IDENTITY,
        "critical_files": list(LOADED_RUNTIME_IDENTITY["critical_files"]),
    }
