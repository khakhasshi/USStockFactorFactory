"""Immutable, content-addressed inputs shared by vector and event audits.

An audit pins one Polars frame; hot reload cannot replace it midway through
direction selection, holdout, rating or event replay. Source files are copied
to immutable objects, never hard-linked to an updater's mutable partitions.
"""
from __future__ import annotations

import contextlib
import contextvars
import glob
import functools
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from .data.panel import PanelStore

PROTOCOL = "immutable_audit_inputs_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
_ACTIVE = contextvars.ContextVar("factorfactory_frozen_panel", default=None)
_PIN_REQUESTED = contextvars.ContextVar("factorfactory_pin_on_first_read", default=False)
_CACHE: dict[tuple, dict] = {}
_LOCK = threading.RLock()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _object_copy(path: Path, root: Path) -> dict:
    before = path.stat()
    sha = hash_file(path)
    target = root / "objects" / sha[:2] / sha
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if hash_file(target) != sha:
            raise ValueError(f"Immutable input object is corrupt: {target}")
    else:
        fd, temp = tempfile.mkstemp(prefix=".snapshot-", dir=target.parent)
        os.close(fd)
        try:
            shutil.copyfile(path, temp)
            if hash_file(Path(temp)) != sha:
                raise ValueError(f"Input changed while snapshotting: {path}")
            os.chmod(temp, 0o444)
            os.replace(temp, target)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while snapshotting: {path}")
    return {"path": str(path.resolve()), "sha256": sha, "bytes": before.st_size,
            "snapshot_path": str(target), "mtime_ns": before.st_mtime_ns}


def code_snapshot(root: Path | None = None, *, persist: bool = True) -> dict:
    """Freeze executable source only; never archive credentials/runtime DBs."""
    root = Path(root or REPO_ROOT / "var" / "audit_snapshots")
    def git(*args):
        result = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, check=False)
        return result.stdout if result.returncode == 0 else b""
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    allowed = {".py", ".rs", ".js", ".ts", ".tsx", ".jsx", ".toml", ".lock"}
    source_paths = sorted({REPO_ROOT / os.fsdecode(p) for p in paths if p and
                           (Path(os.fsdecode(p)).suffix in allowed or Path(os.fsdecode(p)).name == "requirements.txt")})
    files = []
    for path in source_paths:
        if path.is_file():
            item = _object_copy(path, root) if persist else {"sha256": hash_file(path), "bytes": path.stat().st_size}
            files.append(item | {"path": str(path.relative_to(REPO_ROOT))})
    return {"git_commit": git("rev-parse", "HEAD").decode().strip() or None,
            "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
            "code_sha256": digest_json([{k: row[k] for k in ("path", "sha256")} for row in files]),
            "source_files": files, "persisted": bool(persist)}


@dataclass(frozen=True)
class FrozenPanel:
    store: Any
    frame: pl.DataFrame
    trading_dates: tuple[date, ...]
    loaded_identity: str | None
    generation: int
    manifest: dict

    def ensure_loaded(self):
        return self.frame

    def read_snapshot(self):
        return self.frame, self.trading_dates, self.loaded_identity, self.generation

    def __getattr__(self, name):
        return getattr(self.store, name)


def active_snapshot() -> FrozenPanel | None:
    return _ACTIVE.get()


def _same_store(snap: FrozenPanel, store: Any) -> bool:
    return snap.store is store or (getattr(snap.store, "market", None) == getattr(store, "market", None)
        and getattr(snap.store, "panel_glob", None) == getattr(store, "panel_glob", None)
        and getattr(store, "panel_glob", None) is not None)


def read_frozen_panel(store: Any):
    snap = active_snapshot()
    if snap is not None and _same_store(snap, store):
        return snap.read_snapshot()
    if hasattr(store, "read_snapshot"):
        return store.read_snapshot()
    frame = store.ensure_loaded()
    dates = tuple(getattr(store, "trading_dates", frame["trade_date"].unique().sort().to_list()))
    return frame, dates, None, 0


def freeze_panel(store: Any, *, persist: bool = True, root: Path | str | None = None) -> FrozenPanel:
    existing = active_snapshot()
    if existing is not None and _same_store(existing, store):
        return existing
    root = Path(root or REPO_ROOT / "var" / "audit_snapshots").resolve()
    frame, dates, identity, generation = read_frozen_panel(store)
    # Polars clone is zero-copy for Arrow buffers, but separates the DataFrame
    # container from accidental in-place column replacement by another reader.
    frame = frame.clone()
    paths = sorted(Path(p) for p in glob.glob(str(getattr(store, "panel_glob", ""))) if Path(p).is_file())
    signature = tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    # Reject stale already-loaded frames instead of stamping the new disk data
    # hash on results calculated from an older in-memory generation.
    if paths and hasattr(store, "_source_inventory"):
        current = store._source_inventory(force=True).get("identity")
        if identity != current:
            raise ValueError("Loaded panel and source partitions differ; reload before auditing")
    with _LOCK:
        key = (str(root), signature, bool(persist))
        source = _CACHE.get(key)
        if source is None:
            files = [(_object_copy(p, root) if persist else
                     {"path": str(p.resolve()), "sha256": hash_file(p), "bytes": p.stat().st_size}) for p in paths]
            source = {"files": files, "data_sha256": digest_json([{k: f[k] for k in ("path", "sha256")} for f in files])}
            _CACHE[key] = source
    after = tuple((str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths)
    if signature != after:
        raise ValueError("Panel changed during immutable snapshot; retry after updater completes")
    manifest = {"protocol": PROTOCOL, "market": getattr(store, "market", None),
                "panel_glob": getattr(store, "panel_glob", None), "loaded_identity": identity,
                "generation": generation, "rows": frame.height,
                "actual_start": str(dates[0]) if dates else None,
                "actual_end": str(dates[-1]) if dates else None,
                "market_session_count": len(dates), "market_calendar_sha256": digest_json(dates),
                "created_at": datetime.now(timezone.utc).isoformat(), **source,
                "code": code_snapshot(root, persist=persist),
                "immutable_inputs_available": bool(paths and persist),
                "status": "FROZEN" if paths and persist else "IN_MEMORY_ONLY"}
    manifest["snapshot_id"] = digest_json({k: manifest[k] for k in
        ("data_sha256", "market_calendar_sha256", "rows", "actual_start", "actual_end")})
    if persist:
        manifest_path = root / "manifests" / f"{manifest['snapshot_id']}-{manifest['code']['code_sha256']}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
            manifest_path.chmod(0o444)
        manifest["manifest_path"] = str(manifest_path)
    snap = FrozenPanel(store, frame, tuple(dates), identity, generation, manifest)
    if _PIN_REQUESTED.get():
        _ACTIVE.set(snap)
    return snap


def pin_backtest_inputs(function):
    """Pin lazily on first materialization, including all ensemble sleeves."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if _PIN_REQUESTED.get():
            return function(*args, **kwargs)
        active_token = _ACTIVE.set(_ACTIVE.get())
        pin_token = _PIN_REQUESTED.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _PIN_REQUESTED.reset(pin_token)
            _ACTIVE.reset(active_token)
    return wrapped


@contextlib.contextmanager
def frozen_panel(panel_glob: str | None = None, market: str = "us", *, persist: bool = True,
                 root: Path | str | None = None, store: Any = None):
    store = store if store is not None else PanelStore.get(panel_glob, market)
    snap = freeze_panel(store, persist=persist, root=root)
    token = _ACTIVE.set(snap)
    try:
        yield snap
    finally:
        _ACTIVE.reset(token)


def build_run_provenance(expression: str, config: dict, start: str, end: str,
                         actual_dates, frame: pl.DataFrame | None = None,
                         snapshot: FrozenPanel | None = None) -> dict:
    snap = snapshot or active_snapshot()
    dates = sorted(actual_dates)
    result = {"protocol": PROTOCOL, "panel": snap.manifest if snap else None,
              "expression": expression, "expression_sha256": hashlib.sha256(expression.encode()).hexdigest(),
              "config_sha256": digest_json(config), "requested_start": start, "requested_end": end,
              "actual_start": str(dates[0]) if dates else None,
              "actual_end": str(dates[-1]) if dates else None,
              "actual_sessions": len(dates), "actual_calendar_sha256": digest_json(dates),
              "immutable_inputs_available": bool(snap and snap.manifest.get("immutable_inputs_available"))}
    if frame is not None:
        result["prepared_frame_rows"] = frame.height
        result["prepared_frame_sha256"] = hashlib.sha256(frame.hash_rows(seed=7643).to_numpy().tobytes()).hexdigest()
        result["prepared_frame_schema_sha256"] = digest_json({k: str(v) for k, v in frame.schema.items()})
    result["run_fingerprint"] = digest_json(result)
    return result
