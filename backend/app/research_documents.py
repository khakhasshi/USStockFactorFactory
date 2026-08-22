"""Versioned, read-only catalog for important project research documents."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DOCUMENT_ROOT = PROJECT_ROOT / "docs" / "research"
CATALOG_PATH = RESEARCH_DOCUMENT_ROOT / "catalog.json"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,95}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_rows() -> list[dict[str, Any]]:
    if not CATALOG_PATH.is_file():
        return []
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    rows = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("研究文档 catalog.json 必须包含 documents 数组")

    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    root = RESEARCH_DOCUMENT_ROOT.resolve()
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("研究文档目录项必须是对象")
        slug = str(raw.get("slug") or "").strip()
        if not _SLUG_RE.fullmatch(slug) or slug in seen:
            raise ValueError(f"研究文档 slug 非法或重复: {slug}")
        seen.add(slug)
        filename = str(raw.get("file") or "").strip()
        if not filename.endswith(".html") or Path(filename).name != filename:
            raise ValueError(f"研究文档文件名非法: {filename}")
        path = (RESEARCH_DOCUMENT_ROOT / filename).resolve()
        if path.parent != root:
            raise ValueError(f"研究文档越过受控目录: {filename}")
        available = path.is_file()
        output.append({
            "slug": slug,
            "title": str(raw.get("title") or slug),
            "subtitle": str(raw.get("subtitle") or ""),
            "summary": str(raw.get("summary") or ""),
            "category": str(raw.get("category") or "研究结论"),
            "status": str(raw.get("status") or "research_note"),
            "published_at": str(raw.get("published_at") or ""),
            "updated_at": str(raw.get("updated_at") or raw.get("published_at") or ""),
            "tags": [str(value) for value in (raw.get("tags") or [])],
            "file": filename,
            "available": available,
            "size_bytes": path.stat().st_size if available else None,
            "sha256": _sha256(path) if available else None,
            "html_url": f"/api/research-documents/{slug}/html",
        })
    return sorted(output, key=lambda row: (row["updated_at"], row["slug"]), reverse=True)


def research_document_catalog() -> dict[str, Any]:
    rows = _catalog_rows()
    return {
        "schema": "factorfactory.research-documents/v1",
        "root": str(RESEARCH_DOCUMENT_ROOT),
        "count": len(rows),
        "documents": rows,
    }


def research_document_metadata(slug: str) -> dict[str, Any]:
    for row in _catalog_rows():
        if row["slug"] == slug:
            return row
    raise FileNotFoundError(slug)


def resolve_research_document(slug: str) -> Path:
    row = research_document_metadata(slug)
    if not row["available"]:
        raise FileNotFoundError(row["file"])
    return (RESEARCH_DOCUMENT_ROOT / row["file"]).resolve()
