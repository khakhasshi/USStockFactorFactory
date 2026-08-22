from pathlib import Path

import pytest

from backend.app.research_documents import (
    RESEARCH_DOCUMENT_ROOT,
    research_document_catalog,
    research_document_metadata,
    resolve_research_document,
)


DOCUMENT_SLUG = "us-seven-return-source-representatives-20260822"
ASHARE_DOCUMENT_SLUG = "ashare-four-top100-return-sources-20260822"


def test_research_document_catalog_exposes_versioned_html_report():
    catalog = research_document_catalog()
    assert catalog["schema"] == "factorfactory.research-documents/v1"
    row = next(item for item in catalog["documents"] if item["slug"] == DOCUMENT_SLUG)
    assert row["available"] is True
    assert row["sha256"]
    assert row["html_url"].endswith(f"/{DOCUMENT_SLUG}/html")
    assert "NO_COMBINATION" in row["tags"]


def test_resolved_research_document_is_confined_and_contains_audit_conclusion():
    path = resolve_research_document(DOCUMENT_SLUG)
    assert path.parent == RESEARCH_DOCUMENT_ROOT.resolve()
    assert path.suffix == ".html"
    body = path.read_text(encoding="utf-8")
    for factor_id in ("U0001", "U0002", "U0005", "U0006", "U0010", "U0012", "U0013"):
        assert factor_id in body
    assert "6个组合袖套" in body
    assert "NO_COMBINATION" in body


def test_unknown_research_document_does_not_resolve_arbitrary_paths():
    with pytest.raises(FileNotFoundError):
        research_document_metadata("../../README")
    with pytest.raises(FileNotFoundError):
        resolve_research_document("not-registered-document")


def test_ashare_return_source_report_is_registered_and_auditable():
    metadata = research_document_metadata(ASHARE_DOCUMENT_SLUG)
    assert metadata["available"] is True
    assert "RESEARCH_ONLY" in metadata["tags"]
    body = resolve_research_document(ASHARE_DOCUMENT_SLUG).read_text(encoding="utf-8")
    for source_id in ("U0001", "U0003", "U0004", "U0012", "U0015", "U0021", "U0026"):
        assert source_id in body
    assert "13.65" in body
    assert "外部美股库" in body
