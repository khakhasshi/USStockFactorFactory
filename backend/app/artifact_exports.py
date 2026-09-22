"""CSV compatibility exports. Parquet is authoritative for new experiments."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import polars as pl
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse

EXPORTS = {
    "settlement_statement.csv": ("settlement_statement.parquet", "fill_id,order_id,signal_date,trade_date\n"),
    "round_trip_statement.csv": ("round_trip_ledger.parquet", "symbol,entry_date,exit_date,direction,quantity,entry_price,exit_price,net_pnl,exit_reason\n"),
    "factor_attribution.csv": ("factor_attribution.parquet", ""),
}


def export_descriptor(path: Path) -> Path:
    return path.with_name(path.name + '.fdc-export.json')


def archive_export_descriptor(path: Path):
    descriptor = export_descriptor(path)
    if not descriptor.exists():
        return
    # Retain the old download identity when a user reuses an artifact directory.
    record = json.loads(descriptor.read_text())
    version = record['manifest_sha256']
    import re
    if not isinstance(version, str) or not re.fullmatch('[0-9a-f]{64}', version):
        raise RuntimeError('Invalid historical CSV descriptor identity')
    archive = descriptor.parent / '.fdc-export-history' / (version + '.json')
    archive.parent.mkdir(exist_ok=True)
    if archive.exists() and archive.read_bytes() != descriptor.read_bytes():
        raise RuntimeError('Historical CSV descriptor collision')
    import os
    if not archive.exists():
        os.link(descriptor, archive)
    descriptor.unlink()


def frozen_csv_chunks(path: Path):
    """Regenerate the byte-verified historical export from an immutable object."""
    from .config import FDC_ROOT
    import sys
    if str(FDC_ROOT) not in sys.path:
        sys.path.insert(0, str(FDC_ROOT))
    from finance_data_center.snapshots import SnapshotStore, digest, payload_hash
    record = json.loads(export_descriptor(path).read_text())
    expected_manifest = record.pop('manifest_sha256')
    if payload_hash(record) != expected_manifest:
        raise RuntimeError('Historical export manifest hash mismatch')
    if record['serializer'] != f'polars-{pl.__version__}-csv-v1':
        raise RuntimeError('Historical CSV serializer version requires validation')
    source = SnapshotStore(FDC_ROOT).object_path(record['parquet_sha256'])
    if digest(source) != record['parquet_sha256']:
        raise RuntimeError('Historical export object hash mismatch')
    hasher = hashlib.sha256()
    for block in parquet_csv_chunks(source, record['empty_header']):
        hasher.update(block)
        yield block
    if hasher.hexdigest() != record['csv_sha256']:
        raise RuntimeError('Historical CSV reconstruction hash mismatch')


def parquet_csv_chunks(path: Path, empty_header: str = ""):
    """Bounded batches, stable row order and a single header; no disk cache."""
    lazy = pl.scan_parquet(path)
    if lazy.collect_schema().names() == ["empty"]:
        yield empty_header.encode("utf-8")
        return
    first = True
    for frame in lazy.collect_batches(chunk_size=8192, maintain_order=True):
        yield frame.write_csv(include_header=first).encode("utf-8")
        first = False
    if first:
        yield lazy.limit(0).collect().write_csv().encode("utf-8")


def csv_chunks(path: Path):
    if path.exists():
        opener, source = open, path
    elif path.with_suffix(".csv.gz").exists():
        opener, source = gzip.open, path.with_suffix(".csv.gz")
    elif export_descriptor(path).is_file():
        yield from frozen_csv_chunks(path)
        return
    else:
        parquet, header = EXPORTS[path.name]
        yield from parquet_csv_chunks(path.with_name(parquet), header)
        return
    with opener(source, "rb") as handle:
        yield from iter(lambda: handle.read(256 * 1024), b"")


def export_metadata(path: Path, rows: int) -> dict:
    parquet, header = EXPORTS[path.name]
    digest = hashlib.sha256()
    for chunk in parquet_csv_chunks(path.with_name(parquet), header):
        digest.update(chunk)
    return {"filename": path.name, "sha256": digest.hexdigest(), "rows": rows,
            "storage": "on_demand", "source": parquet, "serializer": f"polars-{pl.__version__}-csv-v1"}


def csv_download(path: Path, filename: str, missing_message: str):
    if path.exists():
        return FileResponse(path, media_type="text/csv", filename=filename)
    parquet, _ = EXPORTS[path.name]
    if not path.with_suffix(".csv.gz").exists() and not path.with_name(parquet).exists() and not export_descriptor(path).is_file():
        raise HTTPException(404, missing_message)
    return StreamingResponse(csv_chunks(path), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})
