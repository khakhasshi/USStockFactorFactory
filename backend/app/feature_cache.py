"""Feature matrices with shared columns in the configured data center."""
import sys
from pathlib import Path
import polars as pl
from .config import FDC_ROOT


def store():
    if str(FDC_ROOT) not in sys.path:
        sys.path.insert(0, str(FDC_ROOT))
    from finance_data_center.column_store import ColumnStore
    return ColumnStore(FDC_ROOT)


def exists(path):
    return Path(path).is_file() or store().exists(path)


def read(path):
    if Path(path).is_file():
        return pl.read_parquet(path)
    return pl.from_arrow(store().read(path))


def write(path, frame):
    # Publish and verify the new immutable version before removing a stale cache.
    # Historical versions remain addressable in the data-center registry.
    path = Path(path)
    storage = store()
    import hashlib
    key = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    with storage.store.lock('feature-cache-' + key):
        if path.exists():
            from finance_data_center.filesystem_sharing import writable_files
            storage.import_file(path)
            journal = FDC_ROOT / 'registry/column_tables/cache-replacement.jsonl'
            storage.retire(path, journal=journal, busy=writable_files())
        return storage.write(path, frame.to_arrow())
