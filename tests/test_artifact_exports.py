import gzip
import hashlib
import io
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import polars as pl
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.artifact_exports import EXPORTS, csv_chunks, export_metadata
from backend.app.api import routes
from backend.app.backtest.engine import _write_artifacts


def client_for(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "BACKTEST_ARTIFACT_ROOT", tmp_path)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_real_download_routes_lazy_and_legacy(monkeypatch, tmp_path):
    client = client_for(monkeypatch, tmp_path)
    directory = tmp_path / "00000001"
    directory.mkdir()
    frame = pl.DataFrame({"code": ["000001", "带逗号,换行\n"], "price": [1.23456789, None],
                          "date": [date(2026, 1, 2), date(2026, 1, 3)]})
    for name, (parquet, _) in EXPORTS.items():
        frame.write_parquet(directory / parquet)
    for endpoint, name in [("statement", "settlement_statement.csv"),
                           ("round-trips", "round_trip_statement.csv"),
                           ("factor-attribution", "factor_attribution.csv")]:
        url = f"/api/backtests/1/{endpoint}.csv"
        response = client.get(url)
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        assert response.content == frame.write_csv().encode()
        assert hashlib.sha256(response.content).hexdigest() == export_metadata(directory / name, 2)["sha256"]
        assert not (directory / name).exists()
        original = b'legacy,exact\r\n"",1.0000\r\n'
        archive = (directory / name).with_suffix(".csv.gz")
        archive.write_bytes(gzip.compress(original))
        assert client.get(url).content == original
        (directory / name).write_bytes(b"existing,csv\n")
        assert client.get(url).content == b"existing,csv\n"
    assert client.get("/api/backtests/999/statement.csv").status_code == 404


def test_empty_and_concurrent_batched_export(tmp_path):
    for name, (parquet, header) in EXPORTS.items():
        pl.DataFrame({"empty": []}).write_parquet(tmp_path / parquet)
        assert b"".join(csv_chunks(tmp_path / name)) == header.encode()
    frame = pl.DataFrame({"id": range(20000), "text": ["a,\"b\n"] * 20000})
    frame.write_parquet(tmp_path / "settlement_statement.parquet")
    def download(_):
        return b"".join(csv_chunks(tmp_path / "settlement_statement.csv"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, range(4)))
    assert all(x == frame.write_csv().encode() for x in results)
    assert pl.read_csv(io.BytesIO(results[0])).height == 20000
    assert not list(tmp_path.glob("*.csv"))


def test_object_file_links_work_but_escaped_directories_do_not(monkeypatch, tmp_path):
    import pytest
    from fastapi import HTTPException
    root = tmp_path / 'artifacts'
    root.mkdir()
    directory = root / '00000001'
    directory.mkdir()
    obj = tmp_path / 'immutable.csv'
    obj.write_bytes(b'price\n1.25\n')
    (directory / 'settlement_statement.csv').symlink_to(obj)
    client = client_for(monkeypatch, root)
    response = client.get('/api/backtests/1/statement.csv')
    assert response.status_code == 200
    assert response.content == obj.read_bytes()
    (root / '00000002').symlink_to(tmp_path, target_is_directory=True)
    assert client.get('/api/backtests/2/statement.csv').status_code == 400
    with pytest.raises(HTTPException):
        routes._artifact_path(1, '../../immutable.csv')


def test_reused_artifact_directory_does_not_download_stale_csv(tmp_path):
    import json
    (tmp_path / 'settlement_statement.csv').write_text('old,data\n')
    (tmp_path / 'settlement_statement.csv.fdc-export.json').write_text(json.dumps({'manifest_sha256': 'a' * 64}))
    (tmp_path / 'round_trip_statement.csv.gz').write_bytes(gzip.compress(b'old,data\n'))
    result = dict(trades=[{'fill_id': 'new', 'quantity': 0.5}], events=[],
                  daily_steps=[], config={}, fee_schedule={}, stats={}, integrity={})
    manifest = _write_artifacts(result, tmp_path)
    data = b''.join(csv_chunks(tmp_path / 'settlement_statement.csv'))
    assert b'new,0.5' in data
    assert hashlib.sha256(data).hexdigest() == manifest['files']['statement_csv']['sha256']
    assert not list(tmp_path.glob('*.csv*'))
    assert (tmp_path / '.fdc-export-history' / ('a'*64 + '.json')).exists()


def test_frozen_download_survives_mutated_parquet(monkeypatch, tmp_path):
    import json, sys
    from backend.app import config
    from backend.app.artifact_exports import export_descriptor
    sys.path.insert(0, str(config.FDC_ROOT))
    from finance_data_center.snapshots import SnapshotStore, digest, payload_hash
    monkeypatch.setattr(config, 'FDC_ROOT', tmp_path/'center')
    client = client_for(monkeypatch, tmp_path)
    folder=tmp_path/'00000001';folder.mkdir()
    source=folder/'settlement_statement.parquet';path=folder/'settlement_statement.csv'
    frame=pl.DataFrame({'id':[1,2],'value':[1.25,None]});frame.write_parquet(source)
    content=frame.write_csv().encode();sha=digest(source)
    SnapshotStore(config.FDC_ROOT)._publish_file(source,sha)
    record={'serializer':f'polars-{pl.__version__}-csv-v1','parquet_sha256':sha,
            'csv_sha256':hashlib.sha256(content).hexdigest(),'empty_header':''}
    record['manifest_sha256']=payload_hash(record)
    export_descriptor(path).write_text(json.dumps(record))
    pl.DataFrame({'new':[999]}).write_parquet(source)
    response=client.get('/api/backtests/1/statement.csv')
    assert response.status_code==200
    assert response.content==content
