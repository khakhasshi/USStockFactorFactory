import polars as pl
from polars.testing import assert_frame_equal
from backend.app import feature_cache


def test_production_cache_write_read_and_version(monkeypatch, tmp_path):
    monkeypatch.setattr(feature_cache, 'FDC_ROOT', tmp_path/'center')
    path=tmp_path/'matrix.parquet'
    frame=pl.DataFrame({'symbol':['A','B'],'value':[1.0,None]})
    first=feature_cache.write(path,frame)
    assert feature_cache.exists(path)
    assert not path.exists()
    assert_frame_equal(feature_cache.read(path),frame)
    changed=frame.with_columns(pl.lit(2.0).alias('value'))
    feature_cache.write(path,changed)
    assert_frame_equal(feature_cache.read(path),changed)
    assert_frame_equal(pl.from_arrow(feature_cache.store().read(path,version=first)),frame)
