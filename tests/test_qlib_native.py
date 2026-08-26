from datetime import date, timedelta

import polars as pl
import pytest

from app.data.panel import PanelStore
from app.dsl.engine import parse
from app.qlib_native import (
    ALPHA158_FEATURES,
    QLIB_UPSTREAM_COMMIT,
    QlibDatasetSpec,
    QlibNativeRecorder,
    alpha158_catalog,
    cross_section_zscore,
    fillna,
    process_inf,
    resolve_qlib_task_integration,
    validate_alpha158_catalog,
)


def _series_frame(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({
        "trade_date": [date(2020, 1, 1) + timedelta(days=i) for i in range(len(values))],
        "ts_code": ["X"] * len(values),
        "close": values,
    })


def _factor_values(expression: str, values: list[float]) -> list[float | None]:
    frame = _series_frame(values)
    return (
        parse(expression, ["close"])
        .apply(frame.lazy())
        .collect()["factor"]
        .to_list()
    )


def test_alpha158_catalog_is_complete_unique_and_valid_for_both_markets():
    assert len(ALPHA158_FEATURES) == 158
    assert len({row.name for row in ALPHA158_FEATURES}) == 158
    assert ALPHA158_FEATURES[0].name == "KMID"
    assert ALPHA158_FEATURES[-1].name == "VSUMD60"
    assert QLIB_UPSTREAM_COMMIT == "79633dd9506ea689e5400dea0197717b5b3d74b7"
    for market in ("ashare", "us"):
        assert not [row for row in validate_alpha158_catalog(market) if row["error"]]
        catalog = alpha158_catalog(market)
        assert catalog["feature_count"] == 158
        assert sum(catalog["families"].values()) == 158


def test_qlib_regression_and_extreme_position_operators():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert _factor_values("ts_slope(close,5)", values)[-1] == pytest.approx(1.0)
    assert _factor_values("ts_rsquare(close,5)", values)[-1] == pytest.approx(1.0)
    assert _factor_values("ts_resi(close,5)", values)[-1] == pytest.approx(0.0)
    assert _factor_values("ts_argmax(close,5)", values)[-1] == pytest.approx(5.0)
    assert _factor_values("ts_argmin(close,5)", values)[-1] == pytest.approx(1.0)
    assert _factor_values("ts_quantile(close,5,0.8)", values)[-1] == pytest.approx(5.2)


def test_qlib_elementwise_compatibility_operators():
    frame = _series_frame([-2.0, 0.0, 3.0])
    result = parse(
        "maximum(close,0)+minimum(close,0)+gt(close,0)-lt(close,0)",
        ["close"],
    ).apply(frame.lazy()).collect()["factor"].to_list()
    assert result == pytest.approx([-3.0, 0.0, 4.0])


def test_panel_derives_adjusted_vwap_without_claiming_native_field(tmp_path):
    source = tmp_path / "panel.parquet"
    pl.DataFrame({
        "trade_date": [date(2020, 1, 2)],
        "ts_code": ["X.US"],
        "name": ["X"],
        "open": [20.0],
        "high": [22.0],
        "low": [19.0],
        "close": [21.0],
        "vol": [100.0],
        "amount": [1050.0],
        "adjustment_factor": [2.0],
    }).write_parquet(source)
    panel = PanelStore(
        str(source),
        "us",
        ["open", "high", "low", "close", "vwap", "vol", "amount"],
    ).ensure_loaded()
    assert panel["vwap"][0] == pytest.approx(21.0)


def test_qlib_processors_and_chronological_dataset_contract():
    frame = pl.DataFrame({
        "trade_date": [date(2020, 1, 1)] * 3,
        "x": [1.0, float("inf"), None],
    })
    cleaned = fillna(process_inf(frame, ["x"]), ["x"])
    assert cleaned["x"].to_list() == [1.0, 0.0, 0.0]
    normalized = cross_section_zscore(
        pl.DataFrame({
            "trade_date": [date(2020, 1, 1)] * 3,
            "x": [1.0, 2.0, 3.0],
        }),
        ["x"],
    )
    assert normalized["x"].mean() == pytest.approx(0.0)
    assert set(QlibDatasetSpec("us").resolved_segments()) == {
        "train", "valid", "test", "vault",
    }


def test_qlib_recorder_is_immutable(tmp_path):
    recorder = QlibNativeRecorder("run", tmp_path)
    target = recorder.write_json("signal", {"value": 1})
    assert target.exists()
    with pytest.raises(FileExistsError):
        recorder.write_json("signal", {"value": 2})


def test_task_integration_is_explicit_and_layer1_scoped():
    legacy = resolve_qlib_task_integration(None, layer1_enabled=True)
    assert legacy["enabled"] is False
    enabled = resolve_qlib_task_integration(
        {
            "enabled": True,
            "alpha158_prior_enabled": True,
            "gbdt_candidate_pool_enabled": True,
        },
        layer1_enabled=True,
    )
    assert enabled["effective"] is True
    assert enabled["alpha158_prior_enabled"] is True
    assert enabled["gbdt_candidate_pool_enabled"] is True
    assert enabled["joint_model_enabled"] is True
    assert enabled["residual_distillation_enabled"] is True
    assert enabled["adaptive_budget_enabled"] is True
    assert enabled["dynamic_trial_governance_enabled"] is True
    assert enabled["max_training_rows"] == 250_000
    assert enabled["min_meta_dates"] == 60
    disabled_layer = resolve_qlib_task_integration(True, layer1_enabled=False)
    assert disabled_layer["enabled"] is True
    assert disabled_layer["effective"] is False
    assert disabled_layer["joint_model_enabled"] is False
