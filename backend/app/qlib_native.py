"""Native Qlib research contracts and the complete Alpha158 feature catalog.

This module intentionally does not import ``pyqlib``.  FactorFactory keeps its
own causal panel, execution model, isolation layers, and acceptance gates while
adopting Qlib's useful feature/dataset/processor/recorder abstractions.  Alpha158
formulas are adapted from Microsoft Qlib under the MIT license; see
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import polars as pl

from .config import PROJECT_ROOT, get_layer_bounds
from .dsl.engine import validate

QLIB_NATIVE_SCHEMA = "factorfactory.qlib-native/v1"
ALPHA158_SCHEMA = "factorfactory.qlib-alpha158/v1"
QLIB_TASK_INTEGRATION_SCHEMA = "factorfactory.qlib-task-integration/v2"
QLIB_UPSTREAM_COMMIT = "79633dd9506ea689e5400dea0197717b5b3d74b7"
QLIB_UPSTREAM_SOURCE = (
    "https://github.com/microsoft/qlib/blob/"
    f"{QLIB_UPSTREAM_COMMIT}/qlib/contrib/data/loader.py"
)
ALPHA158_WINDOWS = (5, 10, 20, 30, 60)
ALPHA158_ARTIFACT_ROOT = PROJECT_ROOT / "var" / "reports" / "qlib-alpha158"


def resolve_qlib_task_integration(
    config: dict | bool | None,
    *,
    layer1_enabled: bool,
    alpha158_algorithm_selected: bool = False,
    joint_algorithm_selected: bool = False,
) -> dict:
    """Validate the task-scoped Qlib discovery contract.

    Historical tasks omit this block and remain unchanged.  Selecting the
    explicit Alpha158 search arm is itself an auditable opt-in.
    """
    if config is None:
        raw: dict = {}
    elif isinstance(config, bool):
        raw = {"enabled": config}
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raise ValueError("qlib_integration 必须为布尔值或配置对象")

    def flag(name: str, default: bool) -> bool:
        value = raw.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"qlib_integration.{name} 必须为布尔值")
        return value

    def integer(name: str, default: int, minimum: int, maximum: int) -> int:
        value = raw.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"qlib_integration.{name} 必须为整数")
        try:
            resolved = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"qlib_integration.{name} 必须为整数") from exc
        if not minimum <= resolved <= maximum:
            raise ValueError(
                f"qlib_integration.{name} 必须在 {minimum}..{maximum}"
            )
        return resolved

    def number(name: str, default: float, minimum: float, maximum: float) -> float:
        value = raw.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"qlib_integration.{name} 必须为数值")
        try:
            resolved = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"qlib_integration.{name} 必须为数值") from exc
        if not math.isfinite(resolved) or not minimum <= resolved <= maximum:
            raise ValueError(
                f"qlib_integration.{name} 必须在 {minimum}..{maximum}"
            )
        return resolved

    enabled = (
        flag("enabled", False)
        or alpha158_algorithm_selected
        or joint_algorithm_selected
    )
    alpha158_prior = (
        flag("alpha158_prior_enabled", enabled) or alpha158_algorithm_selected
    )
    gbdt_pool = flag("gbdt_candidate_pool_enabled", enabled)
    joint_model = flag("joint_model_enabled", enabled) or joint_algorithm_selected
    residual_distillation = flag("residual_distillation_enabled", joint_model)
    adaptive_budget = flag("adaptive_budget_enabled", enabled)
    trial_governance = flag("dynamic_trial_governance_enabled", enabled)
    effective = enabled and layer1_enabled
    return {
        "schema": QLIB_TASK_INTEGRATION_SCHEMA,
        "enabled": enabled,
        "effective": effective,
        "alpha158_prior_enabled": effective and alpha158_prior,
        "gbdt_candidate_pool_enabled": effective and gbdt_pool,
        "joint_model_enabled": effective and joint_model,
        "residual_distillation_enabled": (
            effective and joint_model and residual_distillation
        ),
        "adaptive_budget_enabled": effective and adaptive_budget,
        "dynamic_trial_governance_enabled": effective and trial_governance,
        "model_refresh_unique_evals": integer(
            "model_refresh_unique_evals", 100, 20, 100_000
        ),
        "max_training_rows": integer(
            "max_training_rows", 250_000, 5_000, 5_000_000
        ),
        "min_meta_dates": integer(
            "min_meta_dates", 60, 20, 500
        ),
        "structural_prior_share": number(
            "structural_prior_share", 0.10, 0.02, 0.30
        ),
        "include_low_fidelity_vwap": flag(
            "include_low_fidelity_vwap", False
        ),
        "processor_policy": "split_safe_ml_only_no_final_signal_imputation",
        "evaluation_authority": "factorfactory_v4_2_v4_3",
        "upstream_commit": QLIB_UPSTREAM_COMMIT,
        "provenance_recording": True,
    }


@dataclass(frozen=True)
class Alpha158Feature:
    name: str
    expression: str
    family: str
    window: int | None = None
    upstream_expression: str | None = None

    def payload(self, market: str | None = None) -> dict:
        fidelity = "exact_formula_native_fields"
        caveat = None
        if self.name == "VWAP0":
            fidelity = (
                "derived_turnover_volume_proxy"
                if market == "ashare"
                else "low_fidelity_close_times_volume_proxy"
                if market == "us"
                else "market_dependent_vwap_proxy"
            )
            caveat = (
                "A股以当日成交额/成交量并按OHLC复权因子构造；美股 amount "
                "常为 close×volume 代理，VWAP0 可能近似常数，不应据此宣称真实VWAP能力。"
            )
        return {
            **asdict(self),
            "market": market,
            "fidelity": fidelity,
            "caveat": caveat,
            "normalized_hash": hashlib.sha256(
                self.expression.encode("utf-8")
            ).hexdigest()[:16],
        }


def _base_features() -> list[Alpha158Feature]:
    rows = [
        ("KMID", "(close-open)/open", "kbar", "($close-$open)/$open"),
        ("KLEN", "(high-low)/open", "kbar", "($high-$low)/$open"),
        ("KMID2", "(close-open)/(high-low+1e-12)", "kbar", "($close-$open)/($high-$low+1e-12)"),
        ("KUP", "(high-maximum(open,close))/open", "kbar", "($high-Greater($open,$close))/$open"),
        ("KUP2", "(high-maximum(open,close))/(high-low+1e-12)", "kbar", "($high-Greater($open,$close))/($high-$low+1e-12)"),
        ("KLOW", "(minimum(open,close)-low)/open", "kbar", "(Less($open,$close)-$low)/$open"),
        ("KLOW2", "(minimum(open,close)-low)/(high-low+1e-12)", "kbar", "(Less($open,$close)-$low)/($high-$low+1e-12)"),
        ("KSFT", "(2*close-high-low)/open", "kbar", "(2*$close-$high-$low)/$open"),
        ("KSFT2", "(2*close-high-low)/(high-low+1e-12)", "kbar", "(2*$close-$high-$low)/($high-$low+1e-12)"),
        ("OPEN0", "open/close", "price", "$open/$close"),
        ("HIGH0", "high/close", "price", "$high/$close"),
        ("LOW0", "low/close", "price", "$low/$close"),
        ("VWAP0", "vwap/close", "price", "$vwap/$close"),
    ]
    return [
        Alpha158Feature(name, expression, family, upstream_expression=upstream)
        for name, expression, family, upstream in rows
    ]


def _rolling_feature_rows(window: int) -> list[Alpha158Feature]:
    w = int(window)
    lag_close = "delay(close,1)"
    lag_vol = "delay(vol,1)"
    price_delta = f"(close-{lag_close})"
    volume_delta = f"(vol-{lag_vol})"
    absolute_return_volume = f"abs(close/({lag_close}+1e-12)-1)*vol"
    formulas = [
        ("ROC", f"delay(close,{w})/close", "momentum"),
        ("MA", f"ts_mean(close,{w})/close", "trend"),
        ("STD", f"ts_std(close,{w})/close", "volatility"),
        ("BETA", f"ts_slope(close,{w})/close", "trend_geometry"),
        ("RSQR", f"ts_rsquare(close,{w})", "trend_geometry"),
        ("RESI", f"ts_resi(close,{w})/close", "trend_geometry"),
        ("MAX", f"ts_max(high,{w})/close", "price_location"),
        ("MIN", f"ts_min(low,{w})/close", "price_location"),
        ("QTLU", f"ts_quantile(close,{w},0.8)/close", "price_location"),
        ("QTLD", f"ts_quantile(close,{w},0.2)/close", "price_location"),
        ("RANK", f"ts_rank(close,{w})", "price_location"),
        ("RSV", f"(close-ts_min(low,{w}))/(ts_max(high,{w})-ts_min(low,{w})+1e-12)", "price_location"),
        ("IMAX", f"ts_argmax(high,{w})/{w}", "extreme_timing"),
        ("IMIN", f"ts_argmin(low,{w})/{w}", "extreme_timing"),
        ("IMXD", f"(ts_argmax(high,{w})-ts_argmin(low,{w}))/{w}", "extreme_timing"),
        ("CORR", f"ts_corr(close,log(vol+1),{w})", "price_volume"),
        ("CORD", f"ts_corr(close/({lag_close}+1e-12),log(vol/({lag_vol}+1e-12)+1),{w})", "price_volume"),
        ("CNTP", f"ts_mean(gt(close,{lag_close}),{w})", "direction_count"),
        ("CNTN", f"ts_mean(lt(close,{lag_close}),{w})", "direction_count"),
        ("CNTD", f"ts_mean(gt(close,{lag_close}),{w})-ts_mean(lt(close,{lag_close}),{w})", "direction_count"),
        ("SUMP", f"ts_sum(maximum({price_delta},0),{w})/(ts_sum(abs({price_delta}),{w})+1e-12)", "direction_magnitude"),
        ("SUMN", f"ts_sum(maximum(-({price_delta}),0),{w})/(ts_sum(abs({price_delta}),{w})+1e-12)", "direction_magnitude"),
        ("SUMD", f"(ts_sum(maximum({price_delta},0),{w})-ts_sum(maximum(-({price_delta}),0),{w}))/(ts_sum(abs({price_delta}),{w})+1e-12)", "direction_magnitude"),
        ("VMA", f"ts_mean(vol,{w})/(vol+1e-12)", "volume"),
        ("VSTD", f"ts_std(vol,{w})/(vol+1e-12)", "volume"),
        ("WVMA", f"ts_std({absolute_return_volume},{w})/(ts_mean({absolute_return_volume},{w})+1e-12)", "price_volume"),
        ("VSUMP", f"ts_sum(maximum({volume_delta},0),{w})/(ts_sum(abs({volume_delta}),{w})+1e-12)", "volume_direction"),
        ("VSUMN", f"ts_sum(maximum(-({volume_delta}),0),{w})/(ts_sum(abs({volume_delta}),{w})+1e-12)", "volume_direction"),
        ("VSUMD", f"(ts_sum(maximum({volume_delta},0),{w})-ts_sum(maximum(-({volume_delta}),0),{w}))/(ts_sum(abs({volume_delta}),{w})+1e-12)", "volume_direction"),
    ]
    return [
        Alpha158Feature(f"{prefix}{w}", expression, family, w)
        for prefix, expression, family in formulas
    ]


def alpha158_features() -> tuple[Alpha158Feature, ...]:
    rows = _base_features()
    for window in ALPHA158_WINDOWS:
        rows.extend(_rolling_feature_rows(window))
    if len(rows) != 158 or len({row.name for row in rows}) != 158:
        raise RuntimeError("Alpha158 catalog cardinality invariant failed")
    return tuple(rows)


ALPHA158_FEATURES = alpha158_features()


def alpha158_catalog(market: str | None = None) -> dict:
    if market not in {None, "ashare", "us"}:
        raise ValueError("market 必须是 ashare 或 us")
    families: dict[str, int] = {}
    records = []
    for feature in ALPHA158_FEATURES:
        families[feature.family] = families.get(feature.family, 0) + 1
        records.append(feature.payload(market))
    return {
        "schema": ALPHA158_SCHEMA,
        "upstream": {
            "project": "microsoft/qlib",
            "commit": QLIB_UPSTREAM_COMMIT,
            "source": QLIB_UPSTREAM_SOURCE,
            "license": "MIT",
        },
        "feature_count": len(records),
        "families": families,
        "windows": list(ALPHA158_WINDOWS),
        "market": market,
        "features": records,
        "compatibility_policy": {
            "window_warmup": "strict_full_window",
            "upstream_difference": (
                "Qlib rolling operators allow partial early windows; FactorFactory "
                "requires a complete window to reduce short-history/listing artifacts."
            ),
            "label": (
                "FactorFactory uses t-close signal, t+1 open entry, and configured "
                "open-to-open horizon return; Qlib's default Alpha158 label is kept "
                "as provenance only and is not used to bypass executable evaluation."
            ),
        },
    }


def validate_alpha158_catalog(market: str) -> list[dict]:
    from .config import get_dsl_fields

    fields = get_dsl_fields(market)
    return [
        {
            "name": feature.name,
            "expression": feature.expression,
            "error": validate(feature.expression, fields),
        }
        for feature in ALPHA158_FEATURES
    ]


@dataclass(frozen=True)
class QlibDatasetSpec:
    market: str
    universe_n: int = 500
    horizon: int = 5
    segments: dict[str, tuple[str, str]] | None = None

    def resolved_segments(self) -> dict[str, tuple[str, str]]:
        if self.segments:
            return dict(self.segments)
        bounds = get_layer_bounds(self.market)
        return {
            "train": bounds["INNER_PUBLIC"],
            "valid": bounds["META_TRAIN"],
            "test": bounds["META_HOLDOUT"],
            "vault": bounds["FACTOR_VAULT"],
        }

    def payload(self) -> dict:
        return {
            "market": self.market,
            "universe_n": self.universe_n,
            "horizon": self.horizon,
            "segments": {
                name: list(bounds)
                for name, bounds in self.resolved_segments().items()
            },
            "label": "t+1_open_to_t+1+h_open_return",
            "pit_policy": "NON_PIT_RESEARCH",
        }


def process_inf(frame: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    """Qlib ProcessInf analogue without fitting on future data."""
    return frame.with_columns(
        pl.when(pl.col(column).is_finite())
        .then(pl.col(column))
        .otherwise(None)
        .alias(column)
        for column in columns
    )


def cross_section_zscore(
    frame: pl.DataFrame,
    columns: Iterable[str],
    *,
    date_column: str = "trade_date",
    clip: float | None = 3.0,
) -> pl.DataFrame:
    """Qlib CSZScoreNorm analogue, calculated independently on each date."""
    expressions = []
    for column in columns:
        value = (
            (pl.col(column) - pl.col(column).mean().over(date_column))
            / (pl.col(column).std().over(date_column) + 1e-12)
        )
        if clip is not None:
            value = value.clip(-abs(float(clip)), abs(float(clip)))
        expressions.append(value.alias(column))
    return frame.with_columns(expressions)


def fillna(frame: pl.DataFrame, columns: Iterable[str], value: float = 0.0) -> pl.DataFrame:
    """Qlib Fillna analogue applied only after split-safe transforms."""
    return frame.with_columns(
        pl.col(column).fill_null(value).fill_nan(value).alias(column)
        for column in columns
    )


class QlibNativeRecorder:
    """Immutable JSON artifact recorder inspired by Qlib's workflow recorder."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        self.run_id = run_id
        self.root = (root or ALPHA158_ARTIFACT_ROOT) / run_id
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: dict | list) -> Path:
        target = self.root / f"{name}.json"
        if target.exists():
            raise FileExistsError(f"审计产物已存在，不允许覆盖: {target}")
        rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        target.write_text(rendered + "\n", encoding="utf-8")
        return target

    def manifest(self, artifacts: Iterable[Path], metadata: dict) -> dict:
        rows = []
        for path in artifacts:
            content = path.read_bytes()
            rows.append({
                "path": str(path.resolve()),
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
        return {
            "schema": QLIB_NATIVE_SCHEMA,
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "artifacts": rows,
        }


def latest_alpha158_reports(market: str) -> list[dict]:
    if market not in {"ashare", "us"}:
        raise ValueError("market 必须是 ashare 或 us")
    candidates = sorted(
        ALPHA158_ARTIFACT_ROOT.glob(f"*/{market}-*-summary.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    if not candidates:
        return []
    newest_run = candidates[0].parent
    results = []
    for candidate in sorted(newest_run.glob(f"{market}-*-summary.json")):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        payload["artifact_path"] = str(candidate.resolve())
        results.append(payload)
    return results


def alpha158_progress() -> dict:
    path = ALPHA158_ARTIFACT_ROOT / "latest-progress.json"
    if not path.exists():
        return {
            "schema": "factorfactory.qlib-alpha158-progress/v1",
            "state": "not_started",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact_path"] = str(path.resolve())
    return payload


def qlib_native_capabilities() -> dict:
    return {
        "schema": QLIB_NATIVE_SCHEMA,
        "upstream": alpha158_catalog()["upstream"],
        "absorbed": {
            "data_handler": "market-aware panel + derived VWAP provenance",
            "dataset": "chronological train/valid/test/vault segments",
            "processors": ["ProcessInf", "CSZScoreNorm", "Fillna", "DropnaLabel"],
            "feature_loader": "complete Alpha158 catalog mapped to native DSL",
            "model_interface": (
                "sample-level Alpha158 LightGBM with strictly chronological OOF, "
                "training-fitted processors and residual-to-DSL distillation"
            ),
            "recorders": ["SignalRecord", "SigAnaRecord", "PortAnaRecord", "immutable manifest"],
            "workflow": "multi-fidelity discovery then full Gate/HOLDOUT/Vault audit",
            "adaptive_budget": (
                "training-only unique gate improvement per CPU second with exploration floor"
            ),
            "dynamic_trial_governance": (
                "actual/effective trials plus AST and return-source redundancy"
            ),
        },
        "preserved_factorfactory_controls": [
            "two-sided training-only direction freeze",
            "realized turnover and market-specific costs",
            "HAC return confidence and lower bounds",
            "multiple-testing penalty",
            "independent HOLDOUT and Vault gates",
            "2020-to-latest frozen rating",
            "NON_PIT_RESEARCH label",
        ],
        "alpha158": {
            "feature_count": 158,
            "windows": list(ALPHA158_WINDOWS),
            "artifact_root": str(ALPHA158_ARTIFACT_ROOT.resolve()),
            "latest": {
                market: latest_alpha158_reports(market)
                for market in ("ashare", "us")
            },
        },
    }
