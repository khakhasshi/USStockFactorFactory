"""Evaluation Protocol V4.2 with Frozen Rating V4.3.

The mining loop is allowed to see only INNER_PUBLIC and META_TRAIN.  A full
audit is an explicit, persisted action that adds META_HOLDOUT and FACTOR_VAULT.
The protocol evaluates an executable portfolio, not IC in isolation:

* two-sided training-only direction selection with an explicit trials penalty
* direction-adjusted Rank IC and Newey-West significance
* realised target-weight turnover
* long-only absolute and benchmark-relative returns
* long/short leg attribution and borrow proxy
* return confidence bounds, cost breakeven and multiple-testing evidence
* cost stress, drawdown, monotonicity, era/year stability and capacity proxy
* a frozen full-history rating from 2020 through the latest available panel date
* independent HOLDOUT/Vault hard gates that remain separate from that rating

PIT is intentionally outside this score at the user's request.  Every result is
therefore labelled NON_PIT_RESEARCH; an F5 result means execution-ready under
this protocol, not production approval.
"""

from __future__ import annotations

import math
import time
import functools
import inspect
from datetime import date
from statistics import NormalDist

import polars as pl

from ..config import (
    DIRECTION_POLICY_BOTH,
    DIRECTION_POLICY_FIXED,
    EVALUATION_PROTOCOL_VERSION,
    FROZEN_RATING_PROTOCOL_VERSION,
    FROZEN_RATING_WINDOW_START,
    evaluation_config,
    get_dsl_fields,
    get_layer_bounds,
)
from ..data.panel import PanelStore
from ..dsl.engine import parse
from ..factors.return_path import build_return_path_signature
from .ranking import build_live_ranking

DISCOVERY_LAYERS = ["INNER_PUBLIC", "META_TRAIN"]
FULL_LAYERS = ["INNER_PUBLIC", "META_TRAIN", "META_HOLDOUT", "FACTOR_VAULT"]
FROZEN_RATING_LAYER = "FROZEN_RATING"
LAYER_ALIASES = {
    "INNER_PUBLIC": "public",
    "META_TRAIN": "gate",
    "META_HOLDOUT": "holdout",
    "FACTOR_VAULT": "vault",
}
SIGNAL_PREFLIGHT_PROTOCOL = "factorfactory.signal-preflight/v1"
AUDIT_REVISION = "v4.3-causal-event-20260913"


def _panel_frame(panel_glob, market):
    from ..audit_snapshot import read_frozen_panel
    store = PanelStore.get(panel_glob, market)
    # Small synthetic unit fixtures have no on-disk snapshot. They can test
    # arithmetic but can never supply formal provenance.
    return read_frozen_panel(store)[0] if hasattr(store, "read_snapshot") else store.ensure_loaded()


def _frozen_evaluation(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        from ..audit_snapshot import frozen_panel, digest_json
        from ..research_overfit import build_trial_return_evidence
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        with frozen_panel(params.get("panel_glob"), params.get("market", "us")) as snap:
            result = func(*args, **kwargs)
            raw = result.get("runtime", {}).pop("_raw_paths", None)
            if raw:
                evidence_cfg = evaluation_config(params["market"], params.get("evaluation_overrides"))
                evidence_cfg.pop("multiple_testing_trials", None)
                evidence_cfg["base_cost_bps"] = result["parameters"]["base_cost_bps"]
                context = {"market": params["market"], "portfolio_mode": params["portfolio_mode"],
                    "horizon": params["horizon"], "task_name": f"top{params['universe_n']}_{params['horizon']}d",
                    "panel_snapshot_id": snap.manifest["snapshot_id"],
                    "code_version": snap.manifest["code"]["code_sha256"],
                    "config_hash": digest_json({"universe_n": params["universe_n"], "horizon": params["horizon"],
                        "portfolio_mode": params["portfolio_mode"], "effective_config": evidence_cfg,
                        "isolation": get_layer_bounds(params["market"]),
                        "revision": AUDIT_REVISION})}
                result["raw_return_evidence"] = build_trial_return_evidence(paths=raw, context=context,
                    expected_directions=[p["direction"] for p in raw])
            result["audit_revision"] = AUDIT_REVISION
            result["input_snapshot_id"] = snap.manifest["snapshot_id"]
            return result
    return wrapped


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _series_stats(values: list[float], periods_per_year: float = 252.0) -> dict:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {
            "ann_return": None,
            "ann_vol": None,
            "sharpe": None,
            "sortino": None,
            "max_drawdown": None,
            "calmar": None,
            "win_rate": None,
            "tail_5pct": None,
            "final_nav": None,
        }
    n = len(clean)
    mean = sum(clean) / n
    variance = sum((x - mean) ** 2 for x in clean) / max(1, n - 1)
    std = math.sqrt(max(variance, 0.0))
    downside = [min(0.0, x) for x in clean]
    downside_rms = math.sqrt(sum(x * x for x in downside) / n)
    nav = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in clean:
        nav *= max(1e-9, 1.0 + value)
        peak = max(peak, nav)
        max_drawdown = max(max_drawdown, 1.0 - nav / peak)
    ann_return = nav ** (periods_per_year / n) - 1.0
    ann_vol = std * math.sqrt(periods_per_year)
    sharpe = mean / std * math.sqrt(periods_per_year) if std > 1e-12 else 0.0
    sortino = mean / downside_rms * math.sqrt(periods_per_year) if downside_rms > 1e-12 else 0.0
    ordered = sorted(clean)
    tail_index = max(0, min(n - 1, int(0.05 * (n - 1))))
    return {
        "ann_return": round(ann_return, 6),
        "ann_vol": round(ann_vol, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 6),
        "calmar": round(ann_return / max_drawdown, 4) if max_drawdown > 1e-12 else None,
        "win_rate": round(sum(x > 0 for x in clean) / n, 4),
        "tail_5pct": round(ordered[tail_index], 6),
        "final_nav": round(nav, 6),
    }


def _newey_west_mean(values: list[float], lag: int) -> dict:
    """HAC mean inference with a Bartlett kernel."""
    x = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(x)
    if n < 30:
        return {
            "t_stat": None,
            "p_value_two_sided": None,
            "standard_error": None,
            "mean": sum(x) / n if x else None,
            "n": n,
            "lag": min(max(0, lag), max(0, n - 1)),
        }
    mean = sum(x) / n
    residuals = [v - mean for v in x]
    gamma0 = sum(v * v for v in residuals) / n
    long_run_var = gamma0
    use_lag = min(max(0, lag), n - 1)
    for k in range(1, use_lag + 1):
        covariance = sum(residuals[t] * residuals[t - k] for t in range(k, n)) / n
        long_run_var += 2.0 * (1.0 - k / (use_lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run_var, 0.0) / n)
    if standard_error <= 1e-12:
        return {
            "t_stat": 0.0,
            "p_value_two_sided": 1.0,
            "standard_error": 0.0,
            "mean": mean,
            "n": n,
            "lag": use_lag,
        }
    t_stat = mean / standard_error
    p_value = 2.0 * (1.0 - NormalDist().cdf(abs(t_stat)))
    return {
        "t_stat": round(t_stat, 4),
        "p_value_two_sided": round(max(0.0, min(1.0, p_value)), 6),
        "standard_error": standard_error,
        "mean": mean,
        "n": n,
        "lag": use_lag,
    }


def _newey_west_t(values: list[float], lag: int) -> tuple[float | None, float | None]:
    """Compatibility wrapper returning t-stat and two-sided p-value."""
    result = _newey_west_mean(values, lag)
    return result["t_stat"], result["p_value_two_sided"]


def _probabilistic_sharpe_gt_zero(values: list[float]) -> float | None:
    """Probability that the period Sharpe is positive, adjusted for moments."""
    x = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(x)
    if n < 30:
        return None
    mean = sum(x) / n
    variance = sum((value - mean) ** 2 for value in x) / max(1, n - 1)
    std = math.sqrt(max(variance, 0.0))
    if std <= 1e-12:
        return 0.5
    centered = [(value - mean) / std for value in x]
    skew = sum(value**3 for value in centered) / n
    kurtosis = sum(value**4 for value in centered) / n
    sharpe = mean / std
    denominator = math.sqrt(
        max(
            1e-12,
            1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe,
        )
    )
    z_score = sharpe * math.sqrt(n - 1) / denominator
    return round(NormalDist().cdf(z_score), 6)


def _return_confidence(
    values: list[float],
    periods_per_year: float,
    cfg: dict,
) -> dict:
    """Conservative lower bounds for the mean and Sharpe of a return stream."""
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    lag = max(1, min(8, round(len(clean) ** (1.0 / 3.0)))) if clean else 1
    inference = _newey_west_mean(clean, lag)
    if not clean or inference["standard_error"] is None:
        return {
            "available": False,
            "hac_t_stat": inference["t_stat"],
            "hac_p_value": inference["p_value_two_sided"],
            "hac_lag": inference["lag"],
            "probabilistic_sharpe_gt_zero": None,
            "mean_lcb": None,
            "ann_return_lcb": None,
            "sharpe_lcb": None,
            "confidence_level": cfg["return_lcb_confidence"],
        }
    mean = sum(clean) / len(clean)
    variance = sum((value - mean) ** 2 for value in clean) / max(1, len(clean) - 1)
    std = math.sqrt(max(variance, 0.0))
    confidence_level = float(cfg["return_lcb_confidence"])
    z_score = NormalDist().inv_cdf(confidence_level)
    mean_lcb = mean - z_score * float(inference["standard_error"])
    sharpe_lcb = (
        mean_lcb / std * math.sqrt(periods_per_year)
        if std > 1e-12
        else 0.0
    )
    return {
        "available": True,
        "hac_t_stat": inference["t_stat"],
        "hac_p_value": inference["p_value_two_sided"],
        "hac_lag": inference["lag"],
        "probabilistic_sharpe_gt_zero": _probabilistic_sharpe_gt_zero(clean),
        "mean_lcb": round(mean_lcb, 8),
        # Arithmetic annualisation is intentionally conservative and avoids
        # exploding a noisy lower-bound estimate through compounding.
        "ann_return_lcb": round(mean_lcb * periods_per_year, 6),
        "sharpe_lcb": round(sharpe_lcb, 4),
        "confidence_level": confidence_level,
    }


def _grouped_performance(
    keys: list[int],
    returns: list[float],
    periods_per_year: float,
    key_name: str,
) -> list[dict]:
    grouped: dict[int, list[float]] = {}
    for key, value in zip(keys, returns):
        grouped.setdefault(int(key), []).append(float(value))
    rows = []
    for key in sorted(grouped):
        stats = _series_stats(grouped[key], periods_per_year)
        rows.append({
            key_name: key,
            "n_periods": len(grouped[key]),
            "ann_return": stats["ann_return"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
        })
    return rows


def _linear_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return 0.0
    return numerator / math.sqrt(vx * vy)


def _market_exposure(
    portfolio_returns: list[float],
    benchmark_returns: list[float],
    periods_per_year: float,
) -> dict:
    pairs = [
        (float(portfolio), float(benchmark))
        for portfolio, benchmark in zip(portfolio_returns, benchmark_returns)
        if math.isfinite(float(portfolio)) and math.isfinite(float(benchmark))
    ]
    if len(pairs) < 3:
        return {"beta": None, "correlation": None, "alpha_ann": None}
    portfolio = [row[0] for row in pairs]
    benchmark = [row[1] for row in pairs]
    portfolio_mean = sum(portfolio) / len(portfolio)
    benchmark_mean = sum(benchmark) / len(benchmark)
    covariance = sum(
        (p - portfolio_mean) * (b - benchmark_mean)
        for p, b in pairs
    )
    benchmark_variance = sum((b - benchmark_mean) ** 2 for b in benchmark)
    beta = covariance / benchmark_variance if benchmark_variance > 1e-12 else 0.0
    correlation = _linear_correlation(portfolio, benchmark)
    alpha_ann = (portfolio_mean - beta * benchmark_mean) * periods_per_year
    return {
        "beta": round(beta, 4),
        "correlation": round(correlation, 4) if correlation is not None else None,
        "alpha_ann": round(alpha_ann, 6),
    }


def _prepare_factor_base(
    expression: str,
    universe_n: int,
    horizon: int,
    panel_glob: str | None,
    market: str,
    layers: list[str],
    *,
    date_window: tuple[str, str | None] | None = None,
    layer_name_override: str | None = None,
) -> tuple[pl.LazyFrame, str]:
    """Build the direction-neutral part of one factor evaluation.

    This lazy sub-plan is shared by all tested orientations.  The explicit
    cache is consumed by one collect_all call, ensuring the DSL expression,
    cross-sectional factor rank, and forward-return rank are materialised only
    once even when both +1 and -1 are evaluated.
    """
    df = _panel_frame(panel_glob, market)
    fwd = f"fwd_{horizon}"
    if fwd not in df.columns:
        raise ValueError(f"不支持的 horizon: {horizon}")
    pipe = parse(expression, get_dsl_fields(market))
    # Compute rolling/cross-sectional features BEFORE evaluation slicing.
    # A cache is an optimizer barrier: later predicates must not truncate DSL
    # warm-up or change the cross-sectional rank universe.
    source = pipe.apply(df.lazy()).cache()
    label_exit = f"label_exit_date_{horizon}"
    calendar = df.select("trade_date", "layer").unique().sort("trade_date")
    if date_window is not None:
        if not layer_name_override:
            raise ValueError("日期窗口评估必须提供独立层名称")
        start, end = date_window
        source = source.filter(
            pl.col("trade_date") >= date.fromisoformat(start)
        )
        if end is not None:
            source = source.filter(
                pl.col("trade_date") <= date.fromisoformat(end)
            )
        source = source.with_columns(
            pl.lit(layer_name_override).alias("layer")
        )
        calendar = calendar.filter(pl.col("trade_date") >= date.fromisoformat(start))
        if end is not None:
            calendar = calendar.filter(pl.col("trade_date") <= date.fromisoformat(end))
        calendar = calendar.with_columns(pl.lit(layer_name_override).alias("layer"))
        if label_exit in df.columns:
            actual_end = date.fromisoformat(end) if end else df["trade_date"].max()
            source = source.filter(pl.col(label_exit) <= actual_end)
    else:
        source = source.filter(pl.col("layer").is_in(layers))
        calendar = calendar.filter(pl.col("layer").is_in(layers))
        if label_exit in df.columns:
            # Purge every label touching the next layer. Embargo the first h
            # market sessions after a boundary, using a factor-independent
            # calendar. The first training layer has no predecessor.
            bounds = get_layer_bounds(market)
            safe = pl.lit(False)
            for layer in layers:
                dates = calendar.filter(pl.col("layer") == layer)["trade_date"].to_list()
                skip = 0 if layer == "INNER_PUBLIC" else horizon
                if len(dates) <= skip:
                    continue
                safe = safe | ((pl.col("layer") == layer)
                    & (pl.col("trade_date") >= dates[skip])
                    & (pl.col(label_exit) <= date.fromisoformat(bounds[layer][1])))
            source = source.filter(safe)
    # Sampling cohorts do not change when factor values or future labels are
    # missing. Never rank the set of securities using future label availability.
    calendar = calendar.with_columns(pl.col("trade_date").rank(method="dense").over("layer").alias("_date_seq"))
    source = source.join(calendar.lazy(), on=["trade_date", "layer"], how="left")
    base = (
        source
        .filter(pl.col("univ_rank") <= universe_n)
        .with_columns(pl.len().over("trade_date").alias("_eligible_n"))
        .filter(pl.col("factor").is_finite())
        # A constant or numerically near-constant cross-section has no ranking
        # information.  Without this guard, deterministic tie ordering can
        # manufacture a portfolio from row order (the US Alpha158 VWAP0 proxy
        # exposed this failure mode).  Require economically non-trivial daily
        # dispersion before ranks, deciles, IC, or turnover are constructed.
        .with_columns(
            pl.col("factor").std().over("trade_date").alias("_factor_std"),
            pl.col("factor").abs().mean().over("trade_date").alias("_factor_abs_mean"),
        )
        .filter(
            pl.col("_factor_std")
            > (pl.col("_factor_abs_mean") + 1.0) * 1e-10
        )
        .with_columns(
            pl.col("factor").rank(method="average").over("trade_date").alias("_factor_rank"),
            pl.col(fwd).rank(method="average").over("trade_date").alias("_return_rank"),
        )
        .with_columns(
            (pl.col("_factor_rank") / pl.col("_factor_rank").count().over("trade_date")).alias("_factor_pct")
        )
        # fwd_h observations overlap on adjacent dates.  Portfolio statistics
        # therefore use one deterministic, non-overlapping rebalance cohort.
        # This also makes weight turnover an h-day rebalance turnover.
        .select(
            "trade_date",
            "layer",
            "era",
            "ts_code",
            fwd,
            "amount",
            "_eligible_n",
            "_return_rank",
            "_factor_pct",
            "_date_seq",
            "factor",
            (pl.col(label_exit) if label_exit in df.columns else pl.lit(None, dtype=pl.Date)).alias("_label_exit_date"),
            (pl.col("label_entry_date") if "label_entry_date" in df.columns else pl.lit(None, dtype=pl.Date)).alias("_label_entry_date"),
        )
        .cache()
    )
    return base, fwd


def _preflight_decision(
    daily: pl.DataFrame,
    *,
    min_finite_coverage: float = 0.02,
    min_usable_dates: int = 8,
    min_unique_values: int = 3,
) -> dict:
    eligible = int(daily["eligible_n"].sum()) if daily.height else 0
    finite = int(daily["finite_n"].sum()) if daily.height else 0
    coverage = finite / eligible if eligible else 0.0
    usable = daily.filter(
        (pl.col("finite_n") >= min_unique_values)
        & (pl.col("unique_n") >= min_unique_values)
        & pl.col("factor_std").is_finite()
        & (
            pl.col("factor_std")
            > (pl.col("factor_abs_mean").fill_null(0.0) + 1.0) * 1e-10
        )
    ) if daily.height else daily
    usable_rate = usable.height / daily.height if daily.height else 0.0
    reasons: list[str] = []
    if eligible == 0:
        reasons.append("廉价预筛：没有可评价样本")
    if coverage < min_finite_coverage:
        reasons.append("廉价预筛：有限值覆盖率不足")
    if usable.height < min_usable_dates:
        reasons.append("廉价预筛：非常数有效日期不足")
    return {
        "accepted": not reasons,
        "failure_reasons": reasons,
        "eligible_observations": eligible,
        "finite_observations": finite,
        "finite_coverage": round(coverage, 6),
        "observed_dates": daily.height,
        "usable_dates": usable.height,
        "usable_date_rate": round(usable_rate, 6),
        "thresholds": {
            "min_finite_coverage": min_finite_coverage,
            "min_usable_dates": min_usable_dates,
            "min_unique_values_per_date": min_unique_values,
        },
    }


def preflight_expression(
    expression: str,
    universe_n: int,
    horizon: int,
    panel_glob: str | None,
    market: str,
    *,
    sample_modulus: int = 8,
) -> dict:
    """Reject obvious null/constant/low-coverage signals before full V4.2.

    Sampling is deterministic by security and keeps every historical row for
    selected names, so rolling DSL operators retain their causal warm-up.  The
    result never contributes a score or an evaluation trial.
    """
    started = time.perf_counter()
    df = _panel_frame(panel_glob, market)
    fwd = f"fwd_{horizon}"
    if fwd not in df.columns:
        raise ValueError(f"不支持的 horizon: {horizon}")
    pipe = parse(expression, get_dsl_fields(market))
    modulus = max(1, int(sample_modulus))
    source = df.lazy()
    if modulus > 1:
        source = source.filter(
            (pl.col("ts_code").hash(seed=1729) % modulus) == 0
        )
    daily = (
        pipe.apply(source)
        .cache()
        .filter(pl.col("layer").is_in(DISCOVERY_LAYERS))
        .filter(pl.col("univ_rank") <= universe_n)
        .group_by("trade_date")
        .agg(
            pl.len().alias("eligible_n"),
            pl.col("factor").is_finite().sum().alias("finite_n"),
            pl.col("factor").filter(pl.col("factor").is_finite()).n_unique().alias("unique_n"),
            pl.col("factor").filter(pl.col("factor").is_finite()).std().alias("factor_std"),
            pl.col("factor").filter(pl.col("factor").is_finite()).abs().mean().alias("factor_abs_mean"),
        )
        .collect()
    )
    decision = _preflight_decision(daily)
    return {
        "schema": SIGNAL_PREFLIGHT_PROTOCOL,
        **decision,
        "sample_policy": {
            "security_hash_modulus": modulus,
            "layers": list(DISCOVERY_LAYERS),
            "rolling_history_preserved": True,
        },
        "evaluation_performed": False,
        "budget_charged": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def _prepare_direction_work(
    base: pl.LazyFrame,
    horizon: int,
    portfolio_mode: str,
    direction: int,
    cfg: dict,
) -> pl.LazyFrame:
    if portfolio_mode not in {"long_only", "long_short"}:
        raise ValueError("portfolio_mode 必须是 long_only 或 long_short")
    if direction not in {-1, 1}:
        raise ValueError("direction 必须在训练阶段冻结为 1 或 -1")
    top_fraction = float(cfg["top_fraction"])
    tail_fraction = float(cfg["tail_fraction"])
    target_capital = float(cfg["target_capital"])
    return (
        base.with_columns(
            (pl.col("factor") * direction).alias("_oriented_factor"),
            (
                pl.col("_factor_pct")
                if direction > 0
                else 1.0 - pl.col("_factor_pct")
            ).alias("_signal_pct")
        )
        .filter(((pl.col("_date_seq") - 1) % horizon) == 0)
        .with_columns(
            (pl.col("_signal_pct") >= 1.0 - top_fraction).alias("_is_long"),
            (pl.col("_signal_pct") <= tail_fraction).alias("_is_short"),
            (
                (pl.col("_signal_pct") * 10.0)
                .ceil()
                .clip(1, 10)
                .cast(pl.Int8)
            ).alias("_decile"),
        )
        .with_columns(
            pl.col("_is_long").sum().over("trade_date").alias("_n_long"),
            pl.col("_is_short").sum().over("trade_date").alias("_n_short"),
        )
        .with_columns(
            pl.when(pl.col("_is_long"))
            .then(1.0 / pl.col("_n_long"))
            .otherwise(0.0)
            .alias("_long_w"),
            pl.when(pl.col("_is_short"))
            .then(1.0 / pl.col("_n_short"))
            .otherwise(0.0)
            .alias("_short_w"),
        )
        .with_columns(
            (
                pl.col("_long_w")
                if portfolio_mode == "long_only"
                else pl.col("_long_w") - pl.col("_short_w")
            ).alias("_weight")
        )
        .with_columns(
            pl.col("_weight")
            .shift(1)
            .over("ts_code", order_by="trade_date")
            .fill_null(0.0)
            .alias("_previous_seen_weight"),
            pl.col("_date_seq")
            .shift(1)
            .over("ts_code", order_by="trade_date")
            .alias("_previous_seen_seq"),
        )
        .with_columns(
            pl.when(pl.col("_previous_seen_seq") == pl.col("_date_seq") - horizon)
            .then(pl.col("_previous_seen_weight"))
            .otherwise(0.0)
            .alias("_previous_weight")
        )
        .with_columns(
            (pl.col("_weight") - pl.col("_previous_weight"))
            .abs()
            .alias("_current_weight_change"),
            pl.col("_previous_weight").abs().alias("_matched_previous_gross"),
        )
        .with_columns(
            (
                pl.col("_current_weight_change")
                * target_capital
                / pl.when(pl.col("amount") > 0).then(pl.col("amount")).otherwise(None)
            ).alias("_adv_participation")
        )
        .cache()
    )


def _direction_aggregates(
    work: pl.LazyFrame,
    fwd: str,
    portfolio_mode: str,
    target_capital: float,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    target_gross = 1.0 if portfolio_mode == "long_only" else 2.0
    daily_lazy = (
        work.group_by("trade_date", "layer", "era")
        .agg(
            pl.corr("_signal_pct", "_return_rank").alias("ic"),
            pl.corr("_oriented_factor", fwd).alias("pearson_ic"),
            pl.col("_label_entry_date").min().alias("label_entry_date"),
            pl.col("_label_exit_date").max().alias("label_exit_date"),
            ((pl.col("_weight").abs() > 0) & ~pl.col(fwd).is_finite().fill_null(False)).sum().alias("missing_selected_labels"),
            pl.col(fwd).mean().alias("benchmark_return"),
            (pl.col("_long_w") * pl.col(fwd)).sum().alias("long_return"),
            (pl.col("_short_w") * pl.col(fwd)).sum().alias("short_return"),
            pl.col("_current_weight_change").sum().alias("_current_turnover"),
            pl.col("_matched_previous_gross").sum().alias("_matched_previous_gross"),
            pl.col("_adv_participation").quantile(0.95).alias("_current_adv_p95"),
            pl.col("amount").quantile(0.05).alias("_amount_p05"),
            (pl.col("_n_long").first() + (
                pl.col("_n_short").first() if portfolio_mode == "long_short" else pl.lit(0)
            )).alias("_target_positions"),
            pl.col("_date_seq").first().alias("_date_seq"),
            pl.len().alias("n"),
            pl.col("_eligible_n").first().alias("eligible_n"),
        )
        .with_columns(
            pl.when(pl.col("trade_date") == pl.col("trade_date").min().over("layer"))
            .then(0.0)
            .otherwise(
                (pl.lit(target_gross) - pl.col("_matched_previous_gross"))
                .clip(0.0, target_gross)
            )
            .alias("_exit_turnover")
        )
        .with_columns(
            (pl.col("_current_turnover") + pl.col("_exit_turnover")).alias("turnover"),
            (
                pl.col("_exit_turnover")
                * target_capital
                / (
                    pl.col("_target_positions").clip(1, None)
                    * pl.when(pl.col("_amount_p05") > 0)
                    .then(pl.col("_amount_p05"))
                    .otherwise(None)
                )
            ).alias("_exit_adv_proxy"),
        )
        .with_columns(
            pl.max_horizontal(
                pl.col("_current_adv_p95").fill_null(0.0),
                pl.col("_exit_adv_proxy").fill_null(0.0),
            ).alias("adv_participation_p95")
        )
        .with_columns((pl.col("n") / pl.col("eligible_n")).alias("coverage"))
        .filter(pl.col("n") >= 50)
        .with_columns(
            (
                pl.col("long_return")
                if portfolio_mode == "long_only"
                else pl.col("long_return") - pl.col("short_return")
            ).alias("gross_return")
        )
        .drop(
            "_current_turnover",
            "_matched_previous_gross",
            "_current_adv_p95",
            "_amount_p05",
            "_target_positions",
            "_date_seq",
            "_exit_turnover",
            "_exit_adv_proxy",
        )
        .sort("trade_date")
    )
    decile_lazy = (
        work.group_by("layer", "_decile")
        .agg(pl.col(fwd).mean().alias("mean_return"), pl.len().alias("observations"))
        .sort("layer", "_decile")
    )
    return daily_lazy, decile_lazy


def _collect_direction_frames(
    base: pl.LazyFrame,
    fwd: str,
    horizon: int,
    portfolio_mode: str,
    directions: list[int],
    cfg: dict,
) -> tuple[dict[int, tuple[pl.DataFrame, pl.DataFrame]], dict]:
    lazy_frames: list[pl.LazyFrame] = []
    for direction in directions:
        work = _prepare_direction_work(
            base,
            horizon,
            portfolio_mode,
            direction,
            cfg,
        )
        daily_lazy, decile_lazy = _direction_aggregates(
            work,
            fwd,
            portfolio_mode,
            float(cfg["target_capital"]),
        )
        lazy_frames.extend([daily_lazy, decile_lazy])

    # Direction-invariant, bounded signal-rank sketch for second-level dedupe.
    # The deterministic security hash prevents the daily mean percentile
    # (always about 0.5) from erasing cross-sectional identity.
    signal_sketch_lazy = (
        base.with_columns(
            pl.when((pl.col("ts_code").hash(seed=1729) % 2) == 0)
            .then(pl.col("_factor_pct") - 0.5)
            .otherwise(0.5 - pl.col("_factor_pct"))
            .alias("_signed_rank_sketch"),
            ((pl.col("_date_seq") - 1) % 32).cast(pl.Int16).alias("_sketch_bucket"),
        )
        .group_by("layer", "_sketch_bucket")
        .agg(pl.col("_signed_rank_sketch").mean().alias("value"))
        .sort("layer", "_sketch_bucket")
    )
    lazy_frames.append(signal_sketch_lazy)

    # collect_all performs common-subplan elimination across every orientation.
    # The direction-neutral cache above therefore executes exactly once.
    # Common-subplan and common-subexpression elimination are enabled in the
    # Polars default optimization set used by collect_all.
    frames = pl.collect_all(lazy_frames)
    output: dict[int, tuple[pl.DataFrame, pl.DataFrame]] = {}
    for index, direction in enumerate(directions):
        daily = frames[index * 2]
        deciles = frames[index * 2 + 1]
        if daily.height == 0:
            raise ValueError(
                "有效评估样本为空：表达式可能全为 null、常数或覆盖率不足"
            )
        output[direction] = (daily, deciles)
    sketch = frames[-1]
    signature = {
        "schema": "factorfactory.signal-rank-sketch/v1",
        "available": sketch.height >= 8,
        "direction_invariant_comparison": "absolute_correlation",
        "layers": sorted(sketch["layer"].unique().to_list()) if sketch.height else [],
        "buckets": 32,
        "vector": [round(float(value), 10) for value in sketch["value"].to_list()],
    }
    return output, signature


def _prepare_direction_batch(
    expression: str,
    universe_n: int,
    horizon: int,
    portfolio_mode: str,
    directions: list[int],
    panel_glob: str | None,
    market: str,
    layers: list[str],
    cfg: dict,
    *,
    date_window: tuple[str, str | None] | None = None,
    layer_name_override: str | None = None,
) -> tuple[dict[int, tuple[pl.DataFrame, pl.DataFrame]], dict]:
    started = time.perf_counter()
    base, fwd = _prepare_factor_base(
        expression,
        universe_n,
        horizon,
        panel_glob,
        market,
        layers,
        date_window=date_window,
        layer_name_override=layer_name_override,
    )
    planned = time.perf_counter()
    frames, signal_signature = _collect_direction_frames(
        base,
        fwd,
        horizon,
        portfolio_mode,
        directions,
        cfg,
    )
    finished = time.perf_counter()
    return frames, {
        "factor_plan_ms": round((planned - started) * 1000.0, 3),
        "factor_and_portfolio_ms": round((finished - planned) * 1000.0, 3),
        "total_ms": round((finished - started) * 1000.0, 3),
        "direction_count": len(directions),
        "factor_materializations": 1,
        "execution": "polars_native_shared_subplan",
        "signal_rank_signature": signal_signature,
    }


def _prepare_daily(
    expression: str,
    universe_n: int,
    horizon: int,
    portfolio_mode: str,
    direction: int,
    panel_glob: str | None,
    market: str,
    layers: list[str],
    cfg: dict,
    *,
    date_window: tuple[str, str | None] | None = None,
    layer_name_override: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    frames, _ = _prepare_direction_batch(
        expression,
        universe_n,
        horizon,
        portfolio_mode,
        [direction],
        panel_glob,
        market,
        layers,
        cfg,
        date_window=date_window,
        layer_name_override=layer_name_override,
    )
    return frames[direction]


def _layer_metrics(
    daily: pl.DataFrame,
    deciles: pl.DataFrame,
    layer: str,
    horizon: int,
    portfolio_mode: str,
    direction: int,
    cfg: dict,
) -> dict:
    sub = daily.filter(pl.col("layer") == layer).sort("trade_date")
    if sub.height == 0:
        return {
            "layer": layer,
            "n_days": 0,
            "available": False,
            "failure_reasons": ["该层没有有效样本"],
        }
    base_cost_bps = float(cfg["base_cost_bps"])
    borrow_bps = float(cfg["borrow_cost_bps_annual"]) if portfolio_mode == "long_short" else 0.0
    gross = [_safe_float(v) for v in sub["gross_return"].to_list()]
    benchmark = [_safe_float(v) for v in sub["benchmark_return"].to_list()]
    turnover = [_safe_float(v) for v in sub["turnover"].to_list()]
    transaction_cost = [value * base_cost_bps / 10_000.0 for value in turnover]
    borrow_period = borrow_bps / 10_000.0 * horizon / 252.0
    net = [g - c - borrow_period for g, c in zip(gross, transaction_cost)]
    active = (
        [g - b - c for g, b, c in zip(gross, benchmark, transaction_cost)]
        if portfolio_mode == "long_only"
        else list(net)
    )
    periods_per_year = 252.0 / horizon
    ranking_returns = active if portfolio_mode == "long_only" else net
    return_confidence = _return_confidence(ranking_returns, periods_per_year, cfg)
    absolute_return_confidence = (
        _return_confidence(net, periods_per_year, cfg)
        if portfolio_mode == "long_only"
        else return_confidence
    )
    pre_transaction_cost = (
        [g - b for g, b in zip(gross, benchmark)]
        if portfolio_mode == "long_only"
        else [g - borrow_period for g in gross]
    )
    mean_turnover = sum(turnover) / len(turnover)
    cost_breakeven_bps = (
        (sum(pre_transaction_cost) / len(pre_transaction_cost))
        / mean_turnover
        * 10_000.0
        if mean_turnover > 1e-12
        else None
    )
    cost_cushion_multiple = (
        cost_breakeven_bps / base_cost_bps
        if cost_breakeven_bps is not None and base_cost_bps > 0
        else None
    )
    ic = [float(v) for v in sub["ic"].to_list() if v is not None and math.isfinite(v)]
    ic_mean = sum(ic) / max(1, len(ic))
    ic_var = sum((value - ic_mean) ** 2 for value in ic) / max(1, len(ic) - 1)
    ic_std = math.sqrt(max(ic_var, 0.0))
    icir = ic_mean / ic_std * math.sqrt(252.0 / max(1, horizon)) if ic_std > 1e-12 else 0.0
    hac_t, hac_p = _newey_west_t(ic, max(1, horizon - 1))

    eras = (
        sub.group_by("era")
        .agg(
            pl.col("ic").mean().alias("ic"),
            pl.col("gross_return").mean().alias("gross_return"),
        )
        .sort("era")
    )
    era_series = [
        {
            "era": int(row["era"]),
            "ic": round(_safe_float(row["ic"]), 6),
            "gross_return": round(_safe_float(row["gross_return"]), 6),
        }
        for row in eras.iter_rows(named=True)
    ]
    era_consistency = sum(row["ic"] > 0 for row in era_series) / max(1, len(era_series))

    years = (
        sub.with_columns(pl.col("trade_date").dt.year().alias("_year"))
        .group_by("_year")
        .agg(
            pl.col("ic").mean().alias("ic"),
            pl.col("gross_return").mean().alias("gross_return"),
        )
        .sort("_year")
    )
    year_series = [
        {
            "year": int(row["_year"]),
            "ic": round(_safe_float(row["ic"]), 6),
            "gross_return": round(_safe_float(row["gross_return"]), 6),
        }
        for row in years.iter_rows(named=True)
    ]
    era_performance = _grouped_performance(
        [int(value) for value in sub["era"].to_list()],
        ranking_returns,
        periods_per_year,
        "era",
    )
    year_performance = _grouped_performance(
        [value.year for value in sub["trade_date"].to_list()],
        ranking_returns,
        periods_per_year,
        "year",
    )
    profitable_era_rate = (
        sum(_safe_float(row.get("ann_return"), -1.0) > 0 for row in era_performance)
        / len(era_performance)
        if era_performance
        else 0.0
    )
    profitable_year_rate = (
        sum(_safe_float(row.get("ann_return"), -1.0) > 0 for row in year_performance)
        / len(year_performance)
        if year_performance
        else 0.0
    )
    worst_era_sharpe = min(
        (_safe_float(row.get("sharpe"), -99.0) for row in era_performance),
        default=-99.0,
    )

    decile_sub = deciles.filter(pl.col("layer") == layer)
    decile_rows = [
        {
            "decile": int(row["_decile"]),
            "mean_return": round(_safe_float(row["mean_return"]), 8),
            "observations": int(row["observations"]),
        }
        for row in decile_sub.iter_rows(named=True)
    ]
    monotonicity = _linear_correlation(
        [float(row["decile"]) for row in decile_rows],
        [float(row["mean_return"]) for row in decile_rows],
    )
    top = next((row["mean_return"] for row in decile_rows if row["decile"] == 10), None)
    bottom = next((row["mean_return"] for row in decile_rows if row["decile"] == 1), None)

    stress: list[dict] = []
    stress_borrow = (
        float(cfg["stress_borrow_cost_bps_annual"]) / 10_000.0 * horizon / 252.0
        if portfolio_mode == "long_short"
        else 0.0
    )
    for stress_cost in cfg["stress_cost_bps"]:
        stressed = (
            [
                g - b - to * float(stress_cost) / 10_000.0
                for g, b, to in zip(gross, benchmark, turnover)
            ]
            if portfolio_mode == "long_only"
            else [
                g - to * float(stress_cost) / 10_000.0 - stress_borrow
                for g, to in zip(gross, turnover)
            ]
        )
        stats = _series_stats(stressed, 252.0 / horizon)
        stress.append({
            "cost_bps": float(stress_cost),
            "borrow_cost_bps_annual": float(cfg["stress_borrow_cost_bps_annual"])
            if portfolio_mode == "long_short"
            else 0.0,
            "ann_return": stats["ann_return"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
        })

    gross_stats = _series_stats(gross, periods_per_year)
    net_stats = _series_stats(net, periods_per_year)
    active_stats = _series_stats(active, periods_per_year)
    long_stats = _series_stats(
        [_safe_float(v) for v in sub["long_return"].to_list()], periods_per_year
    )
    short_leg_stats = _series_stats(
        [-_safe_float(v) for v in sub["short_return"].to_list()], periods_per_year
    )
    sorted_turnover = sorted(turnover)
    participation = [_safe_float(v) for v in sub["adv_participation_p95"].drop_nulls().to_list()]
    result = {
        "layer": layer,
        "available": True,
        "window_start": str(sub["trade_date"].min()),
        "window_end": str(sub["trade_date"].max()),
        "actual_label_entry_start": str(sub["label_entry_date"].min()) if "label_entry_date" in sub.columns else None,
        "actual_label_exit_end": str(sub["label_exit_date"].max()) if "label_exit_date" in sub.columns else None,
        "n_days": sub.height,
        "n_periods": sub.height,
        "calendar_equivalent_days": sub.height * horizon,
        "horizon": horizon,
        "observations": int(sub["n"].sum()),
        "coverage": round(_safe_float(sub["coverage"].mean()), 4),
        "min_daily_coverage": round(_safe_float(sub["coverage"].min()), 4),
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "icir": round(icir, 4),
        "ic_hit_rate": round(sum(value > 0 for value in ic) / max(1, len(ic)), 4),
        "hac_t_stat": hac_t,
        "hac_p_value": hac_p,
        "era_consistency": round(era_consistency, 4),
        "turnover": round(sum(turnover) / len(turnover), 6),
        "daily_turnover": round(sum(turnover) / len(turnover) / horizon, 6),
        "annualized_turnover": round(
            sum(turnover) / len(turnover) * 252.0 / horizon, 4
        ),
        "one_way_turnover": round(sum(turnover) / len(turnover) / 2.0, 6),
        "turnover_p95": round(sorted_turnover[int(0.95 * (len(sorted_turnover) - 1))], 6),
        "adv_participation_p95": round(max(participation), 6) if participation else None,
        "capacity_metric": "trade_notional_to_daily_amount_p95_proxy",
        "cost_breakeven_bps": (
            round(cost_breakeven_bps, 4)
            if cost_breakeven_bps is not None
            else None
        ),
        "cost_cushion_multiple": (
            round(cost_cushion_multiple, 4)
            if cost_cushion_multiple is not None
            else None
        ),
        "gross": gross_stats,
        "net": net_stats,
        "active": active_stats,
        "long_leg": long_stats,
        "short_leg": short_leg_stats if portfolio_mode == "long_short" else None,
        "long_only_sharpe": net_stats["sharpe"] if portfolio_mode == "long_only" else None,
        "portfolio_mode": portfolio_mode,
        "monotonicity": round(monotonicity, 4) if monotonicity is not None else None,
        "top_bottom_daily_spread": round(_safe_float(top) - _safe_float(bottom), 8)
        if top is not None and bottom is not None
        else None,
        "deciles": decile_rows,
        "era_series": era_series,
        "year_series": year_series,
        "era_performance": era_performance,
        "year_performance": year_performance,
        "profitable_era_rate": round(profitable_era_rate, 4),
        "profitable_year_rate": round(profitable_year_rate, 4),
        "worst_era_sharpe": round(worst_era_sharpe, 4),
        "cost_stress": stress,
        "base_cost_bps": base_cost_bps,
        "borrow_cost_bps_annual": borrow_bps,
        "market_exposure": _market_exposure(net, benchmark, periods_per_year),
        "return_confidence": return_confidence,
        "absolute_return_confidence": absolute_return_confidence,
        # Compressed and normalised; safe for training-layer correlation
        # checks without exposing a dated return series to either LLM.
        "return_path_signature": build_return_path_signature(
            ranking_returns
        ),
        # Compatibility fields used by the existing UI/search context.
        "direction": direction,
    }
    # Historical ic_* fields were Spearman statistics; retain explicitly
    # deprecated aliases for old miners, expose correctly named new metrics.
    pearson = [float(v) for v in sub["pearson_ic"].to_list() if v is not None and math.isfinite(v)] if "pearson_ic" in sub.columns else []
    pmean = sum(pearson) / len(pearson) if pearson else None
    pstd = math.sqrt(sum((v-pmean)**2 for v in pearson) / max(1, len(pearson)-1)) if pearson else None
    result.update({
        "rank_ic_mean": result["ic_mean"], "rank_ic_std": result["ic_std"],
        "rank_icir": result["icir"], "rank_ic_hit_rate": result["ic_hit_rate"],
        "pearson_ic_mean": pmean,
        "pearson_icir": pmean / pstd * math.sqrt(periods_per_year) if pstd and pstd > 1e-12 else None,
        "ic_metric_contract": {"ic_mean": "deprecated_alias_of_rank_ic_mean", "icir": "deprecated_alias_of_rank_icir", "pearson": "pearson_ic_mean", "rank": "rank_ic_mean"},
        "missing_selected_labels": int(sub["missing_selected_labels"].sum()) if "missing_selected_labels" in sub.columns else 0,
        "isolation_policy": {"revision": AUDIT_REVISION, "purge_label_end": True, "embargo_market_sessions": 0 if layer in {"INNER_PUBLIC", FROZEN_RATING_LAYER} else horizon, "rolling_warmup_preserved": True},
    })
    return result


def _relevant_sharpe(metrics: dict, portfolio_mode: str) -> float:
    branch = metrics.get("active") if portfolio_mode == "long_only" else metrics.get("net")
    return _safe_float((branch or {}).get("sharpe"))


def _worst_stress_sharpe(metrics: dict) -> float:
    values = [_safe_float(row.get("sharpe"), -99.0) for row in metrics.get("cost_stress", [])]
    return min(values) if values else -99.0


def _sigmoid_margin(value: float, center: float, scale: float) -> float:
    """Map a signed distance to a stable (0, 1) diagnostic quality.

    Unlike a hard max(0, x), this preserves ordering among failed candidates.
    It is never used to waive a gate failure.
    """
    denominator = max(1e-6, abs(scale))
    z_score = max(-30.0, min(30.0, (value - center) / denominator))
    return 1.0 / (1.0 + math.exp(-z_score))


def _discovery_score(public: dict, gate: dict, portfolio_mode: str, cfg: dict) -> dict:
    reasons: list[str] = []
    if any(m.get("missing_selected_labels", 0) for m in (public, gate)):
        reasons.append("训练选中股票存在缺失日历标签；向量收益不能作为完整证据")
    if not public.get("available") or not gate.get("available"):
        return {
            "score": 0.0,
            "learning_score": 0.0,
            "gate_score": 0.0,
            "score_semantics": "continuous_failure_margin_v4.2",
            "passed": False,
            "components": {},
            "gate_components": {},
            "failure_reasons": ["训练层不完整"],
        }
    coverage = min(_safe_float(public.get("coverage")), _safe_float(gate.get("coverage")))
    icir = min(_safe_float(public.get("icir")), _safe_float(gate.get("icir")))
    sharpe = min(_relevant_sharpe(public, portfolio_mode), _relevant_sharpe(gate, portfolio_mode))
    consistency = min(
        _safe_float(public.get("era_consistency")),
        _safe_float(gate.get("era_consistency")),
    )
    monotonicity = min(
        _safe_float(public.get("monotonicity")),
        _safe_float(gate.get("monotonicity")),
    )
    turnover = max(
        _safe_float(public.get("daily_turnover", public.get("turnover"))),
        _safe_float(gate.get("daily_turnover", gate.get("turnover"))),
    )
    stress = min(_worst_stress_sharpe(public), _worst_stress_sharpe(gate))
    return_t = min(
        _safe_float((public.get("return_confidence") or {}).get("hac_t_stat"), -99.0),
        _safe_float((gate.get("return_confidence") or {}).get("hac_t_stat"), -99.0),
    )
    sharpe_lcb = min(
        _safe_float((public.get("return_confidence") or {}).get("sharpe_lcb"), -99.0),
        _safe_float((gate.get("return_confidence") or {}).get("sharpe_lcb"), -99.0),
    )
    ann_return_lcb = min(
        _safe_float((public.get("return_confidence") or {}).get("ann_return_lcb"), -99.0),
        _safe_float((gate.get("return_confidence") or {}).get("ann_return_lcb"), -99.0),
    )
    profitable_era_rate = min(
        _safe_float(public.get("profitable_era_rate")),
        _safe_float(gate.get("profitable_era_rate")),
    )
    cost_cushion = min(
        _safe_float(public.get("cost_cushion_multiple"), -99.0),
        _safe_float(gate.get("cost_cushion_multiple"), -99.0),
    )
    hac_p = max(
        _safe_float(public.get("hac_p_value"), 1.0),
        _safe_float(gate.get("hac_p_value"), 1.0),
    )
    if coverage < float(cfg["min_coverage"]):
        reasons.append(f"覆盖率 {coverage:.1%} 低于 {float(cfg['min_coverage']):.0%}")
    if icir <= 0:
        reasons.append("训练层方向调整后 ICIR 非正")
    if hac_p > float(cfg["max_hac_p_value"]):
        reasons.append("训练层 HAC 显著性不足")
    if sharpe <= 0:
        reasons.append("训练层成本后组合 Sharpe 非正")
    if return_t < float(cfg["min_return_hac_t"]):
        reasons.append("训练层成本后收益的 HAC 置信度不足")
    if sharpe_lcb <= 0 or ann_return_lcb <= 0:
        reasons.append("训练层成本后收益下置信界非正")
    if consistency < float(cfg["min_era_consistency"]):
        reasons.append("跨 era 方向一致性不足")
    if profitable_era_rate < float(cfg["min_profitable_era_rate"]):
        reasons.append("跨 era 费后盈利比例不足")
    if monotonicity < float(cfg["min_monotonicity"]):
        reasons.append(
            "分位数组合单调性 "
            f"{monotonicity:.2f} 低于 {float(cfg['min_monotonicity']):.2f}"
        )
    if turnover > float(cfg["max_daily_turnover"]):
        reasons.append("实际持仓换手超过任务上限")
    if stress < float(cfg["min_stress_sharpe"]):
        reasons.append("保守成本压力下 Sharpe 为负")
    if cost_cushion < float(cfg["min_cost_cushion_multiple"]):
        reasons.append("成本盈亏平衡缓冲不足")

    trials = max(1, int(cfg["multiple_testing_trials"]))
    alpha = float(cfg["multiple_testing_alpha"])
    selection_hurdle = NormalDist().inv_cdf(1.0 - alpha / trials)
    public_sharpe = _relevant_sharpe(public, portfolio_mode)
    gate_sharpe = _relevant_sharpe(gate, portfolio_mode)
    sharpe_retention = min(1.0, max(0.0, gate_sharpe / max(0.25, public_sharpe)))
    gate_components = {
        "predictive": min(1.0, max(0.0, icir / 2.0)),
        "portfolio_lcb": min(1.0, max(0.0, sharpe_lcb / float(cfg["target_rank_sharpe"]))),
        "selection_confidence": min(1.0, max(0.0, return_t / selection_hurdle)),
        "stability": min(
            1.0,
            max(0.0, 0.5 * consistency / 0.75 + 0.5 * profitable_era_rate / 0.75),
        ),
        "generalization": sharpe_retention,
        "monotonicity": min(1.0, max(0.0, monotonicity)),
        "cost_survival": min(1.0, max(0.0, (stress + 0.5) / 1.5)),
        "implementability": (
            0.50
            * min(
                1.0,
                max(0.0, cost_cushion / float(cfg["target_cost_cushion_multiple"])),
            )
            + 0.30 * math.exp(-turnover / max(0.01, float(cfg["max_daily_turnover"])))
            + 0.20 * min(1.0, max(0.0, coverage))
        ),
    }
    weights = {
        "predictive": 0.10,
        "portfolio_lcb": 0.25,
        "selection_confidence": 0.15,
        "stability": 0.15,
        "generalization": 0.10,
        "monotonicity": 0.05,
        "cost_survival": 0.15,
        "implementability": 0.05,
    }
    gate_geometric = math.exp(
        sum(
            weights[key] * math.log(max(1e-6, value))
            for key, value in gate_components.items()
        )
    )
    gate_weakest = min(gate_components.values())
    gate_score = 5.0 * gate_geometric * (0.5 + 0.5 * gate_weakest)

    # Continuous learning components retain the severity and direction of
    # failure.  This gives the inner context and outer A/B loop a usable
    # gradient before any candidate crosses all hard research gates.
    public_sharpe = _relevant_sharpe(public, portfolio_mode)
    gate_sharpe = _relevant_sharpe(gate, portfolio_mode)
    sharpe_scale = max(0.5, abs(public_sharpe), abs(gate_sharpe))
    cross_layer_agreement = math.exp(
        -abs(public_sharpe - gate_sharpe) / sharpe_scale
    )
    sharpe_level_quality = _sigmoid_margin(sharpe, 0.0, 0.75)
    generalization_quality = math.sqrt(
        max(1e-6, sharpe_level_quality * cross_layer_agreement)
    )
    max_turnover = float(cfg["max_daily_turnover"])
    min_coverage = float(cfg["min_coverage"])
    cost_quality = _sigmoid_margin(
        cost_cushion,
        float(cfg["min_cost_cushion_multiple"]),
        max(0.75, float(cfg["target_cost_cushion_multiple"]) / 3.0),
    )
    turnover_quality = _sigmoid_margin(
        max_turnover - turnover,
        0.0,
        max(0.02, max_turnover * 0.20),
    )
    coverage_quality = _sigmoid_margin(
        coverage,
        min_coverage,
        0.10,
    )
    components = {
        "predictive": _sigmoid_margin(icir, 0.0, 0.50),
        "portfolio_lcb": _sigmoid_margin(
            sharpe_lcb,
            0.0,
            max(0.50, float(cfg["target_rank_sharpe"]) / 3.0),
        ),
        "selection_confidence": _sigmoid_margin(
            return_t,
            selection_hurdle,
            1.50,
        ),
        "stability": 0.50
        * _sigmoid_margin(
            consistency,
            float(cfg["min_era_consistency"]),
            0.20,
        )
        + 0.50
        * _sigmoid_margin(
            profitable_era_rate,
            float(cfg["min_profitable_era_rate"]),
            0.20,
        ),
        "generalization": generalization_quality,
        "monotonicity": _sigmoid_margin(
            monotonicity,
            float(cfg["min_monotonicity"]),
            0.25,
        ),
        "cost_survival": _sigmoid_margin(
            stress,
            float(cfg["min_stress_sharpe"]),
            0.75,
        ),
        "implementability": (
            0.50 * cost_quality
            + 0.30 * turnover_quality
            + 0.20 * coverage_quality
        ),
    }
    learning_geometric = math.exp(
        sum(
            weights[key] * math.log(max(1e-6, value))
            for key, value in components.items()
        )
    )
    weakest = min(components.values())
    score = 5.0 * learning_geometric * (0.5 + 0.5 * weakest)
    passed = (
        not reasons
        and gate_score >= float(cfg["min_research_score"])
    )
    return {
        "score": round(score, 4),
        "learning_score": round(score, 4),
        "gate_score": round(gate_score, 4),
        "score_semantics": "continuous_failure_margin_v4.2",
        "passed": passed,
        "components": {key: round(value, 4) for key, value in components.items()},
        "gate_components": {
            key: round(value, 4)
            for key, value in gate_components.items()
        },
        # These are the conservative PUBLIC/META_TRAIN aggregates that
        # actually drove the discovery decision.  They are safe to feed back
        # to the miner and prevent a single optimistic layer from being shown
        # as the reason for a score.
        "effective_metrics": {
            "coverage": round(coverage, 6),
            "icir": round(icir, 4),
            "portfolio_sharpe": round(sharpe, 4),
            "era_consistency": round(consistency, 4),
            "profitable_era_rate": round(profitable_era_rate, 4),
            "monotonicity": round(monotonicity, 4),
            "daily_turnover": round(turnover, 6),
            "worst_stress_sharpe": round(stress, 4),
            "return_hac_t": round(return_t, 4),
            "sharpe_lcb": round(sharpe_lcb, 4),
            "ann_return_lcb": round(ann_return_lcb, 6),
            "cost_cushion_multiple": round(cost_cushion, 4),
            "hac_p_value": round(hac_p, 6),
        },
        "selection_evidence": {
            "multiple_testing_trials": trials,
            "hurdle_t": round(selection_hurdle, 4),
            "worst_training_return_hac_t": round(return_t, 4),
            "worst_training_sharpe_lcb": round(sharpe_lcb, 4),
            "worst_training_ann_return_lcb": round(ann_return_lcb, 6),
            "worst_training_cost_cushion_multiple": round(cost_cushion, 4),
        },
        "failure_reasons": reasons,
    }


def _layer_gate(metrics: dict, portfolio_mode: str, cfg: dict, label: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not metrics.get("available"):
        return False, [f"{label}: 无有效样本"]
    if metrics.get("missing_selected_labels", 0):
        reasons.append(f"{label}: 所选股票缺少准确日历收益标签，不能将缺失视为零收益")
    horizon = max(1, int(metrics.get("horizon") or 1))
    min_periods = max(30, math.ceil(int(cfg["min_layer_days"]) / horizon))
    if int(metrics.get("n_periods") or metrics.get("n_days") or 0) < min_periods:
        reasons.append(f"{label}: 有效交易日不足")
    if _safe_float(metrics.get("coverage")) < float(cfg["min_coverage"]):
        reasons.append(f"{label}: 覆盖率不足")
    if _safe_float(metrics.get("icir")) <= 0:
        reasons.append(f"{label}: IC 方向反转")
    hac_p = metrics.get("hac_p_value")
    if hac_p is None or _safe_float(hac_p, 1.0) > float(cfg["max_hac_p_value"]):
        reasons.append(f"{label}: HAC 显著性不足")
    if _relevant_sharpe(metrics, portfolio_mode) < float(cfg["min_oos_sharpe"]):
        reasons.append(f"{label}: 成本后 Sharpe 未达标")
    confidence = metrics.get("return_confidence") or {}
    if _safe_float(confidence.get("hac_t_stat"), -99.0) < float(cfg["min_return_hac_t"]):
        reasons.append(f"{label}: 成本后收益 HAC 置信度不足")
    if _safe_float(confidence.get("sharpe_lcb"), -99.0) <= 0:
        reasons.append(f"{label}: 成本后 Sharpe 下置信界非正")
    if _safe_float(confidence.get("ann_return_lcb"), -99.0) <= 0:
        reasons.append(f"{label}: 成本后年化收益下置信界非正")
    relevant = metrics.get("active") if portfolio_mode == "long_only" else metrics.get("net")
    if _safe_float((relevant or {}).get("ann_return")) <= 0:
        reasons.append(f"{label}: 成本后年化收益非正")
    if _safe_float((metrics.get("net") or {}).get("max_drawdown"), 1.0) > float(cfg["max_drawdown"]):
        reasons.append(f"{label}: 绝对组合最大回撤超限")
    if portfolio_mode == "long_only" and _safe_float(
        (metrics.get("active") or {}).get("max_drawdown"), 1.0
    ) > float(cfg["max_drawdown"]):
        reasons.append(f"{label}: 相对基准最大回撤超限")
    if _safe_float(metrics.get("daily_turnover", metrics.get("turnover"))) > float(
        cfg["max_daily_turnover"]
    ):
        reasons.append(f"{label}: 日均等效换手超限")
    if portfolio_mode == "long_short" and abs(_safe_float(
        (metrics.get("market_exposure") or {}).get("beta")
    )) > float(cfg["max_market_beta_long_short"]):
        reasons.append(f"{label}: 多空组合市场 Beta 超限")
    if _safe_float(metrics.get("era_consistency")) < float(cfg["min_era_consistency"]):
        reasons.append(f"{label}: era 稳定性不足")
    if _safe_float(metrics.get("profitable_era_rate")) < float(
        cfg["min_profitable_era_rate"]
    ):
        reasons.append(f"{label}: era 费后盈利比例不足")
    if _safe_float(metrics.get("monotonicity")) < float(cfg["min_monotonicity"]):
        reasons.append(f"{label}: 分层单调性不足")
    if _worst_stress_sharpe(metrics) < float(cfg["min_stress_sharpe"]):
        reasons.append(f"{label}: 压力成本下失效")
    if _safe_float(metrics.get("cost_cushion_multiple"), -99.0) < float(
        cfg["min_cost_cushion_multiple"]
    ):
        reasons.append(f"{label}: 成本盈亏平衡缓冲不足")
    return not reasons, reasons


def _eligibility(layers: dict, discovery: dict, portfolio_mode: str, cfg: dict) -> dict:
    research_pass = bool(discovery.get("passed"))
    holdout_pass, holdout_reasons = _layer_gate(
        layers.get("holdout", {}), portfolio_mode, cfg, "HOLDOUT"
    )
    vault_pass, vault_reasons = _layer_gate(
        layers.get("vault", {}), portfolio_mode, cfg, "VAULT"
    )
    capacity_values = [
        _safe_float(layers.get(name, {}).get("adv_participation_p95"), 99.0)
        for name in ("holdout", "vault")
    ]
    capacity_pass = max(capacity_values) <= float(cfg["max_adv_participation"])
    reasons = list(discovery.get("failure_reasons", []))
    if research_pass:
        reasons.extend(holdout_reasons)
        if holdout_pass:
            reasons.extend(vault_reasons)
    if research_pass and holdout_pass and vault_pass and not capacity_pass:
        reasons.append("目标资金规模的 ADV 参与率超过上限")

    if not research_pass:
        grade, stage = "F1", "discovery_only"
    elif not holdout_pass:
        grade, stage = "F2", "research_pass"
    elif not vault_pass:
        grade, stage = "F3", "oos_pass"
    elif not capacity_pass:
        grade, stage = "F4", "paper_candidate"
    else:
        grade, stage = "F5", "live_candidate_non_pit"
    return {
        "grade": grade,
        "stage": stage,
        "research_pass": research_pass,
        "holdout_pass": holdout_pass,
        "vault_pass": vault_pass,
        "capacity_pass": capacity_pass,
        "failure_reasons": reasons,
        "production_approved": False,
        "policy_label": "NON_PIT_RESEARCH",
        "meaning": "F5 仅表示通过 V4 非 PIT 实战审计，不等于生产批准",
    }


def _evaluate_layers(
    expression: str,
    universe_n: int,
    horizon: int,
    portfolio_mode: str,
    direction: int,
    panel_glob: str | None,
    cost_bps: float | None,
    market: str,
    layers: list[str],
    evaluation_overrides: dict | None,
) -> tuple[dict, dict]:
    cfg = evaluation_config(market, evaluation_overrides)
    if cost_bps is not None:
        cfg["base_cost_bps"] = float(cost_bps)
        if float(cost_bps) not in cfg["stress_cost_bps"]:
            cfg["stress_cost_bps"] = sorted({*cfg["stress_cost_bps"], float(cost_bps)})
    daily, deciles = _prepare_daily(
        expression,
        universe_n,
        horizon,
        portfolio_mode,
        direction,
        panel_glob,
        market,
        layers,
        cfg,
    )
    metrics = {
        LAYER_ALIASES[layer]: _layer_metrics(
            daily,
            deciles,
            layer,
            horizon,
            portfolio_mode,
            direction,
            cfg,
        )
        for layer in layers
    }
    return metrics, cfg


def _evaluate_frozen_rating(
    expression: str,
    universe_n: int,
    horizon: int,
    portfolio_mode: str,
    direction: int,
    panel_glob: str | None,
    cost_bps: float | None,
    market: str,
    evaluation_overrides: dict | None,
) -> tuple[dict, dict]:
    """Evaluate the frozen direction from 2020 through the panel's latest date.

    This view deliberately spans multiple isolation layers, so it is a
    full-history rating rather than independent out-of-sample evidence.  It is
    only called by the explicit full-audit path and is never exposed to either
    proposal LLM.
    """
    cfg = evaluation_config(market, evaluation_overrides)
    if cost_bps is not None:
        cfg["base_cost_bps"] = float(cost_bps)
        if float(cost_bps) not in cfg["stress_cost_bps"]:
            cfg["stress_cost_bps"] = sorted({
                *cfg["stress_cost_bps"],
                float(cost_bps),
            })
    daily, deciles = _prepare_daily(
        expression,
        universe_n,
        horizon,
        portfolio_mode,
        direction,
        panel_glob,
        market,
        FULL_LAYERS,
        cfg,
        date_window=(FROZEN_RATING_WINDOW_START, None),
        layer_name_override=FROZEN_RATING_LAYER,
    )
    metrics = _layer_metrics(
        daily,
        deciles,
        FROZEN_RATING_LAYER,
        horizon,
        portfolio_mode,
        direction,
        cfg,
    )
    evaluated_start = metrics.get("window_start")
    evaluated_end = metrics.get("window_end")
    panel = _panel_frame(panel_glob, market)
    panel_latest = panel["trade_date"].max()
    metrics.update({
        "rating_protocol_version": FROZEN_RATING_PROTOCOL_VERSION,
        "window_start": FROZEN_RATING_WINDOW_START,
        "window_end": str(panel_latest) if panel_latest is not None else None,
        "evaluated_signal_start": evaluated_start,
        "evaluated_signal_end": evaluated_end,
        "window_policy": "2020_to_latest_available",
        "independent_out_of_sample": False,
        "visible_to_research_llms": False,
        "source_scope": (
            "all panel observations from 2020-01-01 through the latest "
            "available date, independent of isolation-layer labels"
        ),
        "overlaps_isolation_layers": [
            "META_TRAIN",
            "META_HOLDOUT",
            "FACTOR_VAULT",
            "post_declared_vault_extension_if_present",
        ],
    })
    return metrics, cfg


def _validate_direction_policy(direction_policy: str) -> str:
    if direction_policy not in {
        DIRECTION_POLICY_BOTH,
        DIRECTION_POLICY_FIXED,
    }:
        raise ValueError(
            "direction_policy 必须是 both_train_select 或 fixed"
        )
    return direction_policy


def _orientation_summary(direction: int, discovery: dict) -> dict:
    """Bounded training-safe evidence retained for direction attribution."""
    return {
        "direction": direction,
        "learning_score": discovery.get("learning_score", discovery.get("score", 0.0)),
        "gate_score": discovery.get("gate_score", 0.0),
        "passed": bool(discovery.get("passed")),
        "components": dict(discovery.get("components") or {}),
        "gate_components": dict(discovery.get("gate_components") or {}),
        "effective_metrics": dict(discovery.get("effective_metrics") or {}),
        "failure_reasons": list(discovery.get("failure_reasons") or []),
    }


def _evaluate_discovery_orientations(
    expression: str,
    universe_n: int,
    horizon: int,
    portfolio_mode: str,
    preferred_direction: int,
    panel_glob: str | None,
    cost_bps: float | None,
    market: str,
    evaluation_overrides: dict | None,
    direction_policy: str,
) -> tuple[dict, dict, dict, dict]:
    """Select direction using training-safe layers only.

    The effective multiple-testing budget is doubled when both signs are
    searched.  The selected sign is returned before any validation layer is
    accessed, so later audit code cannot use out-of-sample results to flip it.
    """
    if preferred_direction not in {-1, 1}:
        raise ValueError("direction 必须为 1 或 -1")
    policy = _validate_direction_policy(direction_policy)
    base_cfg = evaluation_config(market, evaluation_overrides)
    trials_multiplier = 2 if policy == DIRECTION_POLICY_BOTH else 1
    effective_overrides = {
        **(evaluation_overrides or {}),
        "multiple_testing_trials": max(
            1,
            int(base_cfg["multiple_testing_trials"]) * trials_multiplier,
        ),
    }
    directions = (
        [1, -1]
        if policy == DIRECTION_POLICY_BOTH
        else [preferred_direction]
    )
    cfg = evaluation_config(market, effective_overrides)
    if cost_bps is not None:
        cfg["base_cost_bps"] = float(cost_bps)
        if float(cost_bps) not in cfg["stress_cost_bps"]:
            cfg["stress_cost_bps"] = sorted({
                *cfg["stress_cost_bps"],
                float(cost_bps),
            })
    direction_frames, runtime = _prepare_direction_batch(
        expression,
        universe_n,
        horizon,
        portfolio_mode,
        directions,
        panel_glob,
        market,
        DISCOVERY_LAYERS,
        cfg,
    )
    metrics_started = time.perf_counter()
    candidates: list[dict] = []
    for candidate_direction in directions:
        daily, deciles = direction_frames[candidate_direction]
        layers = {
            LAYER_ALIASES[layer]: _layer_metrics(
                daily,
                deciles,
                layer,
                horizon,
                portfolio_mode,
                candidate_direction,
                cfg,
            )
            for layer in DISCOVERY_LAYERS
        }
        discovery = _discovery_score(
            layers["public"],
            layers["gate"],
            portfolio_mode,
            cfg,
        )
        layers["public"]["score"] = discovery["score"]
        layers["gate"]["score"] = discovery["score"]
        candidates.append({
            "direction": candidate_direction,
            "layers": layers,
            "cfg": cfg,
            "discovery": discovery,
        })

    selected = max(
        candidates,
        key=lambda row: (
            bool(row["discovery"].get("passed")),
            _safe_float(
                row["discovery"].get(
                    "learning_score",
                    row["discovery"].get("score"),
                )
            ),
            _safe_float(row["discovery"].get("gate_score")),
            row["direction"] == preferred_direction,
        ),
    )
    selected_direction = int(selected["direction"])
    selection = {
        "policy": policy,
        "selection_scope": "training_safe_discovery_only",
        "selection_rule": (
            "passed_then_learning_score_then_gate_score_"
            "then_preferred_direction_tiebreak"
        ),
        "preferred_direction": preferred_direction,
        "selected_direction": selected_direction,
        "frozen_for_downstream": True,
        "trials_multiplier": trials_multiplier,
        "base_multiple_testing_trials": int(
            base_cfg["multiple_testing_trials"]
        ),
        "effective_multiple_testing_trials": int(
            selected["cfg"]["multiple_testing_trials"]
        ),
        "candidates": {
            f"{row['direction']:+d}": _orientation_summary(
                int(row["direction"]),
                row["discovery"],
            )
            for row in candidates
        },
    }
    discovery = {
        **selected["discovery"],
        "direction_policy": policy,
        "preferred_direction": preferred_direction,
        "selected_direction": selected_direction,
        "direction_selection": selection,
    }
    metrics_finished = time.perf_counter()
    runtime = {
        **runtime,
        "_raw_paths": [{"direction": d,
            "dates": [str(v) for v in frames[0]["trade_date"].to_list()],
            "layers": frames[0]["layer"].to_list(),
            "returns": [float(g) - float(t) * float(cfg["base_cost_bps"]) / 10000.0
                - (float(cfg["borrow_cost_bps_annual"]) * horizon / 252.0 / 10000.0 if portfolio_mode == "long_short" else 0)
                for g, t in zip(frames[0]["gross_return"].to_list(), frames[0]["turnover"].to_list())]}
            for d, frames in direction_frames.items()],
        "metrics_ms": round(
            (metrics_finished - metrics_started) * 1000.0,
            3,
        ),
        "total_ms": round(
            runtime["total_ms"]
            + (metrics_finished - metrics_started) * 1000.0,
            3,
        ),
    }
    return selected["layers"], selected["cfg"], discovery, runtime


@_frozen_evaluation
def evaluate(
    expression: str,
    universe_n: int = 500,
    horizon: int = 5,
    portfolio_mode: str = "long_short",
    direction: int = 1,
    panel_glob: str | None = None,
    cost_bps: float | None = None,
    market: str = "us",
    evaluation_overrides: dict | None = None,
    direction_policy: str = DIRECTION_POLICY_BOTH,
) -> dict:
    """Mining-safe discovery evaluation with training-only sign selection."""
    layers, cfg, discovery, runtime = _evaluate_discovery_orientations(
        expression,
        universe_n,
        horizon,
        portfolio_mode,
        direction,
        panel_glob,
        cost_bps,
        market,
        evaluation_overrides,
        direction_policy,
    )
    selected_direction = int(discovery["selected_direction"])
    return {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "scope": "discovery",
        "policy_label": "NON_PIT_RESEARCH",
        "market": market,
        "portfolio_mode": portfolio_mode,
        "direction": selected_direction,
        "preferred_direction": direction,
        "direction_policy": direction_policy,
        "parameters": {
            "universe_n": universe_n,
            "horizon": horizon,
            "base_cost_bps": cfg["base_cost_bps"],
            "stress_cost_bps": cfg["stress_cost_bps"],
            "target_capital": cfg["target_capital"],
            "multiple_testing_trials": cfg["multiple_testing_trials"],
            "direction_trials_multiplier": (
                discovery["direction_selection"]["trials_multiplier"]
            ),
        },
        "public": layers["public"],
        "gate": layers["gate"],
        "discovery": discovery,
        "runtime": runtime,
        "ranking_policy": (
            "连续学习分仅供 Miner；硬门槛独立；方向只在训练安全层选择并冻结"
        ),
    }


@_frozen_evaluation
def _evaluate_full_vector(
    expression: str,
    universe_n: int = 500,
    horizon: int = 5,
    portfolio_mode: str = "long_short",
    direction: int = 1,
    panel_glob: str | None = None,
    cost_bps: float | None = None,
    market: str = "us",
    evaluation_overrides: dict | None = None,
    direction_policy: str = DIRECTION_POLICY_BOTH,
) -> dict:
    """Explicit audit with four isolation layers plus a full-history rating."""
    full_started = time.perf_counter()
    discovery_layers, cfg, discovery, discovery_runtime = (
        _evaluate_discovery_orientations(
            expression,
            universe_n,
            horizon,
            portfolio_mode,
            direction,
            panel_glob,
            cost_bps,
            market,
            evaluation_overrides,
            direction_policy,
        )
    )
    selected_direction = int(discovery["selected_direction"])
    validation_started = time.perf_counter()
    validation_layers, _ = _evaluate_layers(
        expression,
        universe_n,
        horizon,
        portfolio_mode,
        selected_direction,
        panel_glob,
        cost_bps,
        market,
        ["META_HOLDOUT", "FACTOR_VAULT"],
        cfg,
    )
    validation_finished = time.perf_counter()
    rating_started = time.perf_counter()
    frozen_rating, _ = _evaluate_frozen_rating(
        expression,
        universe_n,
        horizon,
        portfolio_mode,
        selected_direction,
        panel_glob,
        cost_bps,
        market,
        cfg,
    )
    rating_finished = time.perf_counter()
    layers = {
        **discovery_layers,
        **validation_layers,
        "rating": frozen_rating,
    }
    eligibility = _eligibility(layers, discovery, portfolio_mode, cfg)
    ranking = build_live_ranking(layers, eligibility, portfolio_mode, cfg)
    return {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "rating_protocol_version": FROZEN_RATING_PROTOCOL_VERSION,
        "scope": "full_audit",
        "policy_label": "NON_PIT_RESEARCH",
        "market": market,
        "portfolio_mode": portfolio_mode,
        "direction": selected_direction,
        "preferred_direction": direction,
        "direction_policy": direction_policy,
        "parameters": {
            "universe_n": universe_n,
            "horizon": horizon,
            "base_cost_bps": cfg["base_cost_bps"],
            "stress_cost_bps": cfg["stress_cost_bps"],
            "borrow_cost_bps_annual": cfg["borrow_cost_bps_annual"],
            "target_capital": cfg["target_capital"],
            "max_adv_participation": cfg["max_adv_participation"],
            "multiple_testing_trials": cfg["multiple_testing_trials"],
            "multiple_testing_alpha": cfg["multiple_testing_alpha"],
            "return_lcb_confidence": cfg["return_lcb_confidence"],
            "direction_trials_multiplier": (
                discovery["direction_selection"]["trials_multiplier"]
            ),
            "frozen_rating_window_start": FROZEN_RATING_WINDOW_START,
            "frozen_rating_window_end": frozen_rating.get("window_end"),
            "frozen_rating_window_policy": "latest_available",
        },
        "layers": layers,
        "public": layers["public"],
        "gate": layers["gate"],
        "holdout": layers["holdout"],
        "vault": layers["vault"],
        "rating": layers["rating"],
        "discovery": discovery,
        "eligibility": eligibility,
        "ranking": ranking,
        "runtime": {
            "_raw_paths": discovery_runtime.pop("_raw_paths", []),
            "discovery": discovery_runtime,
            "validation_ms": round(
                (validation_finished - validation_started) * 1000.0,
                3,
            ),
            "rating_ms": round(
                (rating_finished - rating_started) * 1000.0,
                3,
            ),
            "total_ms": round(
                (time.perf_counter() - full_started) * 1000.0,
                3,
            ),
        },
    }


def evaluate_full(
    expression: str, universe_n: int = 500, horizon: int = 5,
    portfolio_mode: str = "long_short", direction: int = 1,
    panel_glob: str | None = None, cost_bps: float | None = None,
    market: str = "us", evaluation_overrides: dict | None = None,
    direction_policy: str = DIRECTION_POLICY_BOTH, *,
    trial_history: list | None = None, audit_artifact_dir=None,
) -> dict:
    """Full V4.3 audit. Vector rank is diagnostic, never a promotion bypass."""
    from ..audit_snapshot import frozen_panel, build_run_provenance
    from ..research_overfit import formal_overfit_governance
    from .event_audit import run_event_audit
    with frozen_panel(panel_glob, market) as snap:
        audit = _evaluate_full_vector(expression, universe_n, horizon, portfolio_mode, direction,
            panel_glob, cost_bps, market, evaluation_overrides, direction_policy)
        cfg = evaluation_config(market, evaluation_overrides)
        if cost_bps is not None:
            cfg["base_cost_bps"] = float(cost_bps)
        audit["audit_provenance"] = build_run_provenance(expression,
            {**cfg, "universe_n": universe_n, "horizon": horizon, "portfolio_mode": portfolio_mode,
             "direction": audit["direction"], "isolation": get_layer_bounds(market), "audit_revision": AUDIT_REVISION},
            str(snap.trading_dates[0]), str(snap.trading_dates[-1]), snap.trading_dates)
        if snap.manifest.get("immutable_inputs_available"):
            audit["event_audit"] = run_event_audit(expression, universe_n, horizon, portfolio_mode,
                audit["direction"], market, panel_glob, cfg, snap, audit_artifact_dir)
        else:
            audit["event_audit"] = {"passed": False, "status": "INSUFFICIENT_DATA", "failure_reasons": ["immutable_input_snapshot_missing"]}
        evidence = audit.get("raw_return_evidence")
        history = list(trial_history or [])
        history.append({"id": "current_audit", "statistic": {"raw_return_evidence": evidence}})
        governance = formal_overfit_governance(history, evidence, direction=audit["direction"],
            predeclared_trials=int(cfg["multiple_testing_trials"]))
        if trial_history is None:
            governance.update(passed=False, status="INSUFFICIENT_DATA")
            governance["reasons"].append("registered_trial_history_not_supplied")
        audit["overfit_governance"] = governance
        eligibility = audit["eligibility"]
        eligibility.update(event_pass=bool(audit["event_audit"]["passed"]),
            snapshot_pass=bool(audit["audit_provenance"]["immutable_inputs_available"]),
            overfit_pass=bool(governance["passed"]))
        for key in ("event_pass", "snapshot_pass", "overfit_pass"):
            if not eligibility[key]:
                eligibility["failure_reasons"].append(key)
        eligibility["eligible"] = all(eligibility.get(k, False) for k in (
            "research_pass", "holdout_pass", "vault_pass", "capacity_pass", "event_pass", "snapshot_pass", "overfit_pass"))
        if not eligibility["eligible"] and eligibility["grade"] == "F5":
            eligibility.update(grade="F4", stage="audited_candidate")
        audit["ranking"]["formal_gate_passed"] = eligibility["eligible"]
        audit["ranking"]["promotion_blockers"] = [k for k in ("event_pass", "snapshot_pass", "overfit_pass") if not eligibility[k]]
        return audit


def era_detail(
    expression: str,
    universe_n: int = 500,
    horizon: int = 5,
    panel_glob: str | None = None,
    market: str = "us",
    portfolio_mode: str = "long_short",
    direction: int = 1,
) -> dict:
    """Compatibility view backed by the V4 full audit."""
    audit = evaluate_full(
        expression,
        universe_n,
        horizon,
        portfolio_mode,
        direction,
        panel_glob,
        None,
        market,
        None,
        DIRECTION_POLICY_FIXED,
    )
    eras = []
    layer_summary = {}
    reverse_alias = {alias: layer for layer, alias in LAYER_ALIASES.items()}
    for alias, metrics in audit["layers"].items():
        layer_summary[reverse_alias[alias]] = {
            key: metrics.get(key)
            for key in ("ic_mean", "icir", "era_consistency", "score")
        }
        for row in metrics.get("era_series", []):
            eras.append({
                "era": row["era"],
                "layer": reverse_alias[alias],
                "ic_mean": row["ic"],
                "ic_std": None,
                "days": None,
            })
    return {"layers": layer_summary, "eras": eras, "audit": audit}
