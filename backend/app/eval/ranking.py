"""Live-oriented factor ranking for Frozen Rating Protocol V4.3.

V4.3 ranks the direction-frozen factor on an explicit 2020-to-latest
full-history view.  The four isolation layers remain available as independent
hard gates and neither LLM receives the full-history result.  Because the
rating includes META_TRAIN, HOLDOUT and Vault dates, it is labelled diagnostic
NON_PIT research evidence rather than independent out-of-sample proof.
"""

from __future__ import annotations

import math
from statistics import NormalDist


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _relevant_branch(metrics: dict, portfolio_mode: str) -> dict:
    branch = metrics.get("active") if portfolio_mode == "long_only" else metrics.get("net")
    return branch if isinstance(branch, dict) else {}


def _relevant_sharpe(metrics: dict, portfolio_mode: str) -> float:
    return _safe_float(_relevant_branch(metrics, portfolio_mode).get("sharpe"))


def _worst_stress_sharpe(metrics: dict) -> float:
    values = [
        _safe_float(row.get("sharpe"), -99.0)
        for row in metrics.get("cost_stress", [])
        if isinstance(row, dict)
    ]
    return min(values) if values else -99.0


def _average_rank(values: list[float]) -> list[float]:
    """One-based average ranks with deterministic tie handling."""
    ordered = sorted(enumerate(values), key=lambda pair: (pair[1], pair[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = average
        cursor = end
    return ranks


def _correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    if variance_x <= 1e-12 or variance_y <= 1e-12:
        return 0.0
    return covariance / math.sqrt(variance_x * variance_y)


def build_live_ranking(
    layers: dict,
    eligibility: dict,
    portfolio_mode: str,
    cfg: dict,
) -> dict:
    """Build the full-history frozen rating plus independent hard-gate status."""
    public = layers.get("public") or {}
    gate = layers.get("gate") or {}
    holdout = layers.get("holdout") or {}
    rating = layers.get("rating") or holdout
    uses_full_history_rating = bool(layers.get("rating"))
    if not rating.get("available"):
        return {
            "available": False,
            "score": None,
            "score_pre_vault": None,
            "score_frozen_rating": None,
            "rating_protocol_version": rating.get("rating_protocol_version"),
            "status": "missing_rating_window",
            "basis": "FROZEN_RATING 2020 through latest available panel date",
            "failure_reasons": ["2020 至最新交易日没有有效样本，无法生成冻结评级"],
        }

    relevant = _relevant_branch(rating, portfolio_mode)
    confidence = rating.get("return_confidence") or {}
    holdout_relevant = _relevant_branch(holdout, portfolio_mode)
    holdout_confidence = holdout.get("return_confidence") or {}
    target_sharpe = max(0.1, _safe_float(cfg.get("target_rank_sharpe"), 1.5))
    target_return = max(0.01, _safe_float(cfg.get("target_rank_ann_return"), 0.12))
    target_absolute_return = max(
        0.01,
        _safe_float(cfg.get("target_absolute_ann_return"), target_return),
    )
    target_cost_cushion = max(
        1.0,
        _safe_float(cfg.get("target_cost_cushion_multiple"), 3.0),
    )

    sharpe_lcb = _safe_float(confidence.get("sharpe_lcb"), -99.0)
    ann_return_lcb = _safe_float(confidence.get("ann_return_lcb"), -99.0)
    calmar = _safe_float(relevant.get("calmar"), -99.0)
    relative_profitability = (
        0.45 * _clip01(sharpe_lcb / target_sharpe)
        + 0.35 * _clip01(ann_return_lcb / target_return)
        + 0.20 * _clip01(calmar / 1.5)
    )
    if portfolio_mode == "long_only":
        absolute = rating.get("net") or {}
        absolute_confidence = rating.get("absolute_return_confidence") or {}
        absolute_profitability = (
            0.55
            * _clip01(
                _safe_float(absolute_confidence.get("sharpe_lcb"), -99.0)
                / target_sharpe
            )
            + 0.45
            * _clip01(
                _safe_float(absolute_confidence.get("ann_return_lcb"), -99.0)
                / target_absolute_return
            )
        )
        # A long-only factor must add alpha, but actual absolute P&L still matters.
        profitability = 0.75 * relative_profitability + 0.25 * absolute_profitability
        absolute_snapshot = {
            "sharpe": absolute.get("sharpe"),
            "ann_return": absolute.get("ann_return"),
            "sharpe_lcb": absolute_confidence.get("sharpe_lcb"),
            "ann_return_lcb": absolute_confidence.get("ann_return_lcb"),
        }
    else:
        profitability = relative_profitability
        absolute_snapshot = None

    trial_count = max(1, int(_safe_float(cfg.get("multiple_testing_trials"), 1)))
    alpha = min(0.25, max(1e-6, _safe_float(cfg.get("multiple_testing_alpha"), 0.10)))
    hurdle_t = NormalDist().inv_cdf(1.0 - alpha / trial_count)
    return_t = _safe_float(confidence.get("hac_t_stat"), -99.0)
    psr = _safe_float(confidence.get("probabilistic_sharpe_gt_zero"))
    selection_confidence = (
        0.65 * _clip01(return_t / max(1e-9, hurdle_t))
        + 0.35 * _clip01(psr)
    )

    profitable_era_rate = _safe_float(rating.get("profitable_era_rate"))
    profitable_year_rate = _safe_float(rating.get("profitable_year_rate"))
    worst_era_sharpe = _safe_float(rating.get("worst_era_sharpe"), -99.0)
    stress_sharpe = _worst_stress_sharpe(rating)
    robustness = (
        0.40 * _clip01((stress_sharpe + 0.25) / 1.75)
        + 0.25 * _clip01(profitable_era_rate)
        + 0.15 * _clip01(profitable_year_rate)
        + 0.20 * _clip01((worst_era_sharpe + 0.50) / 2.0)
    )

    train_sharpes = [
        _relevant_sharpe(public, portfolio_mode),
        _relevant_sharpe(gate, portfolio_mode),
    ]
    train_reference = max(0.25, min(train_sharpes))
    holdout_sharpe = _relevant_sharpe(holdout, portfolio_mode)
    sharpe_retention = _clip01(holdout_sharpe / train_reference)
    direction_preserved = float(
        all(
            _safe_float(layer.get("ic_mean")) > 0
            and _relevant_sharpe(layer, portfolio_mode) > 0
            for layer in (public, gate, holdout)
        )
    )
    generalization = 0.75 * sharpe_retention + 0.25 * direction_preserved

    cost_cushion = _safe_float(rating.get("cost_cushion_multiple"))
    turnover = _safe_float(
        rating.get("daily_turnover", rating.get("turnover")),
        99.0,
    )
    max_turnover = max(0.01, _safe_float(cfg.get("max_daily_turnover"), 0.5))
    adv_participation = _safe_float(rating.get("adv_participation_p95"), 99.0)
    max_adv = max(1e-6, _safe_float(cfg.get("max_adv_participation"), 0.05))
    implementation = (
        0.45 * _clip01(cost_cushion / target_cost_cushion)
        + 0.30 * math.exp(-turnover / max_turnover)
        + 0.25 * math.exp(-adv_participation / max_adv)
    )

    signal_quality = (
        0.60 * _clip01(_safe_float(rating.get("icir")) / 2.0)
        + 0.40 * _clip01(_safe_float(rating.get("monotonicity")))
    )
    components = {
        "net_profitability_lcb": profitability,
        "selection_confidence": selection_confidence,
        "cost_regime_robustness": robustness,
        "oos_generalization": generalization,
        "implementability": implementation,
        "signal_quality": signal_quality,
    }
    weights = {
        "net_profitability_lcb": 0.35,
        "selection_confidence": 0.20,
        "cost_regime_robustness": 0.20,
        "oos_generalization": 0.10,
        "implementability": 0.10,
        "signal_quality": 0.05,
    }
    weighted = sum(weights[name] * components[name] for name in weights)
    critical_floor = min(
        components["net_profitability_lcb"],
        components["selection_confidence"],
        components["cost_regime_robustness"],
        components["implementability"],
    )
    rating_score_raw = 100.0 * weighted * (0.70 + 0.30 * critical_floor)

    if not eligibility.get("research_pass"):
        cap, status = 24.9, "research_rejected"
    elif not eligibility.get("holdout_pass"):
        cap, status = 49.9, "holdout_rejected"
    elif not eligibility.get("vault_pass"):
        cap, status = 64.9, "vault_rejected"
    elif not eligibility.get("capacity_pass"):
        cap, status = 79.9, "capacity_limited"
    elif rating_score_raw >= 75.0:
        cap, status = 100.0, "capital_priority_non_pit"
    elif rating_score_raw >= 60.0:
        cap, status = 100.0, "paper_priority"
    else:
        cap, status = 100.0, "passed_low_conviction"
    score = min(rating_score_raw, cap)

    warnings: list[str] = []
    if return_t < hurdle_t:
        warnings.append(
            f"冻结评级窗口收益 t={return_t:.2f} 未达到 {trial_count} 次检验校正门槛 {hurdle_t:.2f}"
        )
    if sharpe_retention < 0.5:
        warnings.append("HOLDOUT 费后 Sharpe 不足训练层保守值的 50%")
    if cost_cushion < target_cost_cushion:
        warnings.append("成本盈亏平衡缓冲未达到目标值")

    return {
        "available": True,
        "score": round(score, 2),
        "score_frozen_rating": round(rating_score_raw, 2),
        "rating_protocol_version": rating.get("rating_protocol_version"),
        "score_pre_vault": (
            None if uses_full_history_rating else round(rating_score_raw, 2)
        ),
        "status": status,
        "basis": (
            "FROZEN_RATING 2020 through latest available panel date; "
            "full-window NON_PIT diagnostic; isolation-layer hard gates remain separate"
            if uses_full_history_rating
            else "PUBLIC + META_TRAIN + META_HOLDOUT; Vault numeric outcomes excluded"
        ),
        "rating_window": {
            "start": rating.get("window_start"),
            "end": rating.get("window_end"),
            "policy": rating.get("window_policy") or (
                "legacy_holdout" if not uses_full_history_rating else None
            ),
            "independent_out_of_sample": bool(
                rating.get("independent_out_of_sample", not uses_full_history_rating)
            ),
        },
        "vault_seal": "pass" if eligibility.get("vault_pass") else "fail",
        "components": {
            name: round(value, 4) for name, value in components.items()
        },
        "weights": weights,
        "evidence": {
            "portfolio_basis": "active" if portfolio_mode == "long_only" else "net",
            "rating_sharpe": relevant.get("sharpe"),
            "rating_ann_return": relevant.get("ann_return"),
            "rating_sharpe_lcb": confidence.get("sharpe_lcb"),
            "rating_ann_return_lcb": confidence.get("ann_return_lcb"),
            "rating_return_hac_t": confidence.get("hac_t_stat"),
            "rating_psr_gt_zero": confidence.get("probabilistic_sharpe_gt_zero"),
            "holdout_sharpe": holdout_relevant.get("sharpe"),
            "holdout_ann_return": holdout_relevant.get("ann_return"),
            "holdout_sharpe_lcb": holdout_confidence.get("sharpe_lcb"),
            "holdout_ann_return_lcb": holdout_confidence.get("ann_return_lcb"),
            "holdout_return_hac_t": holdout_confidence.get("hac_t_stat"),
            "holdout_psr_gt_zero": holdout_confidence.get("probabilistic_sharpe_gt_zero"),
            "multiple_testing_trials": trial_count,
            "multiple_testing_hurdle_t": round(hurdle_t, 4),
            "cost_breakeven_bps": rating.get("cost_breakeven_bps"),
            "cost_cushion_multiple": rating.get("cost_cushion_multiple"),
            "worst_stress_sharpe": round(stress_sharpe, 4),
            "profitable_era_rate": rating.get("profitable_era_rate"),
            "profitable_year_rate": rating.get("profitable_year_rate"),
            "sharpe_retention": round(sharpe_retention, 4),
            "adv_participation_p95": rating.get("adv_participation_p95"),
            "absolute_long_only": absolute_snapshot,
        },
        "warnings": warnings,
        "failure_reasons": list(eligibility.get("failure_reasons") or []),
        "policy_label": "NON_PIT_RESEARCH",
        "production_approved": False,
    }


def ranking_diagnostics(items: list[dict], minimum_sample: int = 8) -> dict:
    """Measure frozen pre-vault rank against numeric vault outcomes."""
    clean = []
    for item in items:
        score = _safe_float(item.get("score_pre_vault"), math.nan)
        outcome = _safe_float(item.get("vault_ann_return"), math.nan)
        sharpe = _safe_float(item.get("vault_sharpe"), math.nan)
        if all(math.isfinite(value) for value in (score, outcome, sharpe)):
            clean.append({
                "factor_id": item.get("factor_id"),
                "score": score,
                "outcome": outcome,
                "sharpe": sharpe,
            })
    n = len(clean)
    base = {
        "sample_size": n,
        "minimum_sample": minimum_sample,
        "basis": "frozen pre-vault score versus FACTOR_VAULT fee-after outcome",
        "uses_vault_to_tune_score": False,
    }
    if n < minimum_sample:
        return {
            **base,
            "status": "insufficient_sample",
            "message": f"至少需要 {minimum_sample} 个 V4 完整审计因子，当前只有 {n} 个",
            "metrics": None,
        }

    scores = [row["score"] for row in clean]
    outcomes = [row["outcome"] for row in clean]
    spearman = _correlation(_average_rank(scores), _average_rank(outcomes))
    ordered = sorted(clean, key=lambda row: (row["score"], row["factor_id"] or 0))
    quartile_n = max(1, math.ceil(n / 4))
    bottom = ordered[:quartile_n]
    top = ordered[-quartile_n:]

    bucket_count = min(4, n)
    buckets: list[dict] = []
    for bucket in range(bucket_count):
        start = bucket * n // bucket_count
        end = (bucket + 1) * n // bucket_count
        rows = ordered[start:end]
        buckets.append({
            "bucket": bucket + 1,
            "n": len(rows),
            "mean_score": round(sum(row["score"] for row in rows) / len(rows), 4),
            "mean_vault_ann_return": round(
                sum(row["outcome"] for row in rows) / len(rows),
                6,
            ),
            "positive_rate": round(
                sum(row["outcome"] > 0 and row["sharpe"] > 0 for row in rows)
                / len(rows),
                4,
            ),
        })
    monotonicity = _correlation(
        [float(row["bucket"]) for row in buckets],
        [float(row["mean_vault_ann_return"]) for row in buckets],
    )
    positive = lambda rows: sum(  # noqa: E731
        row["outcome"] > 0 and row["sharpe"] > 0 for row in rows
    ) / len(rows)
    top_positive_rate = positive(top)
    bottom_positive_rate = positive(bottom)
    overall_positive_rate = positive(clean)
    calibrated = (
        spearman is not None
        and spearman >= 0.25
        and top_positive_rate >= overall_positive_rate
        and top_positive_rate > bottom_positive_rate
        and monotonicity is not None
        and monotonicity > 0
    )
    return {
        **base,
        "status": "calibrated" if calibrated else "needs_review",
        "message": (
            "高分组在冻结 Vault 中表现出更好的费后结果"
            if calibrated
            else "当前证据不能证明高分因子在冻结 Vault 中更赚钱"
        ),
        "metrics": {
            "spearman_score_vs_vault_return": round(spearman or 0.0, 4),
            "bucket_monotonicity": round(monotonicity or 0.0, 4),
            "top_quartile_positive_rate": round(top_positive_rate, 4),
            "bottom_quartile_positive_rate": round(bottom_positive_rate, 4),
            "overall_positive_rate": round(overall_positive_rate, 4),
            "top_quartile_mean_vault_ann_return": round(
                sum(row["outcome"] for row in top) / len(top),
                6,
            ),
            "bottom_quartile_mean_vault_ann_return": round(
                sum(row["outcome"] for row in bottom) / len(bottom),
                6,
            ),
            "buckets": buckets,
        },
    }
