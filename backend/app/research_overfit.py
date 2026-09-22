"""Multiple-testing diagnostics for an append-only factor research campaign.

All functions are deterministic and operate on training/validation evidence.
They must never be fed the frozen rating layer by an automated researcher.
"""

from __future__ import annotations

import itertools
import hashlib
import json
import math
from datetime import date
from statistics import NormalDist
from typing import Any

import numpy as np


PROTOCOL = "factorfactory.dynamic-overfit-governance/v2"
RAW_RETURN_PROTOCOL = "factorfactory.raw-trial-net-returns/v1"
DISCOVERY_LAYERS = frozenset({"INNER_PUBLIC", "META_TRAIN"})
_NORMAL = NormalDist()


def effective_trial_count(score_or_return_matrix: list[list[float]] | np.ndarray) -> float:
    """Eigenvalue participation ratio of correlated trials, bounded to [1, N]."""
    matrix = np.asarray(score_or_return_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return 1.0
    n_trials = matrix.shape[1]
    if n_trials == 1 or matrix.shape[0] < 3:
        return float(n_trials)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.atleast_2d(np.nan_to_num(corr, nan=0.0))
    np.fill_diagonal(corr, 1.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    denominator = float(np.sum(eigenvalues * eigenvalues))
    result = (float(np.sum(eigenvalues)) ** 2 / denominator) if denominator > 0 else 1.0
    return round(max(1.0, min(float(n_trials), result)), 6)


def expected_max_sharpe(effective_trials: float, sharpe_std: float = 1.0) -> float:
    """Expected best noise Sharpe using the Bailey-Lopez de Prado approximation."""
    if not math.isfinite(float(effective_trials)) or not math.isfinite(float(sharpe_std)) or sharpe_std < 0:
        raise ValueError("trial count and nonnegative Sharpe dispersion must be finite")
    trials = max(1.0, float(effective_trials))
    if trials <= 1.0:
        return 0.0
    euler_gamma = 0.5772156649015329
    first = _NORMAL.inv_cdf(max(1e-12, min(1 - 1e-12, 1.0 - 1.0 / trials)))
    second = _NORMAL.inv_cdf(
        max(1e-12, min(1 - 1e-12, 1.0 - 1.0 / (trials * math.e)))
    )
    return max(0.0, float(sharpe_std) * ((1.0 - euler_gamma) * first + euler_gamma * second))


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    observations: int,
    effective_trials: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float = 1.0,
    annualization_factor: float = 1.0,
) -> dict:
    """PSR above the selection hurdle, using *per-observation* Sharpe units.

    Both input Sharpes must use the same units.  Pass the number of periods per
    year when supplying annualized values; the moment correction must never
    mix annualized Sharpe with a count of daily/horizon observations.
    """
    annualization = float(annualization_factor)
    if annualization <= 0 or not math.isfinite(annualization):
        raise ValueError("annualization_factor must be finite and positive")
    if not all(math.isfinite(float(value)) for value in (observed_sharpe, skewness, kurtosis)):
        raise ValueError("Sharpe and return moments must be finite")
    n = int(observations)
    if n < 3:
        return {"protocol": PROTOCOL, "available": False, "passed_95": False,
                "reason": "need_3_return_observations"}
    sr = float(observed_sharpe) / math.sqrt(annualization)
    benchmark = expected_max_sharpe(effective_trials, float(sharpe_std) / math.sqrt(annualization))
    variance_term = max(
        1e-12,
        1.0 - float(skewness) * sr + ((float(kurtosis) - 1.0) / 4.0) * sr * sr,
    )
    z = (sr - benchmark) * math.sqrt(n - 1.0) / math.sqrt(variance_term)
    probability = _NORMAL.cdf(z)
    return {
        "protocol": PROTOCOL,
        "available": True,
        "sharpe_units": "per_observation_not_annualized",
        "input_annualization_factor": annualization,
        "observed_sharpe": round(sr, 6),
        "expected_max_noise_sharpe": round(benchmark, 6),
        "effective_trials": round(float(effective_trials), 6),
        "observations": n,
        "z": round(z, 6),
        "dsr_probability": round(probability, 6),
        "passed_95": probability >= 0.95,
    }


def harvey_liu_haircut(observed_sharpe: float, *, observations: int, trials: float) -> dict:
    """Conservative family-wise Bonferroni haircut expressed in Sharpe units."""
    n = max(2, int(observations))
    attempts = max(1.0, float(trials))
    t_stat = float(observed_sharpe) * math.sqrt(n)
    raw_p = 2.0 * (1.0 - _NORMAL.cdf(abs(t_stat)))
    adjusted_p = min(1.0, raw_p * attempts)
    critical = _NORMAL.inv_cdf(1.0 - min(0.499999, 0.05 / (2.0 * attempts)))
    adjusted_sharpe = math.copysign(max(0.0, abs(t_stat) - critical) / math.sqrt(n), t_stat)
    return {
        "raw_p_value": round(raw_p, 8),
        "familywise_p_value": round(adjusted_p, 8),
        "haircut_sharpe": round(adjusted_sharpe, 6),
        "passed_5pct_familywise": adjusted_p < 0.05,
    }


def cscv_pbo(
    period_return_matrix: list[list[float]] | np.ndarray,
    *,
    blocks: int = 8,
    max_splits: int = 512,
) -> dict:
    """Combinatorially symmetric cross-validation probability of backtest overfit.

    Rows are chronological periods and columns are candidate strategies. For
    each symmetric split, the in-sample winner is ranked out of sample. PBO is
    the fraction whose OOS relative-rank logit is <= 0.
    """
    values = np.asarray(period_return_matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 8 or values.shape[1] < 2:
        return {"protocol": PROTOCOL, "available": False, "reason": "need_8_periods_2_trials"}
    if not np.isfinite(values).all():
        return {"protocol": PROTOCOL, "available": False, "reason": "nonfinite_raw_returns"}
    block_count = min(max(4, int(blocks)), values.shape[0], 16)
    if block_count % 2:
        block_count -= 1
    block_rows = [chunk for chunk in np.array_split(np.arange(values.shape[0]), block_count) if len(chunk)]
    all_combinations = list(itertools.combinations(range(block_count), block_count // 2))
    # A->B and B->A have different IS winners.  Always include both roles.
    canonical = [part for part in all_combinations if 0 in part]
    pair_budget = max(1, int(max_splits) // 2)
    if len(canonical) > pair_budget:
        indices = np.linspace(0, len(canonical) - 1, pair_budget, dtype=int)
        canonical = [canonical[index] for index in indices]
    combinations = []
    for part in canonical:
        combinations.extend([part, tuple(i for i in range(block_count) if i not in part)])
    logits: list[float] = []
    winners: list[int] = []
    split_failures: list[float] = []
    for train_blocks in combinations:
        train_idx = np.concatenate([block_rows[index] for index in train_blocks])
        test_idx = np.concatenate(
            [block_rows[index] for index in range(block_count) if index not in train_blocks]
        )
        train_score = _column_sharpes(values[train_idx])
        test_score = _column_sharpes(values[test_idx])
        tied_winners = np.flatnonzero(train_score == np.max(train_score))
        split_logits = []
        for winner in tied_winners:
            # Midranks make ties and column permutations immaterial.
            tied = int(np.sum(test_score == test_score[winner]))
            rank = 1 + int(np.sum(test_score < test_score[winner])) + (tied - 1) / 2.0
            omega = max(1e-9, min(1 - 1e-9, rank / (values.shape[1] + 1.0)))
            split_logits.append(math.log(omega / (1.0 - omega)))
            winners.append(int(winner))
        logits.extend(split_logits)
        split_failures.append(sum(value <= 0 for value in split_logits) / len(split_logits))
    pbo = float(np.mean(split_failures))
    return {
        "protocol": PROTOCOL,
        "available": True,
        "periods": int(values.shape[0]),
        "trials": int(values.shape[1]),
        "blocks": block_count,
        "splits": len(combinations),
        "all_splits": len(all_combinations),
        "complementary_roles_included": True,
        "selection_metric": "nonannualized_sample_sharpe",
        "tie_policy": "all_is_winners_oos_midranks",
        "split_sampling": "all" if len(combinations) == len(all_combinations) else "deterministic_complement_pairs",
        "pbo": round(pbo, 6),
        "median_oos_rank_logit": round(float(np.median(logits)), 6),
        "distinct_is_winners": len(set(winners)),
        "passed_pbo_25pct": pbo <= 0.25,
    }


def _column_sharpes(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0, ddof=1)
    scores = np.divide(means, stds, out=np.zeros_like(means), where=stds > 1e-15)
    scores[(stds <= 1e-15) & (means > 1e-15)] = np.inf
    scores[(stds <= 1e-15) & (means < -1e-15)] = -np.inf
    return scores


def return_matrix_diagnostics(period_return_matrix, *, candidate_index: int = 0) -> dict:
    """Diagnostic-only DSR/PBO, derived exclusively from raw return columns.

    A submitted Sharpe, sample count or moment cannot override the matrix.
    This endpoint has no campaign/provenance proof and cannot promote factors.
    """
    matrix = np.asarray(period_return_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 8 or matrix.shape[1] < 2:
        raise ValueError("need at least 8 chronological periods and 2 trial columns")
    if not np.isfinite(matrix).all() or float(np.min(matrix)) < -1.0:
        raise ValueError("require finite raw simple net returns >= -1")
    if not 0 <= int(candidate_index) < matrix.shape[1]:
        raise ValueError("candidate_index is outside the return matrix")
    scores = _column_sharpes(matrix)
    if not np.isfinite(scores).all():
        raise ValueError("nonzero constant-return columns have undefined finite Sharpe")
    returns = matrix[:, int(candidate_index)]
    std = float(np.std(returns, ddof=0))
    if std <= 1e-15:
        raise ValueError("candidate return variance is zero")
    z = (returns - np.mean(returns)) / std
    effective = effective_trial_count(matrix)
    sharpe = float(scores[int(candidate_index)])
    observations = matrix.shape[0]
    score_std = max(float(np.std(scores, ddof=1)), 1 / math.sqrt(observations - 1))
    return {
        "protocol": PROTOCOL,
        "scope": "caller_supplied_raw_period_net_returns_diagnostic_only",
        "formal_admission_evidence": False,
        "candidate_index": int(candidate_index),
        "candidate_selection": "explicit_column_not_automatic_best",
        "actual_trials": matrix.shape[1], "observations": observations,
        "effective_trials": effective,
        "sharpe_units": "per_observation_not_annualized",
        "observed_sharpe": sharpe,
        "sharpe_dispersion": score_std,
        "dispersion_source": "std_of_per_trial_period_sharpes_with_null_sampling_floor",
        "dsr": deflated_sharpe_ratio(sharpe, observations=observations,
                                       effective_trials=effective,
                                       skewness=float(np.mean(z ** 3)),
                                       kurtosis=float(np.mean(z ** 4)), sharpe_std=score_std),
        "pbo": cscv_pbo(matrix),
        "harvey_liu": harvey_liu_haircut(sharpe, observations=observations, trials=effective),
        "winner_curse": winner_curse(sharpe, effective_trials=effective, score_std=score_std),
    }


def _digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def build_trial_return_evidence(
    *, paths: list[dict], context: dict, expected_directions: list[int] | tuple[int, ...] = (1, -1),
) -> dict:
    """JSON payload for the existing append-only Trial.statistic ledger.

    Persist it only in the trial ledger, never in an LLM feedback envelope.
    Each path consists of dates, fee-net simple returns and layer names.  The
    function refuses frozen rating/HOLDOUT/Vault data and incomplete directions.
    Dates must identify chronological, non-overlapping return observations.
    """
    expected = sorted(set(int(value) for value in expected_directions))
    if not expected or any(value not in {-1, 1} for value in expected):
        raise ValueError("directions must be a nonempty subset of {-1, +1}")
    clean_paths = []
    for path in paths:
        dates = [str(value)[:10] for value in path.get("dates", [])]
        returns = [float(value) for value in path.get("returns", [])]
        layers = list(path.get("layers", []))
        direction = int(path["direction"])
        if not dates or len(dates) != len(returns) or len(dates) != len(layers):
            raise ValueError("dated raw returns and discovery layers must have equal nonzero length")
        if dates != sorted(set(dates)):
            raise ValueError("raw return dates must be strictly increasing and unique")
        for value in dates:
            date.fromisoformat(value)
        if any(layer not in DISCOVERY_LAYERS for layer in layers):
            raise ValueError("holdout/vault/rating evidence is prohibited in overfit discovery ledger")
        if not np.isfinite(returns).all() or min(returns) < -1.0:
            raise ValueError("net simple returns must be finite and >= -1")
        clean_paths.append({"direction": direction, "dates": dates, "returns": returns, "layers": layers})
    if sorted(path["direction"] for path in clean_paths) != expected:
        raise ValueError("every attempted direction must have exactly one raw return path")
    payload = {
        "protocol": RAW_RETURN_PROTOCOL,
        "scope": "discovery_only_all_attempted_directions",
        "return_semantics": "fee_net_simple_return_not_normalized",
        "context": dict(context),
        "expected_directions": expected,
        "paths": sorted(clean_paths, key=lambda path: path["direction"]),
        "holdout_vault_consumed": False,
    }
    return {**payload, "sha256": _digest(payload)}


def _validated_evidence(value: Any) -> tuple[dict | None, str | None]:
    if not isinstance(value, dict) or value.get("protocol") != RAW_RETURN_PROTOCOL:
        return None, "missing_exact_raw_return_evidence"
    try:
        rebuilt = build_trial_return_evidence(paths=value["paths"], context=value["context"],
                                             expected_directions=value["expected_directions"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, "invalid_raw_return_evidence"
    if rebuilt != value:
        return None, "raw_return_evidence_hash_or_semantics_mismatch"
    return rebuilt, None


def formal_overfit_governance(
    trials: list[Any], candidate_evidence: dict | None, *, direction: int = 1,
    predeclared_trials: int = 1000, dsr_threshold: float = 0.95,
    pbo_threshold: float = 0.25, min_observations: int = 60,
    max_matrix_cells: int = 25_000_000,
) -> dict:
    """Fail-closed formal DSR/PBO gate over the complete registered campaign.

    No winner/recency sampling or reconstructed normalized sketches is allowed.
    A comparison cohort is the complete task/config/snapshot cohort supplied by
    the caller, and its ID/hash list is recorded.  Historical missing paths are
    reported, not silently dropped.  Raw trial count (plus predeclared minimum)
    is used as a conservative upper bound on effective tests; correlation is
    diagnostic only and can never discount unrecorded attempts.
    """
    def read(row, key, default=None):
        return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)

    result = {"protocol": PROTOCOL, "status": "INSUFFICIENT_DATA", "available": False,
              "passed": False, "reasons": [], "dsr": {"available": False},
              "pbo": {"available": False}, "holdout_vault_consumed": False,
              "thresholds": {"dsr_min": float(dsr_threshold), "pbo_max": float(pbo_threshold)},
              "evidence_policy": "all_registered_evaluated_trials_no_winner_sampling"}
    candidate, error = _validated_evidence(candidate_evidence)
    if error:
        result["reasons"].append(f"candidate:{error}")
    evaluated = [row for row in trials if (read(row, "statistic", {}) or {}).get("evaluation_performed", True) is not False]
    trial_ids = [str(read(row, "id", index)) for index, row in enumerate(evaluated)]
    result.update(registered_trials=len(trials), evaluated_trials=len(evaluated),
                  pre_evaluation_rejections=len(trials) - len(evaluated),
                  trial_ids_sha256=_digest({"trial_ids": trial_ids}))
    evidence_rows = []
    missing_ids = []
    attempted_directions = 0
    for index, row in enumerate(evaluated):
        statistic = dict(read(row, "statistic", {}) or {})
        evidence, reason = _validated_evidence(statistic.get("raw_return_evidence"))
        if reason:
            missing_ids.append(str(read(row, "id", index)))
            attempted_directions += max(1, int(statistic.get("directions_evaluated") or 2))
        else:
            attempted_directions += len(evidence["paths"])
            evidence_rows.append(evidence)
    result.update(raw_evidence_trials=len(evidence_rows), missing_evidence_trials=len(missing_ids),
                  missing_trial_ids_sample=missing_ids[:30], attempted_direction_trials=attempted_directions,
                  predeclared_trials=max(1, int(predeclared_trials)),
                  effective_trials=max(1, int(predeclared_trials), attempted_directions),
                  effective_trials_method="conservative_all_direction_count_upper_bound")
    if missing_ids:
        result["reasons"].append("incomplete_historical_trial_returns")
    if not evaluated:
        result["reasons"].append("no_registered_trial_campaign")
    if result["reasons"]:
        return result
    selected = next((path for path in candidate["paths"] if path["direction"] == int(direction)), None)
    if selected is None:
        result["reasons"].append("candidate_direction_missing")
        return result
    context = candidate["context"]
    required_context = {"market", "portfolio_mode", "horizon", "task_name", "panel_snapshot_id", "code_version", "config_hash"}
    if not required_context.issubset(context) or any(context[key] in (None, "") for key in required_context):
        result["reasons"].append("candidate_evaluation_context_incomplete")
        return result
    contexts = [{key: item["context"].get(key) for key in required_context} for item in evidence_rows]
    if any(item != {key: context[key] for key in required_context} for item in contexts):
        result["reasons"].append("heterogeneous_campaign_context_requires_complete_frozen_cohort_replay")
        return result
    if not any(item["sha256"] == candidate["sha256"] for item in evidence_rows):
        result["reasons"].append("candidate_not_registered_in_campaign")
        return result
    paths = [path for item in evidence_rows for path in item["paths"]]
    if any(path["dates"] != selected["dates"] or path["layers"] != selected["layers"] for path in paths):
        result["reasons"].append("nonidentical_observation_calendar_no_fill_or_intersection_allowed")
        return result
    if len(selected["dates"]) < int(min_observations) or len(paths) < 2:
        result["reasons"].append("insufficient_complete_return_periods_or_trials")
        return result
    if len(paths) * len(selected["dates"]) > int(max_matrix_cells):
        result["reasons"].append("complete_matrix_exceeds_audit_resource_limit_no_subset_substitution")
        return result
    matrix = np.asarray([path["returns"] for path in paths], dtype=float).T
    returns = np.asarray(selected["returns"], dtype=float)
    std = float(np.std(returns, ddof=1))
    if std <= 1e-15:
        result["reasons"].append("candidate_zero_return_variance")
        return result
    scores = _column_sharpes(matrix)
    if not np.isfinite(scores).all():
        result["reasons"].append("campaign_contains_zero_variance_nonzero_return_path")
        return result
    centered = returns - np.mean(returns)
    population_std = float(np.std(returns, ddof=0))
    skewness = float(np.mean((centered / population_std) ** 3))
    kurtosis = float(np.mean((centered / population_std) ** 4))
    score_std = float(np.std(scores, ddof=1))
    # Avoid a spuriously zero expected-winner hurdle for duplicate/equal-score
    # trials.  The null sampling SE is a conservative dispersion floor.
    score_std = max(score_std, 1.0 / math.sqrt(len(returns) - 1))
    dsr = deflated_sharpe_ratio(float(np.mean(returns)) / std, observations=len(returns),
                               effective_trials=result["effective_trials"], skewness=skewness,
                               kurtosis=kurtosis, sharpe_std=score_std)
    pbo = cscv_pbo(matrix)
    dsr_pass = bool(dsr.get("available") and dsr["dsr_probability"] >= dsr_threshold)
    pbo_pass = bool(pbo.get("available") and pbo["pbo"] <= pbo_threshold)
    result.update(available=True, status="PASS" if dsr_pass and pbo_pass else "FAIL",
                  passed=dsr_pass and pbo_pass, dsr=dsr, pbo=pbo,
                  dsr_pass=dsr_pass, pbo_pass=pbo_pass, period_count=len(returns),
                  actual_start=selected["dates"][0], actual_end=selected["dates"][-1],
                  raw_evidence_sha256=_digest({"hashes": [item["sha256"] for item in evidence_rows]}),
                  candidate_evidence_sha256=candidate["sha256"], comparison_context=context,
                  sharpe_dispersion=score_std, sharpe_dispersion_floor="1/sqrt(T-1)",
                  dsr_dependence_assumption="nonoverlapping_period_returns; serial_dependence_not_removed")
    if not dsr_pass:
        result["reasons"].append("dsr_below_threshold")
    if not pbo_pass:
        result["reasons"].append("pbo_above_threshold_or_unavailable")
    return result


def winner_curse(observed_best: float, *, effective_trials: float, score_std: float) -> dict:
    expected_luck = expected_max_sharpe(effective_trials, score_std)
    return {
        "observed_best": round(float(observed_best), 6),
        "expected_selection_luck": round(expected_luck, 6),
        "debiased_best": round(float(observed_best) - expected_luck, 6),
    }
