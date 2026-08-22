"""Multiple-testing diagnostics for an append-only factor research campaign.

All functions are deterministic and operate on training/validation evidence.
They must never be fed the frozen rating layer by an automated researcher.
"""

from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np


PROTOCOL = "factorfactory.dynamic-overfit-governance/v1"
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
    trials = max(1.0, float(effective_trials))
    if trials <= 1.0:
        return 0.0
    euler_gamma = 0.5772156649015329
    first = _NORMAL.inv_cdf(max(1e-12, min(1 - 1e-12, 1.0 - 1.0 / trials)))
    second = _NORMAL.inv_cdf(
        max(1e-12, min(1 - 1e-12, 1.0 - 1.0 / (trials * math.e)))
    )
    return float(sharpe_std) * ((1.0 - euler_gamma) * first + euler_gamma * second)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    observations: int,
    effective_trials: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sharpe_std: float = 1.0,
) -> dict:
    """Return the probability that Sharpe exceeds the expected lucky winner."""
    benchmark = expected_max_sharpe(effective_trials, sharpe_std)
    sr = float(observed_sharpe)
    n = max(2, int(observations))
    variance_term = max(
        1e-12,
        1.0 - float(skewness) * sr + ((float(kurtosis) - 1.0) / 4.0) * sr * sr,
    )
    z = (sr - benchmark) * math.sqrt(n - 1.0) / math.sqrt(variance_term)
    probability = _NORMAL.cdf(z)
    return {
        "protocol": PROTOCOL,
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
    block_count = min(max(4, int(blocks)), values.shape[0])
    if block_count % 2:
        block_count -= 1
    block_rows = [chunk for chunk in np.array_split(np.arange(values.shape[0]), block_count) if len(chunk)]
    combinations = list(itertools.combinations(range(block_count), block_count // 2))
    # A split and its complement carry the same information; keep one side.
    combinations = combinations[: max(1, min(int(max_splits), len(combinations) // 2))]
    logits: list[float] = []
    winners: list[int] = []
    for train_blocks in combinations:
        train_idx = np.concatenate([block_rows[index] for index in train_blocks])
        test_idx = np.concatenate(
            [block_rows[index] for index in range(block_count) if index not in train_blocks]
        )
        train_score = np.nanmean(values[train_idx], axis=0)
        test_score = np.nanmean(values[test_idx], axis=0)
        winner = int(np.nanargmax(train_score))
        rank = 1 + int(np.sum(test_score < test_score[winner]))
        omega = max(1e-9, min(1 - 1e-9, rank / (values.shape[1] + 1.0)))
        logits.append(math.log(omega / (1.0 - omega)))
        winners.append(winner)
    pbo = sum(value <= 0.0 for value in logits) / len(logits)
    return {
        "protocol": PROTOCOL,
        "available": True,
        "periods": int(values.shape[0]),
        "trials": int(values.shape[1]),
        "blocks": block_count,
        "splits": len(logits),
        "pbo": round(pbo, 6),
        "median_oos_rank_logit": round(float(np.median(logits)), 6),
        "distinct_is_winners": len(set(winners)),
        "passed_pbo_25pct": pbo <= 0.25,
    }


def winner_curse(observed_best: float, *, effective_trials: float, score_std: float) -> dict:
    expected_luck = expected_max_sharpe(effective_trials, score_std)
    return {
        "observed_best": round(float(observed_best), 6),
        "expected_selection_luck": round(expected_luck, 6),
        "debiased_best": round(float(observed_best) - expected_luck, 6),
    }
