"""Time-ordered OOF residual search and diversity-aware beam selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PROTOCOL = "factorfactory.residual-oof-beam/v1"


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(left) != len(right):
        return 0.0
    return float(np.corrcoef(_rank(left), _rank(right))[0, 1])


def _ridge_predict(train_x, train_y, test_x, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients


def time_ordered_oof_residuals(
    target: list[float] | np.ndarray,
    incumbent_predictions: list[list[float]] | np.ndarray,
    *,
    folds: int = 5,
    ridge_alpha: float = 1e-3,
) -> np.ndarray:
    """Predict every fold from other chronological folds and return OOF residuals."""
    y = np.asarray(target, dtype=float)
    x = np.asarray(incumbent_predictions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(y) != len(x) or len(y) < max(10, folds * 2):
        raise ValueError("target/incumbent rows must match and cover at least two rows per fold")
    blocks = np.array_split(np.arange(len(y)), min(int(folds), len(y) // 2))
    prediction = np.zeros(len(y), dtype=float)
    all_rows = np.arange(len(y))
    for test_rows in blocks:
        train_rows = np.setdiff1d(all_rows, test_rows, assume_unique=True)
        prediction[test_rows] = _ridge_predict(x[train_rows], y[train_rows], x[test_rows], ridge_alpha)
    return y - prediction


@dataclass(frozen=True)
class BeamCandidate:
    name: str
    residual_rank_ic: float
    incremental_oof_ic: float
    independence: float
    stability: float
    turnover: float
    complexity: float
    score: float


def residual_oof_beam_search(
    *,
    target: list[float] | np.ndarray,
    incumbent_predictions: list[list[float]] | np.ndarray,
    candidates: dict[str, list[float] | np.ndarray],
    folds: int = 5,
    beam_width: int = 5,
    turnover: dict[str, float] | None = None,
    complexity: dict[str, float] | None = None,
) -> dict:
    residual = time_ordered_oof_residuals(target, incumbent_predictions, folds=folds)
    incumbent = np.asarray(incumbent_predictions, dtype=float)
    if incumbent.ndim == 1:
        incumbent = incumbent[:, None]
    rows: list[BeamCandidate] = []
    for name, raw in candidates.items():
        prediction = np.asarray(raw, dtype=float)
        if len(prediction) != len(residual):
            raise ValueError(f"candidate {name!r} length mismatch")
        residual_ic = rank_correlation(prediction, residual)
        base_ic = rank_correlation(np.mean(incumbent, axis=1), np.asarray(target, dtype=float))
        joint_ic = rank_correlation(
            np.mean(np.column_stack([incumbent, prediction]), axis=1),
            np.asarray(target, dtype=float),
        )
        incremental = joint_ic - base_ic
        correlations = [abs(rank_correlation(prediction, incumbent[:, index])) for index in range(incumbent.shape[1])]
        independence = 1.0 - max(correlations, default=0.0)
        blocks = np.array_split(np.arange(len(prediction)), min(folds, len(prediction)))
        block_ics = [rank_correlation(prediction[index], residual[index]) for index in blocks]
        stability = max(0.0, 1.0 - float(np.std(block_ics)))
        turn = float((turnover or {}).get(name, 0.0))
        comp = float((complexity or {}).get(name, 0.0))
        score = 0.35 * residual_ic + 0.25 * incremental + 0.20 * independence + 0.20 * stability - 0.10 * turn - 0.05 * comp
        rows.append(BeamCandidate(name, residual_ic, incremental, independence, stability, turn, comp, score))
    rows.sort(key=lambda row: (row.score, row.independence, row.name), reverse=True)
    return {
        "protocol": PROTOCOL,
        "folds": folds,
        "beam_width": min(beam_width, len(rows)),
        "residual_std": round(float(np.std(residual)), 8),
        "beam": [
            {
                key: (round(value, 8) if isinstance(value, float) else value)
                for key, value in row.__dict__.items()
            }
            for row in rows[:beam_width]
        ],
    }
