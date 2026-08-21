"""Training-safe return-path sketches used to detect economic duplicates."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable


SIGNATURE_BUCKETS = 48


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def build_return_path_signature(
    returns: Iterable[float],
    *,
    buckets: int = SIGNATURE_BUCKETS,
) -> dict:
    values = [value for item in returns if (value := _finite(item)) is not None]
    if not values:
        return {"available": False, "n": 0, "vector": [], "fingerprint": ""}
    size = max(4, min(int(buckets), len(values)))
    compressed = []
    for index in range(size):
        start = index * len(values) // size
        end = (index + 1) * len(values) // size
        window = values[start:end]
        compressed.append(sum(window) / max(1, len(window)))
    mean = sum(compressed) / len(compressed)
    variance = sum((value - mean) ** 2 for value in compressed) / max(
        1, len(compressed) - 1
    )
    scale = math.sqrt(max(variance, 0.0))
    vector = (
        [(value - mean) / scale for value in compressed]
        if scale > 1e-12
        else [0.0 for _ in compressed]
    )
    rounded = [round(value, 6) for value in vector]
    fingerprint = hashlib.sha256(
        ",".join(f"{value:.6f}" for value in rounded).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "available": True,
        "protocol": f"chronological_block_zscore_v1_{size}",
        "n": len(values),
        "buckets": size,
        "vector": rounded,
        "fingerprint": fingerprint,
    }


def return_path_correlation(left: dict | None, right: dict | None) -> float | None:
    left_values = list((left or {}).get("vector") or [])
    right_values = list((right or {}).get("vector") or [])
    if len(left_values) < 4 or len(left_values) != len(right_values):
        return None
    pairs = [
        (a, b)
        for raw_a, raw_b in zip(left_values, right_values)
        if (a := _finite(raw_a)) is not None and (b := _finite(raw_b)) is not None
    ]
    if len(pairs) < 4:
        return None
    xs, ys = zip(*pairs)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((value - mx) ** 2 for value in xs)
    vy = sum((value - my) ** 2 for value in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in pairs)
    return max(-1.0, min(1.0, cov / math.sqrt(vx * vy)))


def combined_training_signature(public: dict, gate: dict) -> dict:
    left = (public or {}).get("return_path_signature") or {}
    right = (gate or {}).get("return_path_signature") or {}
    vectors = list(left.get("vector") or []) + list(right.get("vector") or [])
    if not vectors:
        return {"available": False, "vector": [], "fingerprint": ""}
    fingerprint = hashlib.sha256(
        "|".join(
            [str(left.get("fingerprint") or ""), str(right.get("fingerprint") or "")]
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "available": bool(left.get("available") and right.get("available")),
        "protocol": "public_plus_meta_train_return_path_v1",
        "vector": vectors,
        "fingerprint": fingerprint,
        "public_fingerprint": left.get("fingerprint"),
        "gate_fingerprint": right.get("fingerprint"),
    }

