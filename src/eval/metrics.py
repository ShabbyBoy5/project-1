from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def l2_distance(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    return float(np.linalg.norm(a - b))


def l1_distance(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    return float(np.sum(np.abs(a - b)))


def cf_summary(results: Sequence[dict]) -> Dict[str, float]:
    if not results:
        return {"n": 0, "flip_rate": 0.0, "mean_l2": np.nan, "mean_l1": np.nan, "mean_score_delta": np.nan}
    n = len(results)
    flips = [1 if r.get("success", False) else 0 for r in results]
    deltas = [r.get("score_orig", np.nan) - r.get("score_cf", np.nan) for r in results]
    l2s = []
    l1s = []
    for r in results:
        if "x_orig" in r and "x_cf" in r:
            l2s.append(l2_distance(r["x_orig"], r["x_cf"]))
            l1s.append(l1_distance(r["x_orig"], r["x_cf"]))
    return {
        "n": n,
        "flip_rate": float(np.mean(flips)),
        "mean_l2": float(np.nanmean(l2s)) if l2s else np.nan,
        "mean_l1": float(np.nanmean(l1s)) if l1s else np.nan,
        "mean_score_delta": float(np.nanmean(deltas)),
    }
