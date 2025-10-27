from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LassoCV


@dataclass
class Edge:
    src: int
    dst: int
    lag: int  # src at t-lag -> dst at t
    weight: float


def _lagged_design(X: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lagged design for VAR: returns (Y, Z) where
    Y: (T - max_lag, D) targets at time t
    Z: (T - max_lag, D * max_lag) concatenated lags for all variables
    """
    T, D = X.shape
    T_eff = T - max_lag
    if T_eff <= 5:
        raise ValueError("Time series too short for the requested max_lag")
    Y = X[max_lag:, :].copy()
    Z = np.zeros((T_eff, D * max_lag), dtype=float)
    for lag in range(1, max_lag + 1):
        Z[:, (lag - 1) * D : lag * D] = X[max_lag - lag : T - lag, :]
    return Y, Z


def fit_var_lasso_granger(
    X: np.ndarray,
    max_lag: int = 5,
    cv: int = 3,
    coef_threshold: float = 1e-6,
) -> List[Edge]:
    """
    Sparse VAR via LassoCV. For each target j, fit Lasso over concatenated lagged predictors.
    Edges are reported where absolute coefficient exceeds threshold.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D [T, D]")
    T, D = X.shape
    Y, Z = _lagged_design(X, max_lag=max_lag)

    edges: List[Edge] = []
    for j in range(D):
        y = Y[:, j]
        # Fit Lasso with CV for alpha
        model = LassoCV(cv=cv, random_state=0, n_jobs=None).fit(Z, y)
        coef = model.coef_  # shape [D * max_lag]
        for lag in range(1, max_lag + 1):
            block = coef[(lag - 1) * D : lag * D]
            for i in range(D):
                w = float(block[i])
                if abs(w) > coef_threshold:
                    edges.append(Edge(src=i, dst=j, lag=lag, weight=w))
    return edges
