from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import json
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import HuberRegressor


@dataclass
class LinearNodeModel:
    parents: List[Tuple[int, int]]  # (parent_index, lag)
    coef: np.ndarray  # shape [len(parents)]
    intercept: float
    sigma: float  # residual std
    z_mean: np.ndarray | None = None  # standardization mean for Z
    z_std: np.ndarray | None = None   # standardization std for Z (avoid zero)
    method: str = "ols"               # ols|ridge|lasso|huber
    r2: float | None = None           # in-sample R^2 for the node


@dataclass
class LinearLaggedSCM:
    columns: Sequence[str]
    models: Dict[int, LinearNodeModel]  # key: node index
    max_lag: int

    def predict_t(self, X_hist: np.ndarray, t: int) -> np.ndarray:
        """
        Predict variables at time t using lagged parents from X_hist.
        X_hist: [T, D]
        """
        D = len(self.columns)
        x = np.zeros(D, dtype=float)
        for j in range(D):
            m = self.models.get(j)
            if m is None or len(m.parents) == 0:
                # no parents -> mean-only
                x[j] = m.intercept if m is not None else 0.0
                continue
            z = []
            for (i, lag) in m.parents:
                z.append(float(X_hist[t - lag, i]))
            z = np.array(z, dtype=float)
            # standardize if model expects it
            if m.z_mean is not None and m.z_std is not None and m.z_mean.size == z.size:
                denom = np.where(m.z_std == 0.0, 1.0, m.z_std)
                z_std = (z - m.z_mean) / denom
                x[j] = float(m.intercept + np.dot(m.coef, z_std))
            else:
                x[j] = float(m.intercept + np.dot(m.coef, z))
        return x

    def to_json(self) -> str:
        obj = {
            "columns": list(self.columns),
            "max_lag": int(self.max_lag),
            "models": {
                str(j): {
                    "parents": [[int(i), int(lag)] for (i, lag) in m.parents],
                    "coef": m.coef.tolist(),
                    "intercept": float(m.intercept),
                    "sigma": float(m.sigma),
                    "z_mean": (m.z_mean.tolist() if m.z_mean is not None else None),
                    "z_std": (m.z_std.tolist() if m.z_std is not None else None),
                    "method": m.method,
                    "r2": (float(m.r2) if m.r2 is not None else None),
                } for j, m in self.models.items()
            },
        }
        return json.dumps(obj, indent=2)


def _parents_by_node(edges: Sequence[dict], D: int) -> Dict[int, List[Tuple[int, int]]]:
    parents: Dict[int, List[Tuple[int, int]]] = {j: [] for j in range(D)}
    for e in edges:
        src = int(e["src"]); dst = int(e["dst"]); lag = int(e["lag"])
        parents[dst].append((src, lag))
    return parents


def _design_for_node(X: np.ndarray, j: int, parents: List[Tuple[int, int]], max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build design for target j using given parents with their lags.
    Returns (y, Z) with shape [T_eff], [T_eff, P].
    """
    T, D = X.shape
    T_eff = T - max_lag
    y = X[max_lag:, j]
    Z = np.zeros((T_eff, len(parents)), dtype=float)
    for k, (i, lag) in enumerate(parents):
        Z[:, k] = X[max_lag - lag:T - lag, i]
    return y, Z


def learn_linear_scm_from_edges(
    X: np.ndarray,
    edges: Sequence[dict],
    columns: Sequence[str],
    max_lag: int,
    method: str = "ols",  # ols|ridge|lasso|huber
    alpha: float = 1.0,
    standardize: bool = True,
) -> LinearLaggedSCM:
    """
    Fit a linear lagged SCM given time series X (T x D) and lagged edges.
    Each node j is regressed on its lagged parents; noise std estimated from residuals.
    """
    T, D = X.shape
    # Validate edges respect lag constraints and node indices
    for e in edges:
        src = int(e["src"]); dst = int(e["dst"]); lag = int(e["lag"])
        if not (0 <= src < D and 0 <= dst < D):
            raise ValueError(f"Edge has out-of-range node index: src={src}, dst={dst}, D={D}")
        if lag < 1 or lag > max_lag:
            raise ValueError(f"Edge lag {lag} outside allowed range [1,{max_lag}] for edge {src}->{dst}")

    parents_map = _parents_by_node(edges, D)
    models: Dict[int, LinearNodeModel] = {}
    for j in range(D):
        parents = parents_map.get(j, [])
        if len(parents) == 0:
            # Mean-only model
            y = X[max_lag:, j]
            mu = float(np.mean(y))
            sigma = float(np.std(y - mu))
            models[j] = LinearNodeModel(parents=[], coef=np.zeros(0), intercept=mu, sigma=sigma, z_mean=None, z_std=None, method=method, r2=None)
            continue
        y, Z = _design_for_node(X, j, parents, max_lag)
        if Z.size == 0:
            mu = float(np.mean(y)); sigma = float(np.std(y - mu))
            models[j] = LinearNodeModel(parents=[], coef=np.zeros(0), intercept=mu, sigma=sigma, z_mean=None, z_std=None, method=method, r2=None)
            continue
        # standardize predictors if requested
        if standardize:
            z_mean = Z.mean(axis=0)
            z_std = Z.std(axis=0)
            z_std[z_std == 0.0] = 1.0
            Z_fit = (Z - z_mean) / z_std
        else:
            z_mean = None
            z_std = None
            Z_fit = Z

        # choose estimator
        mth = method.lower()
        if mth == "ols":
            reg = LinearRegression().fit(Z_fit, y)
        elif mth == "ridge":
            reg = Ridge(alpha=alpha, random_state=0 if hasattr(Ridge(), 'random_state') else None).fit(Z_fit, y)
        elif mth == "lasso":
            reg = Lasso(alpha=alpha, max_iter=10000).fit(Z_fit, y)
        elif mth == "huber":
            reg = HuberRegressor(alpha=alpha).fit(Z_fit, y)
        else:
            raise ValueError("Unknown method; use 'ols', 'ridge', 'lasso', or 'huber'")

        y_hat = reg.predict(Z_fit)
        resid = y - y_hat
        sigma = float(np.std(resid))
        # in-sample R^2
        sst = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
        ssr = float(np.sum((y_hat - np.mean(y)) ** 2))
        r2 = max(min(ssr / sst, 1.0), 0.0)
        models[j] = LinearNodeModel(
            parents=parents,
            coef=np.asarray(reg.coef_, dtype=float),
            intercept=float(reg.intercept_),
            sigma=sigma,
            z_mean=(z_mean if z_mean is None else np.asarray(z_mean, dtype=float)),
            z_std=(z_std if z_std is None else np.asarray(z_std, dtype=float)),
            method=mth,
            r2=r2,
        )

    return LinearLaggedSCM(columns=columns, models=models, max_lag=max_lag)
