from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson


def residual_diagnostics(resid: np.ndarray, lags: int = 20) -> Dict[str, float]:
    resid = np.asarray(resid, dtype=float)
    lb = acorr_ljungbox(resid, lags=[min(lags, len(resid)//4)], return_df=True)
    lb_stat = float(lb["lb_stat"].iloc[0])
    lb_pvalue = float(lb["lb_pvalue"].iloc[0])
    dw = float(durbin_watson(resid))
    sk = float(stats.skew(resid, bias=False))
    ku = float(stats.kurtosis(resid, fisher=True, bias=False))
    return {
        "ljung_box_stat": lb_stat,
        "ljung_box_p": lb_pvalue,
        "durbin_watson": dw,
        "resid_skew": sk,
        "resid_kurtosis": ku,
    }


def condition_number(Z: np.ndarray) -> float:
    # Z: design matrix
    s = np.linalg.svd(Z, compute_uv=False)
    if s.min() == 0:
        return float(np.inf)
    return float(s.max() / s.min())


def vif_scores(Z: np.ndarray) -> List[float]:
    # Very simple VIF approximation: regress each column on the others
    # and compute 1/(1-R^2)
    Z = np.asarray(Z, dtype=float)
    n, p = Z.shape
    vifs = []
    for j in range(p):
        X = np.delete(Z, j, axis=1)
        y = Z[:, j]
        # Solve via least squares
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ beta
        ssr = float(np.sum((y_hat - y.mean())**2))
        sst = float(np.sum((y - y.mean())**2)) + 1e-12
        r2 = min(max(ssr / sst, 0.0), 0.999999)
        vifs.append(float(1.0 / (1.0 - r2)))
    return vifs
