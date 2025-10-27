#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causality.pcmci_pipeline import build_series_from_csv
from src.scm.learn_scm import learn_linear_scm_from_edges
from src.eval.diagnostics import residual_diagnostics, condition_number, vif_scores


def one_step_forecast_rmse(X: np.ndarray, scm, max_lag: int) -> float:
    T, D = X.shape
    preds = []
    trues = []
    for t in range(max_lag, T):
        x_hat = scm.predict_t(X, t)
        preds.append(x_hat)
        trues.append(X[t])
    preds = np.vstack(preds)
    trues = np.vstack(trues)
    return float(np.sqrt(np.mean((preds - trues) ** 2)))


def multi_step_forecast_rmse(X: np.ndarray, scm, max_lag: int, horizon: int = 3) -> float:
    T, D = X.shape
    errors = []
    for t in range(max_lag, T - horizon):
        hist = X.copy()
        preds = []
        cur_t = t
        for h in range(horizon):
            x_hat = scm.predict_t(hist, cur_t)
            preds.append(x_hat)
            # append prediction to history for next step
            hist[cur_t] = x_hat
            cur_t += 1
        pred_arr = np.vstack(preds)
        true_arr = X[t : t + horizon]
        errors.append((pred_arr - true_arr) ** 2)
    if not errors:
        return float('nan')
    se = np.mean(np.vstack(errors))
    return float(np.sqrt(se))


def per_node_residuals(X: np.ndarray, scm, max_lag: int):
    T, D = X.shape
    resids = np.zeros((T - max_lag, D))
    for t in range(max_lag, T):
        x_hat = scm.predict_t(X, t)
        resids[t - max_lag] = X[t] - x_hat
    return resids


def autocorr(x: np.ndarray, nlags: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    denom = np.dot(x, x) + 1e-12
    ac = np.correlate(x, x, mode="full")
    mid = len(ac) // 2
    acf = ac[mid: mid + nlags + 1] / denom
    return acf[1:]  # drop lag 0


def main():
    ap = argparse.ArgumentParser(description="Evaluate linear lagged SCM: forecast RMSE and residual diagnostics.")
    ap.add_argument("--csv", type=str, default="artifacts/ecg5000/features.csv")
    ap.add_argument("--edges", type=str, default="artifacts/causality_ecg5000/graph_fused.json")
    ap.add_argument("--out", type=str, default="artifacts/scm/eval_report.json")
    ap.add_argument("--cols", type=str, nargs="*", default=None)
    ap.add_argument("--tau_max", type=int, default=3)
    ap.add_argument("--method", type=str, default="ridge", choices=["ols","ridge","lasso","huber"], help="Per-node regression method for SCM fit during eval")
    ap.add_argument("--alpha", type=float, default=1.0, help="Regularization strength for SCM eval fit")
    args = ap.parse_args()

    # Build X
    if args.cols is None:
        # fallback to SCM-ish subset
        candidates = ["mean","std","spec_centroid","spec_bw","acf_lag1","acf_lag2"]
    else:
        candidates = args.cols
    X, meta = build_series_from_csv(args.csv, candidates)

    # Learn SCM from edges
    edges = json.loads(Path(args.edges).read_text())
    scm = learn_linear_scm_from_edges(
        X, edges, columns=candidates, max_lag=args.tau_max, method=args.method, alpha=args.alpha, standardize=True
    )

    # Forecast metrics
    rmse_1 = one_step_forecast_rmse(X, scm, args.tau_max)
    rmse_h = multi_step_forecast_rmse(X, scm, args.tau_max, horizon=3)

    # Residuals and diagnostics per node
    resids = per_node_residuals(X, scm, args.tau_max)
    diag = {"overall": {"rmse_1step": rmse_1, "rmse_3step": rmse_h}}
    for j, name in enumerate(candidates):
        rj = resids[:, j]
        d = residual_diagnostics(rj, lags=20)
        # include in-sample R^2 if available and residual autocorrelation summary
        try:
            r2 = scm.models.get(j).r2 if scm.models.get(j) is not None else None
        except Exception:
            r2 = None
        acf_vals = autocorr(rj, nlags=10)
        diag[name] = {**d, "r2_in": (float(r2) if r2 is not None else None), "resid_acf": [float(v) for v in acf_vals]}

    # Condition number and VIF for each node's design
    # reuse design matrix building used in learner
    from src.scm.learn_scm import _parents_by_node, _design_for_node
    parents_map = _parents_by_node(edges, len(candidates))
    Z_info = {}
    for j, name in enumerate(candidates):
        parents = parents_map.get(j, [])
        if not parents:
            continue
        _, Z = _design_for_node(X, j, parents, args.tau_max)
        Z_info[name] = {
            "cond_number": condition_number(Z),
            "vif": [float(x) for x in vif_scores(Z)],
            "parents": [(int(i), int(l)) for (i, l) in parents],
        }

    # Simple Granger-style holdout validation: temporal split
    split_t = int(0.8 * X.shape[0])
    X_tr, X_te = X[:split_t], X[split_t:]
    scm_tr = learn_linear_scm_from_edges(X_tr, edges, columns=candidates, max_lag=args.tau_max, method=args.method, alpha=args.alpha, standardize=True)
    # per-node out-of-sample R^2 compared to mean-only baseline
    granger_holdout = {}
    from src.scm.learn_scm import _parents_by_node, _design_for_node
    parents_map = _parents_by_node(edges, len(candidates))
    for j, name in enumerate(candidates):
        parents = parents_map.get(j, [])
        if not parents:
            continue
        y_te, Z_te = _design_for_node(X_te, j, parents, args.tau_max)
        # standardize using training stats if available
        m = scm_tr.models.get(j)
        if m is None or Z_te.size == 0:
            continue
        if m.z_mean is not None and m.z_std is not None and m.z_mean.size == Z_te.shape[1]:
            denom = np.where(m.z_std == 0.0, 1.0, m.z_std)
            Z_te_std = (Z_te - m.z_mean) / denom
        else:
            Z_te_std = Z_te
        y_hat_te = float(m.intercept) + Z_te_std @ m.coef
        resid_te = y_te - y_hat_te
        sst_te = float(np.sum((y_te - np.mean(y_te)) ** 2)) + 1e-12
        ssr_te = float(np.sum((y_hat_te - np.mean(y_te)) ** 2))
        r2_out = max(min(ssr_te / sst_te, 1.0), 0.0)
        # mean-only baseline
        mu = float(np.mean(y_te))
        ssr_null = float(np.sum((np.full_like(y_te, mu) - np.mean(y_te)) ** 2))
        r2_null = max(min(ssr_null / sst_te, 1.0), 0.0)
        granger_holdout[name] = {"r2_out": r2_out, "r2_null": r2_null, "delta_vs_null": float(r2_out - r2_null)}

    report = {"forecast": diag, "design": Z_info, "granger_holdout": granger_holdout}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"rmse_1step": rmse_1, "rmse_3step": rmse_h}, indent=2))
    print(f"Saved SCM eval report: {out}")


if __name__ == "__main__":
    main()
