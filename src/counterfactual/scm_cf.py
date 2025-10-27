from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from src.scm.learn_scm import LinearLaggedSCM, LinearNodeModel


def load_scm_json(path: str | Path) -> LinearLaggedSCM:
    obj = json.loads(Path(path).read_text())
    cols = obj["columns"]
    max_lag = int(obj.get("max_lag", 1))
    models: Dict[int, LinearNodeModel] = {}
    for j_str, md in obj["models"].items():
        j = int(j_str)
        parents = [(int(i), int(l)) for i, l in md.get("parents", [])]
        coef = np.array(md.get("coef", []), dtype=float)
        intercept = float(md.get("intercept", 0.0))
        sigma = float(md.get("sigma", 0.0))
        z_mean = md.get("z_mean", None)
        z_std = md.get("z_std", None)
        method = md.get("method", "ols")
        models[j] = LinearNodeModel(
            parents=parents,
            coef=coef,
            intercept=intercept,
            sigma=sigma,
            z_mean=(np.array(z_mean, dtype=float) if z_mean is not None else None),
            z_std=(np.array(z_std, dtype=float) if z_std is not None else None),
            method=method,
        )
    return LinearLaggedSCM(columns=cols, models=models, max_lag=max_lag)


def build_ordered_matrix(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    # Sort by record then r_sample if present; else keep current order
    if "record" in df.columns and "r_sample" in df.columns:
        df2 = df.sort_values(["record", "r_sample"]).reset_index(drop=True)
    else:
        df2 = df.reset_index(drop=True)
    X = df2[list(cols)].to_numpy(dtype=float)
    idx_map = df2.index.to_numpy()
    return X, idx_map


def project_time_t_to_scm(
    scm: LinearLaggedSCM,
    X_hist: np.ndarray,
    t: int,
    x_do: np.ndarray,
    intervened: Sequence[int],
) -> np.ndarray:
    """
    Construct x_t consistent with SCM given interventions.
    For nodes in intervened, keep x_do[j]. For others, set to SCM predicted value using X_hist.
    """
    # Predict via SCM for all non-intervened nodes (uses standardization if present),
    # then override intervened with do-values.
    D = len(scm.columns)
    x = scm.predict_t(X_hist, t).astype(float)
    for j in intervened:
        x[j] = float(x_do[j])
    return x


def fit_isolation_forest_pipeline(X_train: np.ndarray, contamination: Optional[float | str] = 'auto') -> Pipeline:
    if contamination is None:
        contamination = 'auto'
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", IsolationForest(random_state=0, contamination=contamination)),
    ])
    pipe.fit(X_train)
    return pipe


def fit_anomaly_pipeline(model: str, X_train: np.ndarray, **kwargs) -> Pipeline:
    model = model.lower()
    if model == 'isoforest':
        contamination = kwargs.get('contamination', 'auto')
        return fit_isolation_forest_pipeline(X_train, contamination=contamination)
    elif model == 'lof':
        n_neighbors = kwargs.get('n_neighbors', 20)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)),
        ])
        pipe.fit(X_train)
        return pipe
    elif model == 'ocsvm':
        nu = kwargs.get('nu', 0.1)
        kernel = kwargs.get('kernel', 'rbf')
        gamma = kwargs.get('gamma', 'scale')
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)),
        ])
        pipe.fit(X_train)
        return pipe
    else:
        raise ValueError("Unknown anomaly model; use 'isoforest', 'lof', or 'ocsvm'")


def predict_normal_flag(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    return (clf.predict(scaler.transform(X)) == 1).astype(int)


def anomaly_score(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    # higher more normal; invert to anomaly score
    return -clf.decision_function(scaler.transform(X))


@dataclass
class CFResult:
    index: int  # original dataframe index
    success: bool
    pred_orig: int  # 1=normal, 0=anomaly at x_orig
    intervened_cols: List[str]
    x_orig: List[float]
    x_cf: List[float]
    score_orig: float
    score_cf: float
    l1: float
    l2: float
    influence_paths: Optional[Dict[str, List[str]]] = None


def grid_candidates_from_train(df: pd.DataFrame, cols: Sequence[str], train_mask: np.ndarray, n_quantiles: int = 5) -> Dict[int, np.ndarray]:
    qs = np.linspace(0.1, 0.9, n_quantiles)
    cand: Dict[int, np.ndarray] = {}
    for j, c in enumerate(cols):
        vals = df.loc[train_mask, c].to_numpy(dtype=float)
        # guard
        if vals.size == 0:
            cand[j] = np.array([df[c].median()])
        else:
            cand[j] = np.quantile(vals, qs).astype(float)
    return cand


def _build_ancestor_map(scm: LinearLaggedSCM) -> Dict[int, List[int]]:
    # adjacency: parent -> child (ignore lag values)
    D = len(scm.columns)
    children_of: Dict[int, List[int]] = {i: [] for i in range(D)}
    for child, m in scm.models.items():
        for (par, _lag) in m.parents:
            children_of[par].append(child)
    # compute ancestors for each node via DFS on reversed graph
    parents_of: Dict[int, List[int]] = {i: [] for i in range(D)}
    for child, m in scm.models.items():
        for (par, _lag) in m.parents:
            parents_of[child].append(par)
    ancestor_map: Dict[int, List[int]] = {}
    for j in range(D):
        visited = set()
        stack = list(parents_of[j])
        while stack:
            p = stack.pop()
            if p in visited:
                continue
            visited.add(p)
            stack.extend(parents_of[p])
        ancestor_map[j] = sorted(list(visited))
    return ancestor_map


def find_cf_for_index(
    df: pd.DataFrame,
    scm: LinearLaggedSCM,
    pipe: Pipeline,
    cols: Sequence[str],
    t_global: int,
    train_mask: np.ndarray,
    max_k: int = 1,
    bounds_pct: Tuple[float, float] = (1.0, 99.0),
    prefer_mahalanobis: bool = True,
    cost_mode: str = "first",  # 'first' | 'best'
    lambda_maha: float = 0.0,
    priority: Optional[List[str]] = None,  # order of feature names to try first
    # strictness controls
    count_already_normal_as_success: bool = False,
    max_l2: Optional[float] = None,
    max_maha: Optional[float] = None,
    min_score_drop: float = 0.0,
    max_resid_z: float = 3.0,
    beam_width: int = 10,
) -> Optional[CFResult]:
    # Build ordered matrix and map to time index t in ordered space
    if "record" in df.columns and "r_sample" in df.columns:
        df_ord = df.sort_values(["record", "r_sample"]).reset_index()
    else:
        df_ord = df.reset_index()
    # find the row with original index == t_global
    row = df_ord[df_ord["index"] == t_global]
    if row.empty:
        return None
    t_ord = int(row.index[0])

    # need sufficient history
    if t_ord < scm.max_lag:
        return None

    X_hist, _ = build_ordered_matrix(df, cols)
    x_orig = X_hist[t_ord].copy()
    y_pred_orig = predict_normal_flag(pipe, x_orig.reshape(1, -1))[0]
    score_orig = float(anomaly_score(pipe, x_orig.reshape(1, -1))[0])
    # Helper to compute distances
    def _distances(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        da = a - b
        l1 = float(np.sum(np.abs(da)))
        l2 = float(np.linalg.norm(da))
        return l1, l2
    if y_pred_orig == 1:
        # already normal
        if count_already_normal_as_success:
            return CFResult(index=t_global, success=True, pred_orig=int(y_pred_orig), intervened_cols=[], x_orig=x_orig.tolist(), x_cf=x_orig.tolist(), score_orig=score_orig, score_cf=score_orig, l1=0.0, l2=0.0)
        else:
            return None

    # Build candidate values per variable
    cand_map = grid_candidates_from_train(df, cols, train_mask, n_quantiles=5)
    # bounds from train percentiles
    train_vals = df.loc[train_mask, cols].to_numpy(dtype=float)
    lo_pct, hi_pct = bounds_pct
    lo = np.nanpercentile(train_vals, lo_pct, axis=0)
    hi = np.nanpercentile(train_vals, hi_pct, axis=0)
    # precompute (approximate) mahalanobis scaling via inverse covariance diagonal
    VI = None
    diag_scale = None
    if prefer_mahalanobis and train_vals.shape[0] > train_vals.shape[1]:
        cov = np.cov(train_vals, rowvar=False)
        # regularize for stability
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        try:
            VI = np.linalg.inv(cov)
            diag_scale = np.sqrt(np.maximum(np.diag(VI), 1e-12))
        except np.linalg.LinAlgError:
            VI = None
            diag_scale = None

    def maha_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = a - b
        if VI is not None:
            try:
                return float(np.sqrt(np.dot(d, VI @ d)))
            except Exception:
                pass
        if diag_scale is not None:
            return float(np.linalg.norm(d * diag_scale))
        return float(np.linalg.norm(d))
    D = len(cols)
    # Compute training anomaly score stats for normalization
    try:
        s_train = anomaly_score(pipe, train_vals)
        s_mu = float(np.mean(s_train)); s_sd = float(np.std(s_train) + 1e-9)
    except Exception:
        s_mu, s_sd = 0.0, 1.0
    # Precompute centroid for Mahalanobis normalization
    x_mu = np.nanmean(train_vals, axis=0)
    if VI is not None:
        # distribution of distances to centroid
        m_train = np.array([np.sqrt((x - x_mu) @ VI @ (x - x_mu)) for x in train_vals])
        m_sd = float(np.std(m_train) + 1e-9)
        m_p95 = float(np.percentile(m_train, 95.0))
    else:
        m_sd = 1.0
        m_p95 = 1.0
    best: Optional[CFResult] = None
    best_success: Optional[CFResult] = None
    best_cost: float = float("inf")

    # establish feature priority order
    D = len(cols)
    if priority:
        priority_set = [c for c in priority if c in cols]
        rest = [c for c in cols if c not in priority_set]
        col_order = priority_set + rest
    else:
        # default spectral-first order if present
        spectral_first = ["spec_centroid", "spec_bw"]
        priority_set = [c for c in spectral_first if c in cols]
        rest = [c for c in cols if c not in priority_set]
        col_order = priority_set + rest
    j_order = [cols.index(c) for c in col_order]

    # Helper: residual z-score consistency check for intervened nodes
    def is_scm_consistent(x_vec: np.ndarray, tol_z: float = 3.0) -> bool:
        # ensure intervened nodes are not egregiously off the SCM residual scale
        x_hat = scm.predict_t(X_hist, t_ord)
        ok = True
        for j in range(D):
            if j in j_order:  # only meaningful for candidates we touch; cheap upper bound
                m = scm.models.get(j)
                if m is None or m.sigma is None or m.sigma == 0.0:
                    continue
                z = abs(x_vec[j] - x_hat[j]) / float(m.sigma)
                if z > tol_z:
                    ok = False
                    break
        return ok

    # Normalize cost terms for comparability across datasets
    def composite_cost(s_cf: float, m_dist: float, lambda_maha: float, normalize: bool = True) -> float:
        if normalize:
            s_term = (s_cf - s_mu) / s_sd
            m_term = (m_dist / m_p95)  # scale by 95th percentile distance
        else:
            s_term, m_term = s_cf, m_dist
        return float(s_term + lambda_maha * m_term)

    # Try k=1 interventions greedy
    ancestor_map = _build_ancestor_map(scm)
    for j in j_order:
        # try candidate values ordered by approximate mahalanobis distance
        vals = list(cand_map[j])
        if diag_scale is not None:
            vals.sort(key=lambda v: abs(v - x_orig[j]) * float(diag_scale[j]))
        for val in vals:
            x_do = x_orig.copy()
            # apply bounds
            x_do[j] = float(np.clip(val, lo[j], hi[j]))
            x_cf = project_time_t_to_scm(scm, X_hist, t_ord, x_do, intervened=[j])
            # SCM-consistency check (residual z-score)
            if not is_scm_consistent(x_cf, tol_z=max_resid_z):
                continue
            y_pred = predict_normal_flag(pipe, x_cf.reshape(1, -1))[0]
            s_cf = float(anomaly_score(pipe, x_cf.reshape(1, -1))[0])
            l1, l2 = _distances(x_cf, x_orig)
            # strictness checks
            m_dist = maha_dist(x_cf, x_orig)
            if max_l2 is not None and l2 > max_l2:
                pass  # over L2 budget
            elif max_maha is not None and m_dist > max_maha:
                pass  # over Mahalanobis budget
            elif y_pred == 1 and (score_orig - s_cf) >= float(min_score_drop):
                if cost_mode == "first":
                    infl = {cols[j]: [cols[a] for a in ancestor_map.get(j, [])]}
                    return CFResult(index=t_global, success=True, pred_orig=int(y_pred_orig), intervened_cols=[cols[j]], x_orig=x_orig.tolist(), x_cf=x_cf.tolist(), score_orig=score_orig, score_cf=s_cf, l1=l1, l2=l2, influence_paths=infl)
                else:
                    cost = composite_cost(s_cf, m_dist, float(lambda_maha), normalize=True)
                    if cost < best_cost:
                        best_cost = cost
                        infl = {cols[j]: [cols[a] for a in ancestor_map.get(j, [])]}
                        best_success = CFResult(index=t_global, success=True, pred_orig=int(y_pred_orig), intervened_cols=[cols[j]], x_orig=x_orig.tolist(), x_cf=x_cf.tolist(), score_orig=score_orig, score_cf=s_cf, l1=l1, l2=l2, influence_paths=infl)
            # track best score decrease even if not flipping
            if best is None or s_cf < best.score_cf:
                best = CFResult(index=t_global, success=False, pred_orig=int(y_pred_orig), intervened_cols=[cols[j]], x_orig=x_orig.tolist(), x_cf=x_cf.tolist(), score_orig=score_orig, score_cf=s_cf, l1=l1, l2=l2)

    # Optionally, try k=2 if requested
    if max_k >= 2:
        for j1 in j_order:
            for j2 in range(j1 + 1, D):
                if j2 not in j_order:
                    continue
                vals1 = list(cand_map[j1])
                vals2 = list(cand_map[j2])
                if diag_scale is not None:
                    vals1.sort(key=lambda v: abs(v - x_orig[j1]) * float(diag_scale[j1]))
                    vals2.sort(key=lambda v: abs(v - x_orig[j2]) * float(diag_scale[j2]))
                for v1 in vals1:
                    for v2 in vals2:
                        x_do = x_orig.copy()
                        x_do[j1] = float(np.clip(v1, lo[j1], hi[j1]))
                        x_do[j2] = float(np.clip(v2, lo[j2], hi[j2]))
                        x_cf = project_time_t_to_scm(scm, X_hist, t_ord, x_do, intervened=[j1, j2])
                        if not is_scm_consistent(x_cf, tol_z=max_resid_z):
                            continue
                        y_pred = predict_normal_flag(pipe, x_cf.reshape(1, -1))[0]
                        s_cf = float(anomaly_score(pipe, x_cf.reshape(1, -1))[0])
                        l1, l2 = _distances(x_cf, x_orig)
                        m_dist = maha_dist(x_cf, x_orig)
                        if max_l2 is not None and l2 > max_l2:
                            continue
                        if max_maha is not None and m_dist > max_maha:
                            continue
                        if y_pred == 1 and (score_orig - s_cf) >= float(min_score_drop):
                            if cost_mode == "first":
                                infl = {
                                    cols[j1]: [cols[a] for a in ancestor_map.get(j1, [])],
                                    cols[j2]: [cols[a] for a in ancestor_map.get(j2, [])],
                                }
                                return CFResult(index=t_global, success=True, pred_orig=int(y_pred_orig), intervened_cols=[cols[j1], cols[j2]], x_orig=x_orig.tolist(), x_cf=x_cf.tolist(), score_orig=score_orig, score_cf=s_cf, l1=l1, l2=l2, influence_paths=infl)
                            else:
                                cost = composite_cost(s_cf, m_dist, float(lambda_maha), normalize=True)
                                if cost < best_cost:
                                    best_cost = cost
                                    infl = {
                                        cols[j1]: [cols[a] for a in ancestor_map.get(j1, [])],
                                        cols[j2]: [cols[a] for a in ancestor_map.get(j2, [])],
                                    }
                                    best_success = CFResult(index=t_global, success=True, pred_orig=int(y_pred_orig), intervened_cols=[cols[j1], cols[j2]], x_orig=x_orig.tolist(), x_cf=x_cf.tolist(), score_orig=score_orig, score_cf=s_cf, l1=l1, l2=l2, influence_paths=infl)
                        if best is None or s_cf < best.score_cf:
                            best = CFResult(index=t_global, success=False, pred_orig=int(y_pred_orig), intervened_cols=[cols[j1], cols[j2]], x_orig=x_orig.tolist(), x_cf=x_cf.tolist(), score_orig=score_orig, score_cf=s_cf, l1=l1, l2=l2)

    # Optional: beam search for k>2
    if max_k > 2:
        beam_width = min(int(beam_width), D)
        # state: (intervened_set, x_cf, s_cf, m_dist)
        initial = (tuple(), x_orig.copy(), float(anomaly_score(pipe, x_orig.reshape(1, -1))[0]), 0.0)
        beam = [initial]
        seen = set([tuple()])
        for depth in range(1, max_k + 1):
            candidates_states = []
            for inter_set, x_cur, s_cur, m_cur in beam:
                remaining = [j for j in j_order if j not in inter_set]
                for j in remaining:
                    for val in cand_map[j]:
                        x_do = x_cur.copy(); x_do[j] = float(np.clip(val, lo[j], hi[j]))
                        new_set = tuple(sorted(inter_set + (j,)))
                        if new_set in seen:
                            continue
                        x_cf = project_time_t_to_scm(scm, X_hist, t_ord, x_do, intervened=list(new_set))
                        if not is_scm_consistent(x_cf, tol_z=max_resid_z):
                            continue
                        s_cf = float(anomaly_score(pipe, x_cf.reshape(1, -1))[0])
                        l1, l2 = _distances(x_cf, x_orig)
                        m_dist = maha_dist(x_cf, x_orig)
                        if (max_l2 is not None and l2 > max_l2) or (max_maha is not None and m_dist > max_maha):
                            continue
                        cost = composite_cost(s_cf, m_dist, float(lambda_maha), normalize=True)
                        candidates_states.append((new_set, x_cf, s_cf, m_dist, cost))
                        seen.add(new_set)
                        # check success
                        y_pred = predict_normal_flag(pipe, x_cf.reshape(1, -1))[0]
                        if y_pred == 1 and (score_orig - s_cf) >= float(min_score_drop):
                            cf_cols = [cols[jj] for jj in new_set]
                            infl = {c: [cols[a] for a in ancestor_map.get(jj, [])] for c, jj in zip(cf_cols, new_set)}
                            return CFResult(index=t_global, success=True, pred_orig=int(y_pred_orig), intervened_cols=cf_cols, x_orig=x_orig.tolist(), x_cf=x_cf.tolist(), score_orig=score_orig, score_cf=s_cf, l1=l1, l2=l2, influence_paths=infl)
            # prune to beam width by lowest cost
            if not candidates_states:
                break
            candidates_states.sort(key=lambda x: x[-1])
            beam = [(s, x, sc, md) for (s, x, sc, md, c) in candidates_states[:beam_width]]

    if best_success is not None:
        return best_success
    return best
