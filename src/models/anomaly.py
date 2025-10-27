from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


FeatureMatrix = np.ndarray  # shape [N, D]


def _as_colnames(X: FeatureMatrix, cols: Optional[List[str]]) -> List[str]:
    if cols is not None:
        return cols
    return [f"f{i}" for i in range(X.shape[1])]


@dataclass
class AnomalyResult:
    model_name: str
    anomaly_score: np.ndarray  # higher = more anomalous, shape [N]
    y_pred: np.ndarray  # 1=normal, 0=anomaly for consistency with existing column
    metrics: Dict[str, float]
    calibrated_score: Optional[np.ndarray] = None  # optional percentile-calibrated score


def _scores_isoforest(model: IsolationForest, X: FeatureMatrix) -> np.ndarray:
    # decision_function: higher -> more normal; anomaly_score = -decision_function
    s = model.decision_function(X)
    return -s


def _scores_lof(model: LocalOutlierFactor, X: FeatureMatrix) -> np.ndarray:
    # with novelty=True, decision_function exists: higher -> more normal
    s = model.decision_function(X)
    return -s


def _scores_ocsvm(model: OneClassSVM, X: FeatureMatrix) -> np.ndarray:
    # decision_function: signed distance, higher -> more normal
    s = model.decision_function(X)
    return -s


def fit_isolation_forest(X_train: FeatureMatrix, random_state: int = 0, contamination: Optional[float | str] = None) -> Pipeline:
    if contamination is None:
        contamination = 'auto'
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", IsolationForest(random_state=random_state, contamination=contamination)),
    ]).fit(X_train)


def fit_lof_novelty(X_train: FeatureMatrix, n_neighbors: int = 20) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)),
    ]).fit(X_train)


def fit_ocsvm(X_train: FeatureMatrix, nu: float = 0.1, kernel: str = "rbf", gamma: str | float = "scale") -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)),
    ]).fit(X_train)


def _predict_normal_flag(pipeline: Pipeline, X: FeatureMatrix) -> np.ndarray:
    # Map estimator's prediction to 1=normal, 0=anomaly
    est = pipeline.named_steps["clf"]
    yhat = est.predict(pipeline.named_steps["scaler"].transform(X))
    # For IF/OCSVM, normal=1, anomaly=-1; for LOF novelty, same
    return (yhat == 1).astype(int)


def evaluate_scores(y_true_binary_normal_is_1: Optional[np.ndarray], anomaly_score: np.ndarray, y_pred_normal_is_1: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if y_true_binary_normal_is_1 is not None:
        y_true = 1 - y_true_binary_normal_is_1  # convert to 1=anomaly for AUC on anomaly_score
        try:
            metrics["roc_auc_anom"] = float(roc_auc_score(y_true, anomaly_score))
        except Exception:
            pass
        try:
            metrics["pr_auc_anom"] = float(average_precision_score(y_true, anomaly_score))
        except Exception:
            pass
        # Accuracy using model's binary prediction (1=normal -> predict anomaly when 0)
        y_pred_anom = 1 - y_pred_normal_is_1
        try:
            metrics["acc_pred"] = float(accuracy_score(y_true, y_pred_anom))
        except Exception:
            pass
    return metrics


def run_anomaly_model(
    X_train: FeatureMatrix,
    X_all: FeatureMatrix,
    model: str = "isoforest",
    y_true_binary_normal_is_1: Optional[np.ndarray] = None,
    **kwargs,
) -> AnomalyResult:
    if model == "isoforest":
        pipe = fit_isolation_forest(X_train, **kwargs)
        score_fn = _scores_isoforest
    elif model == "lof":
        pipe = fit_lof_novelty(X_train, **kwargs)
        score_fn = _scores_lof
    elif model == "ocsvm":
        pipe = fit_ocsvm(X_train, **kwargs)
        score_fn = _scores_ocsvm
    else:
        raise ValueError("model must be one of: 'isoforest','lof','ocsvm'")

    est = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    Xs = scaler.transform(X_all)
    anomaly_score = score_fn(est, Xs)
    y_pred_normal = _predict_normal_flag(pipe, X_all)
    metrics = evaluate_scores(y_true_binary_normal_is_1, anomaly_score, y_pred_normal)
    return AnomalyResult(model_name=model, anomaly_score=anomaly_score, y_pred=y_pred_normal, metrics=metrics)


def calibrate_scores_quantile(raw_scores: np.ndarray, ref_scores: np.ndarray) -> np.ndarray:
    """
    Map raw anomaly scores to calibrated percentiles using a reference distribution (e.g., validation).
    Returns values in [0,1] where higher = more anomalous percentile.
    """
    ref = np.asarray(ref_scores, dtype=float)
    x = np.asarray(raw_scores, dtype=float)
    ref_sorted = np.sort(ref)
    # For each score, compute empirical CDF
    ranks = np.searchsorted(ref_sorted, x, side="right")
    return ranks / max(len(ref_sorted), 1)
