#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.anomaly import run_anomaly_model, calibrate_scores_quantile


def choose_features(df: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested:
        cols = [c for c in requested if c in df.columns]
    else:
        candidates = [
            "PR_s", "QRS_s", "QTc_Bazett_s", "HR_bpm", "ST_dev", "R_amp", "T_amp",
            "mean", "std", "spec_centroid", "spec_bw",
            "acf_lag1", "acf_lag2", "acf_lag3"
        ]
        cols = [c for c in candidates if c in df.columns]
        if len(cols) < 3:
            # fallback to all numeric non-meta columns
            meta = {"label", "y_binary_normal_is_1", "record", "r_sample", "split"}
            cols = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]
    return cols


def main():
    ap = argparse.ArgumentParser(description="Train an anomaly model and score ECG5000 features.")
    ap.add_argument("--csv", type=str, default="artifacts/ecg5000/features.csv")
    ap.add_argument("--model", type=str, default="isoforest", choices=["isoforest", "lof", "ocsvm"])
    ap.add_argument("--features", type=str, nargs="*", default=None, help="Explicit feature columns list")
    ap.add_argument("--outdir", type=str, default="artifacts/anomaly_ecg5000")
    ap.add_argument("--contamination", type=float, default=None, help="Contamination for IsolationForest")
    ap.add_argument("--nu", type=float, default=0.1, help="nu for OneClassSVM")
    ap.add_argument("--calibrate", action="store_true", help="Calibrate anomaly scores to percentiles using validation split if available")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    cols = choose_features(df, args.features)
    if len(cols) < 2:
        raise SystemExit("Need at least 2 features for anomaly model.")
    X = df[cols].to_numpy(dtype=float)

    # Train on train-split AND only normal beats if label exists
    if "split" in df.columns:
        train_mask = (df["split"].to_numpy() == 0)
    else:
        train_mask = np.ones(len(df), dtype=bool)
    if "y_binary_normal_is_1" in df.columns:
        normal_mask = (df["y_binary_normal_is_1"].to_numpy() == 1)
        train_mask = train_mask & normal_mask
    X_train = X[train_mask]

    kwargs = {}
    if args.model == "isoforest":
        kwargs["contamination"] = args.contamination
    elif args.model == "ocsvm":
        kwargs["nu"] = args.nu

    y_true = df["y_binary_normal_is_1"].to_numpy() if "y_binary_normal_is_1" in df.columns else None
    res = run_anomaly_model(X_train, X, model=args.model, y_true_binary_normal_is_1=y_true, **kwargs)

    # Save augmented CSV
    score_col = f"anomaly_score_{args.model}"
    pred_col = f"anomaly_pred_{args.model}_normal_is_1"
    df[score_col] = res.anomaly_score
    if args.calibrate:
        # Use validation split (split==1) if available; else fall back to training normals
        if "split" in df.columns:
            val_mask = (df["split"].to_numpy() == 1)
        else:
            # temporal validation: last 20%
            n = len(df)
            val_mask = np.zeros(n, dtype=bool)
            val_mask[int(0.8 * n):] = True
        ref = df.loc[val_mask, score_col].to_numpy(dtype=float)
        if ref.size == 0:
            ref = df.loc[train_mask, score_col].to_numpy(dtype=float)
        df[f"{score_col}_cal"] = calibrate_scores_quantile(df[score_col].to_numpy(dtype=float), ref)
    df[pred_col] = res.y_pred
    out_csv = outdir / f"features_with_{args.model}.csv"
    df.to_csv(out_csv, index=False)

    # Save metrics
    (outdir / f"metrics_{args.model}.json").write_text(json.dumps(res.metrics, indent=2))

    print(f"Saved: {out_csv}")
    print(f"Metrics: {res.metrics}")


if __name__ == "__main__":
    main()
