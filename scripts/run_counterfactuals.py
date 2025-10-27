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

from src.counterfactual.scm_cf import load_scm_json, fit_anomaly_pipeline, find_cf_for_index


def select_anomalies(df: pd.DataFrame, colname: str | None) -> np.ndarray:
    if colname and colname in df.columns:
        return (df[colname].to_numpy().astype(int) == 0)
    # otherwise, select all indices as candidates (will be filtered by model)
    return np.ones(len(df), dtype=bool)


def main():
    ap = argparse.ArgumentParser(description="Generate SCM-based counterfactuals to flip IsolationForest predictions.")
    ap.add_argument("--csv", type=str, default="artifacts/anomaly_ecg5000/features_with_isoforest.csv")
    ap.add_argument("--scm", type=str, default="artifacts/scm/ecg5000_linear_lagged.json")
    ap.add_argument("--model", type=str, default="isoforest", choices=["isoforest","lof","ocsvm"], help="Anomaly model to flip")
    ap.add_argument("--pred_col", type=str, default=None, help="Optional column with model's normal flag (1=normal); defaults to anomaly_pred_{model}_normal_is_1 if present")
    ap.add_argument("--outdir", type=str, default="artifacts/counterfactuals_ecg5000")
    ap.add_argument("--max_k", type=int, default=1, help="Max number of variables to intervene")
    ap.add_argument("--bounds_pct", type=float, nargs=2, default=[0.0, 100.0], help="Lower and upper percentiles for per-feature bounds from train data (relaxed)")
    ap.add_argument("--prefer_mahalanobis", action="store_true", help="Prioritize candidates by approximate Mahalanobis distance")
    ap.add_argument("--cost_mode", type=str, default="best", choices=["first","best"], help="Return first feasible CF or best composite-cost CF")
    ap.add_argument("--lambda_maha", type=float, default=0.1, help="Weight for Mahalanobis distance in composite cost")
    ap.add_argument("--priority", type=str, nargs="*", default=["spec_centroid","spec_bw"], help="Feature names to prioritize first in search order")
    ap.add_argument("--count_already_normal_as_success", action="store_true", help="If set, counts already-normal rows as successful CFs with zero change")
    ap.add_argument("--min_score_drop", type=float, default=0.0, help="Minimum anomaly score decrease required for a flip to count as success")
    ap.add_argument("--max_l2", type=float, default=None, help="Maximum L2 distance allowed for a CF to count as success")
    ap.add_argument("--max_maha", type=float, default=None, help="Maximum Mahalanobis distance allowed for a CF to count as success")
    ap.add_argument("--limit", type=int, default=50, help="Max number of anomalies to process")
    ap.add_argument("--beam_width", type=int, default=10, help="Beam width when max_k>2 (approximate search)")
    ap.add_argument("--max_resid_z", type=float, default=3.0, help="Reject CFs whose intervened residual z-score exceeds this (SCM consistency)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scm = load_scm_json(args.scm)
    cols = list(scm.columns)

    df = pd.read_csv(args.csv)
    # Build training mask: train split and (if present) normal label
    if "split" in df.columns:
        train_mask = (df["split"].to_numpy() == 0)
    else:
        train_mask = np.ones(len(df), dtype=bool)
    if "y_binary_normal_is_1" in df.columns:
        train_mask = train_mask & (df["y_binary_normal_is_1"].to_numpy() == 1)

    # Fit anomaly model on training normals using the same feature columns as SCM
    X = df[cols].to_numpy(dtype=float)
    pipe = fit_anomaly_pipeline(args.model, X[train_mask])

    # Choose candidate indices
    default_pred_col = f"anomaly_pred_{args.model}_normal_is_1"
    use_pred_col = args.pred_col if args.pred_col else (default_pred_col if default_pred_col in df.columns else None)
    cand_mask = select_anomalies(df, use_pred_col)
    cand_idx = np.where(cand_mask)[0]

    results = []
    processed = 0
    for idx in cand_idx:
        if processed >= args.limit:
            break
        res = find_cf_for_index(
            df, scm, pipe, cols, t_global=int(idx), train_mask=train_mask,
            max_k=args.max_k, bounds_pct=tuple(args.bounds_pct), prefer_mahalanobis=bool(args.prefer_mahalanobis),
            cost_mode=args.cost_mode, lambda_maha=args.lambda_maha, priority=args.priority,
            count_already_normal_as_success=bool(args.count_already_normal_as_success),
            max_l2=args.max_l2, max_maha=args.max_maha, min_score_drop=args.min_score_drop,
            max_resid_z=args.max_resid_z, beam_width=args.beam_width
        )
        if res is None:
            continue
        results.append({
            "index": int(res.index),
            "success": bool(res.success),
            "pred_orig": int(res.pred_orig) if hasattr(res, "pred_orig") else None,
            "intervened_cols": res.intervened_cols,
            "score_orig": float(res.score_orig),
            "score_cf": float(res.score_cf),
            "x_orig": res.x_orig,
            "x_cf": res.x_cf,
            "influence_paths": (res.influence_paths if hasattr(res, "influence_paths") else None),
        })
        processed += 1

    out_json = outdir / f"cf_results_{args.model}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved counterfactuals: {out_json} (n={len(results)})")


if __name__ == "__main__":
    main()
