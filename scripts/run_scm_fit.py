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

from src.causality.pcmci_pipeline import build_series_from_csv
from src.scm.learn_scm import learn_linear_scm_from_edges


def main():
    ap = argparse.ArgumentParser(description="Fit a linear lagged SCM from features CSV and fused edges JSON.")
    ap.add_argument("--csv", type=str, required=True, help="Features CSV (e.g., artifacts/ecg5000/features.csv)")
    ap.add_argument("--edges", type=str, required=True, help="Edges JSON (e.g., artifacts/.../graph_fused.json)")
    ap.add_argument("--columns", type=str, default=None, help="Optional columns file to override column names")
    ap.add_argument("--out", type=str, default="artifacts/scm/model_linear_lagged.json")
    ap.add_argument("--tau_max", type=int, default=3, help="Max lag used when fitting SCM")
    ap.add_argument("--feature_cols", type=str, nargs="*", default=None, help="Features to include; defaults match run_causality")
    ap.add_argument("--method", type=str, default="ridge", choices=["ols","ridge","lasso","huber"], help="Per-node regression method")
    ap.add_argument("--alpha", type=float, default=1.0, help="Regularization strength for ridge/lasso or alpha for huber")
    args = ap.parse_args()

    # choose features like the causality script
    df = pd.read_csv(args.csv)
    candidates = [
        "PR_s", "QRS_s", "QTc_Bazett_s", "HR_bpm", "ST_dev", "R_amp", "T_amp",
        "mean", "std", "spec_centroid", "spec_bw", "acf_lag1", "acf_lag2",
    ] if args.feature_cols is None else args.feature_cols
    cols = [c for c in candidates if c in df.columns]
    if len(cols) < 2:
        raise SystemExit("Not enough overlapping columns to fit SCM.")

    X, meta = build_series_from_csv(args.csv, cols)
    edges = json.loads(Path(args.edges).read_text())
    model = learn_linear_scm_from_edges(
        X, edges, columns=cols, max_lag=args.tau_max, method=args.method, alpha=args.alpha, standardize=True
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(model.to_json())
    print(f"Saved SCM model: {out}")


if __name__ == "__main__":
    main()
