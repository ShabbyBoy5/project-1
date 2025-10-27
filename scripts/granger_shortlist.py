#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pairwise_granger(df: pd.DataFrame, cols: list[str], max_lag: int = 3) -> pd.DataFrame:
    n = len(cols)
    pvals = np.ones((n, n))
    for i, src in enumerate(cols):
        for j, dst in enumerate(cols):
            if i == j:
                continue
            # y depends on x's lags
            yx = df[[dst, src]].dropna().to_numpy(dtype=float)
            try:
                res = grangercausalitytests(yx, maxlag=max_lag, verbose=False)
                # take min pvalue across lags from F-test
                pv = min(float(res[lag][0]["ssr_ftest"][1]) for lag in res)
            except Exception:
                pv = 1.0
            pvals[i, j] = pv
    return pd.DataFrame(pvals, index=cols, columns=cols)


def main():
    ap = argparse.ArgumentParser(description="Pairwise Granger causality shortlist between features.")
    ap.add_argument("--csv", type=str, default="artifacts/ecg5000/features.csv")
    ap.add_argument("--outdir", type=str, default="artifacts/granger_ecg5000")
    ap.add_argument("--cols", type=str, nargs="*", default=None)
    ap.add_argument("--max_lag", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.cols is None:
        candidates = ["mean","std","spec_centroid","spec_bw","acf_lag1","acf_lag2"]
        cols = [c for c in candidates if c in df.columns]
    else:
        cols = [c for c in args.cols if c in df.columns]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pmat = pairwise_granger(df, cols, max_lag=args.max_lag)
    pmat.to_csv(outdir / "granger_pvalues.csv")
    # shortlist edges with p<0.05
    edges = []
    for i, src in enumerate(cols):
        for j, dst in enumerate(cols):
            if i == j:
                continue
            pv = pmat.iloc[i, j]
            if pv < 0.05:
                edges.append({"src": i, "dst": j, "lag": 1, "pval": float(pv)})
    (outdir / "granger_shortlist.json").write_text(json.dumps(edges, indent=2))
    print(f"Saved Granger p-values and shortlist to {outdir}")


if __name__ == "__main__":
    main()
