#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Make 'src' importable when running as a script
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.causality.pcmci_pipeline import build_series_from_csv, run_pcmci, save_edges_json
from src.causality.granger_var_lasso import fit_var_lasso_granger
from src.causality.graph_fusion import fuse_union, fuse_intersection


def _pick_existing_columns(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]


def main():
    ap = argparse.ArgumentParser(description="Run PCMCI+ and VAR-LASSO causality on feature CSV and fuse graphs.")
    ap.add_argument("--csv", type=str, default="artifacts/mitbih/all_records_features_wavelet_clinical.csv",
                    help="Path to features CSV (augmented preferred). Fallback to non-augmented if missing.")
    ap.add_argument("--fallback_csv", type=str, default="artifacts/mitbih/all_records_features.csv",
                    help="Fallback CSV if augmented CSV is unavailable.")
    ap.add_argument("--tau_max", type=int, default=5)
    ap.add_argument("--max_lag", type=int, default=5)
    ap.add_argument("--pc_alpha", type=float, default=0.05)
    ap.add_argument("--fdr", type=str, default=None, help="FDR correction method for PCMCI ('fdr_bh', etc.)")
    ap.add_argument("--outdir", type=str, default="artifacts/causality")
    ap.add_argument("--fuse", type=str, choices=["union", "intersection"], default="union")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        csv_path = Path(args.fallback_csv)
    if not csv_path.exists():
        raise SystemExit(f"Features CSV not found: {args.csv} or {args.fallback_csv}. Run feature pipeline first.")

    df = pd.read_csv(csv_path)
    # Candidate feature columns: prefer clinical + some robust statistical features
    candidates = [
        "PR_s", "QRS_s", "QTc_Bazett_s", "HR_bpm", "ST_dev", "R_amp", "T_amp",
        "mean", "std", "spec_centroid", "spec_bw", "acf_lag1", "acf_lag2",
    ]
    cols = _pick_existing_columns(df, candidates)
    if len(cols) < 3:
        # fallback to a small set of basic features
        basic = _pick_existing_columns(df, ["mean", "std", "spec_centroid", "spec_bw"]) 
        cols = basic
    if len(cols) < 2:
        raise SystemExit("Not enough overlapping columns in CSV to build a multivariate series.")

    X, meta = build_series_from_csv(csv_path, cols)

    # PCMCI+
    pcmci_edges = run_pcmci(X, tau_max=args.tau_max, test="ParCorr", pc_alpha=args.pc_alpha, fdr_method=args.fdr)
    pcmci_json = [e.__dict__ for e in pcmci_edges]

    # VAR-LASSO Granger
    granger_edges = fit_var_lasso_granger(X, max_lag=args.max_lag)
    granger_json = [{"src": e.src, "dst": e.dst, "lag": e.lag, "score": float(abs(e.weight))} for e in granger_edges]

    # Fuse
    if args.fuse == "union":
        fused = fuse_union(pcmci_json, granger_json, weight_mode="avg")
    else:
        fused = fuse_intersection(pcmci_json, granger_json, weight_mode="avg")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # Save artifacts
    (outdir / "pcmci_edges.json").write_text(json.dumps(pcmci_json, indent=2))
    (outdir / "granger_edges.json").write_text(json.dumps(granger_json, indent=2))
    (outdir / "graph_fused.json").write_text(json.dumps(fused, indent=2))
    # Column names for reference
    (outdir / "columns.json").write_text(json.dumps(meta["columns"], indent=2))

    print(f"Saved causality artifacts to: {outdir}")


if __name__ == "__main__":
    main()
