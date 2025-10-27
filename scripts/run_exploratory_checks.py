#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_checks(csv_path: Path, out_dir: Path, columns: list[str] | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if columns is None:
        meta = {"label", "y_binary_normal_is_1", "record", "r_sample", "split"}
        columns = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]
    report = {}

    # ACF plots and ADF tests
    for c in columns:
        x = df[c].to_numpy(dtype=float)
        # ACF
        acf_vals = acf(x, nlags=min(40, max(5, len(x)//10)), fft=True)
        plt.figure(figsize=(6, 3))
        plt.stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        plt.title(f"ACF: {c}")
        plt.xlabel("lag")
        plt.ylabel("acf")
        plt.tight_layout()
        plt.savefig(out_dir / f"acf_{c}.png", dpi=150)
        plt.close()
        # ADF
        try:
            adf = adfuller(x, autolag="AIC")
            report[c] = {
                "adf_stat": float(adf[0]),
                "adf_pvalue": float(adf[1]),
                "lags_used": int(adf[2]),
                "nobs": int(adf[3]),
            }
        except Exception as e:
            report[c] = {"adf_error": str(e)}

    (out_dir / "exploratory_report.json").write_text(__import__("json").dumps(report, indent=2))
    print(f"Saved ACF plots and ADF report to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Exploratory checks: ACF plots and ADF stationarity tests.")
    ap.add_argument("--csv", type=str, default="artifacts/ecg5000/features.csv")
    ap.add_argument("--out", type=str, default="artifacts/exploratory_ecg5000")
    ap.add_argument("--columns", type=str, nargs="*", default=None)
    args = ap.parse_args()
    run_checks(Path(args.csv), Path(args.out), args.columns)


if __name__ == "__main__":
    main()
