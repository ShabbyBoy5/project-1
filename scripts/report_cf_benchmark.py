#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import cf_summary


def summarize_model(cf_path: Path) -> dict:
    data = json.loads(cf_path.read_text())
    s = cf_summary(data)
    # intervention frequencies among successful CFs
    counts = Counter()
    for r in data:
        if r.get("success", False):
            for c in r.get("intervened_cols", []):
                counts[c] += 1
    top = counts.most_common(5)
    s["top_intervened"] = top
    return s


def main():
    ap = argparse.ArgumentParser(description="Aggregate CF summaries across models and export CSV and intervention stats.")
    ap.add_argument("--dir", type=str, default="artifacts/counterfactuals_ecg5000")
    ap.add_argument("--out", type=str, default="artifacts/counterfactuals_ecg5000/benchmark")
    args = ap.parse_args()

    indir = Path(args.dir)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in ["isoforest", "lof", "ocsvm"]:
        cf_file = indir / f"cf_results_{model}.json"
        if not cf_file.exists():
            continue
        s = summarize_model(cf_file)
        rows.append({
            "model": model,
            "n": s.get("n", 0),
            "flip_rate": s.get("flip_rate", np.nan),
            "mean_l1": s.get("mean_l1", np.nan),
            "mean_l2": s.get("mean_l2", np.nan),
            "mean_score_delta": s.get("mean_score_delta", np.nan),
            "top_intervened": "; ".join([f"{k}:{v}" for k, v in s.get("top_intervened", [])]),
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(outdir / "cf_benchmark_summary.csv", index=False)
        print(df)
        print(f"Saved: {outdir / 'cf_benchmark_summary.csv'}")
    else:
        print("No CF result files found to summarize.")


if __name__ == "__main__":
    main()
