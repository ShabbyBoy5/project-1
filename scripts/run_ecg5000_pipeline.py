#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

# Ensure project root on sys.path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ecg5000 import run_ecg5000_pipeline


def main():
    ap = argparse.ArgumentParser(description="Run ECG5000 feature extraction pipeline.")
    ap.add_argument("--data_dir", type=str, default="data", help="Directory containing ECG5000_TRAIN.txt and ECG5000_TEST.txt")
    ap.add_argument("--out_dir", type=str, default="artifacts/ecg5000", help="Output directory")
    ap.add_argument("--no_wavelet", action="store_true", help="Disable wavelet augmentation for faster runs")
    args = ap.parse_args()
    out = run_ecg5000_pipeline(args.data_dir, args.out_dir, augment_wavelet=(not args.no_wavelet))
    print(f"Saved features to: {out}")


if __name__ == "__main__":
    main()
