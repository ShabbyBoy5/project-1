from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.feature_extraction import (
    time_features,
    welch_features,
    simple_morph_ecg,
    acf_features,
    FS as DEFAULT_FS,
)
from src.wavelet_based_delineation import wavelet_delineate_beat_unaligned


@dataclass
class ECG5000Data:
    X: np.ndarray  # shape [N, T]
    y: np.ndarray  # shape [N,]
    split: np.ndarray  # 0=train, 1=test


def _load_ucr_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, dtype=float)
    y = arr[:, 0].astype(int)
    X = arr[:, 1:].astype(np.float32)
    return X, y


def load_ecg5000(data_dir: str | Path) -> ECG5000Data:
    data_dir = Path(data_dir)
    train_txt = data_dir / "ECG5000_TRAIN.txt"
    test_txt = data_dir / "ECG5000_TEST.txt"
    if not train_txt.exists() or not test_txt.exists():
        raise FileNotFoundError("Expected ECG5000_TRAIN.txt and ECG5000_TEST.txt in data_dir")
    Xtr, ytr = _load_ucr_txt(train_txt)
    Xte, yte = _load_ucr_txt(test_txt)
    X = np.concatenate([Xtr, Xte], axis=0)
    y = np.concatenate([ytr, yte], axis=0)
    split = np.concatenate([
        np.zeros((len(ytr),), dtype=int),
        np.ones((len(yte),), dtype=int)
    ])
    return ECG5000Data(X=X, y=y, split=split)


def extract_features_matrix(X: np.ndarray, fs: int = DEFAULT_FS) -> pd.DataFrame:
    rows = []
    for i in range(X.shape[0]):
        x = X[i]
        # Per-series z-score to stabilize features
        mu, sd = float(np.mean(x)), float(np.std(x) + 1e-8)
        xz = (x - mu) / sd
        feats = {}
        feats.update(time_features(xz))
        feats.update(welch_features(xz, fs=fs))
        feats.update(simple_morph_ecg(xz))
        feats.update(acf_features(xz, max_lag=10))
        rows.append(feats)
    return pd.DataFrame(rows)


def build_features_df(data: ECG5000Data, fs: int = DEFAULT_FS) -> pd.DataFrame:
    df = extract_features_matrix(data.X, fs=fs)
    df["label"] = data.y
    # Binary: class 1 is normal in ECG5000. Others are anomalies.
    df["y_binary_normal_is_1"] = (data.y == 1).astype(int)
    # Provide sequential index as a pseudo-time for downstream ordering
    n = len(data.y)
    df["record"] = ["ECG5000"] * n
    df["r_sample"] = np.arange(n, dtype=int)
    df["split"] = data.split  # 0=train, 1=test
    return df


def run_ecg5000_pipeline(
    data_dir: str | Path = "data",
    out_dir: str | Path = "artifacts/ecg5000",
    augment_wavelet: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_ecg5000(data_dir)
    df = build_features_df(data)
    if augment_wavelet:
        PR_list, QRS_list, QT_list, R_amp_list, T_amp_list = [], [], [], [], []
        for i in range(data.X.shape[0]):
            d = wavelet_delineate_beat_unaligned(data.X[i], fs=DEFAULT_FS)
            PR_list.append(d.get("PR_s", np.nan))
            QRS_list.append(d.get("QRS_s", np.nan))
            QT_list.append(d.get("QT_s", np.nan))
            R_amp_list.append(d.get("R_amp", np.nan))
            T_amp_list.append(d.get("T_amp", np.nan))
        df["PR_s"] = PR_list
        df["QRS_s"] = QRS_list
        df["QT_s"] = QT_list
        df["R_amp"] = R_amp_list
        df["T_amp"] = T_amp_list
    out_csv = out_dir / "features.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    path = run_ecg5000_pipeline()
    print("Saved:", path)
