from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


@dataclass
class Edge:
    src: int
    dst: int
    lag: int  # positive lag meaning src at t-lag -> dst at t
    score: float
    pval: float


def build_series_from_csv(
    csv_path: str | Path,
    feature_cols: Sequence[str],
    record_col: str = "record",
    time_col: str = "r_sample",
    dropna: str = "any",
) -> Tuple[np.ndarray, Dict[str, List]]:
    """
    Build a multivariate time series X (T x D) by concatenating records ordered by time.
    Returns X and metadata with lists of (record, time) for each row, and column names.
    """
    df = pd.read_csv(csv_path)
    # keep only columns that exist
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        raise ValueError("No requested feature columns found in CSV")
    # order by record then time
    if record_col in df.columns and time_col in df.columns:
        df = df.sort_values([record_col, time_col]).reset_index(drop=True)
    # drop nans in selected cols
    if dropna == "any":
        df = df.dropna(subset=cols)
    elif dropna == "all":
        df = df.dropna(subset=cols, how="all")
    X = df[cols].to_numpy(dtype=float)
    meta = {
        "records": df[record_col].astype(str).tolist() if record_col in df.columns else ["NA"] * len(df),
        "times": df[time_col].astype(int).tolist() if time_col in df.columns else list(range(len(df))),
        "columns": cols,
    }
    return X, meta


def run_pcmci(
    X: np.ndarray,
    tau_max: int = 5,
    test: str = "ParCorr",
    pc_alpha: float | None = 0.05,
    fdr_method: str | None = None,
) -> List[Edge]:
    """
    Run PCMCI(+) on multivariate series X (T x D). Returns significant edges list.
    score is the value in val_matrix (e.g., partial correlation). pval from p_matrix.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D [T, D]")
    T, D = X.shape
    dataframe = pp.DataFrame(X)
    if test.lower() == "parcorr":
        ci_test = ParCorr()
    elif test.lower() == "gpdc":
        try:
            from tigramite.independence_tests.gpdc import GPDC  # type: ignore
        except Exception as e:
            raise ImportError("GPDC requires optional dependency 'dcor'. Install it or use ParCorr.") from e
        ci_test = GPDC()
    else:
        raise ValueError(f"Unknown test '{test}'. Use 'ParCorr' or 'GPDC'.")

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ci_test, verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    val_matrix = results.get("val_matrix")  # shape [D, D, tau_max]
    p_matrix = results.get("p_matrix")
    q_matrix = None
    if fdr_method:
        q_matrix = pcmci.get_corrected_pvalues(p_matrix, fdr_method=fdr_method)

    edges: List[Edge] = []
    for i in range(D):
        for j in range(D):
            for tau in range(1, tau_max + 1):
                score = float(val_matrix[i, j, tau]) if val_matrix is not None else np.nan
                pval = float(p_matrix[i, j, tau]) if p_matrix is not None else 1.0
                passed = pval < (pc_alpha if pc_alpha is not None else 0.05)
                if fdr_method and q_matrix is not None:
                    passed = float(q_matrix[i, j, tau]) < (pc_alpha if pc_alpha is not None else 0.05)
                    pval = float(q_matrix[i, j, tau])
                if passed:
                    edges.append(Edge(src=i, dst=j, lag=tau, score=score, pval=pval))
    return edges


def save_edges_json(edges: List[Edge], out_path: str | Path, columns: Sequence[str]) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "src": e.src,
            "dst": e.dst,
            "lag": e.lag,
            "score": e.score,
            "pval": e.pval,
            "src_name": columns[e.src] if e.src < len(columns) else str(e.src),
            "dst_name": columns[e.dst] if e.dst < len(columns) else str(e.dst),
        }
        for e in edges
    ]
    out.write_text(json.dumps(data, indent=2))
