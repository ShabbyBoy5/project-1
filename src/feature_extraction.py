# mitbih_feature_pipeline.py

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, stats
from sklearn.preprocessing import StandardScaler
import wfdb

# ----------------- Config -----------------
FS = 360  # Hz (MIT-BIH sampling frequency)
# Keep only beat annotations; exclude non-beat marks like '+' etc.
# WFDB annotation symbols list: main beat types include 'N', 'L', 'R', 'V', 'A', etc.
# You can extend this map as needed.
AAMI_MAP = {
    # N class
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # SVEB (S)
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # VEB (V)
    'V': 'V', 'E': 'V',
    # F (fusion of V and N)
    'F': 'F',
    # Q (unknown / paced etc.) - optional
    '/': 'Q', 'f': 'Q', 'Q': 'Q', 'P': 'Q'
}
VALID_BEAT_SYMBOLS = set(AAMI_MAP.keys())

# ----------------- IO helpers -----------------
def read_record_signal(record_name: str, data_dir: str | None = None, pn_dir: str | None = None):
    """
    Reads a MIT-BIH record from either a local directory (data_dir) or PhysioNet (pn_dir).
    Chooses MLII if present, otherwise uses the first channel.
    Returns signal (np.ndarray, shape [T,]), fields dict.
    """
    if data_dir:
        rec_path = str((Path(data_dir) / record_name).as_posix())
        rec = wfdb.rdrecord(rec_path)
    elif pn_dir:
        rec = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    else:
        rec = wfdb.rdrecord(record_name)
    sig = rec.p_signal  # shape [T, n_sig]
    sig_names = [str(x) for x in rec.sig_name]
    # Prefer MLII else V5 else channel 0
    ch = 0
    if 'MLII' in sig_names:
        ch = sig_names.index('MLII')
    elif 'V5' in sig_names:
        ch = sig_names.index('V5')
    return sig[:, ch].astype(np.float32), {'fs': rec.fs, 'sig_name': sig_names}

def read_annotations(record_name: str, data_dir: str | None = None, pn_dir: str | None = None):
    """
    Reads ATR annotations for the record from local directory or PhysioNet.
    Returns sample indices and symbols.
    """
    if data_dir:
        rec_path = str((Path(data_dir) / record_name).as_posix())
        ann = wfdb.rdann(rec_path, 'atr')
    elif pn_dir:
        ann = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)
    else:
        ann = wfdb.rdann(record_name, 'atr')
    return ann.sample.astype(int), list(ann.symbol)

# ----------------- Filtering -----------------
def bandpass_filter(x, fs=FS, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, x)

def notch_filter(x, fs=FS, f0=60.0, Q=30.0):
    b, a = signal.iirnotch(f0, Q, fs)
    return signal.filtfilt(b, a, x)

# ----------------- Beat segmentation -----------------
def segment_beats(signal_1d, ann_samples, ann_symbols, fs=FS,
                  pre_ms=200, post_ms=400, keep_symbols=VALID_BEAT_SYMBOLS):
    """
    Extract fixed windows around annotated beats.
    pre_ms/post_ms define window relative to annotation sample (often near R-peak).
    Returns beats_X (N, W), beat_symbols (N,), beat_locs (N,)
    """
    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(post_ms * 1e-3 * fs))
    W = pre + post
    beats = []
    labels = []
    locs = []
    T = len(signal_1d)
    for s, sym in zip(ann_samples, ann_symbols):
        if sym not in keep_symbols:
            continue
        start = s - pre
        end = s + post
        if start >= 0 and end <= T:
            seg = signal_1d[start:end].copy()
            beats.append(seg)
            labels.append(sym)
            locs.append(s)
    if len(beats) == 0:
        return np.zeros((0, W), dtype=np.float32), [], []
    return np.stack(beats).astype(np.float32), labels, locs

# ----------------- Feature extraction -----------------
def time_features(x):
    dx = np.diff(x)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "skew": float(stats.skew(x)),
        "kurt": float(stats.kurtosis(x, fisher=True)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "zcr": float(((x[:-1] * x[1:]) < 0).sum()),
        "slope": float(np.polyfit(np.arange(len(x)), x, 1)[0]),
        "energy": float(np.sum(x**2)),
        "abs_change": float(np.mean(np.abs(dx)))
    }

def welch_features(x, fs=FS):
    f, Pxx = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    Pxx = Pxx + 1e-12
    centroid = np.sum(f * Pxx) / np.sum(Pxx)
    bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * Pxx) / np.sum(Pxx))
    dom_freq = f[np.argmax(Pxx)]
    flatness = np.exp(np.mean(np.log(Pxx))) / np.mean(Pxx)
    return {
        "spec_centroid": float(centroid),
        "spec_bw": float(bandwidth),
        "spec_dom_freq": float(dom_freq),
        "spec_flatness": float(flatness),
        "psd_mean": float(Pxx.mean()),
        "psd_var": float(Pxx.var())
    }

def simple_morph_ecg(x, prominence=0.2):
    # Local peaks; note beats are already centered near R, but morphology helps
    peaks, props = signal.find_peaks(x, prominence=prominence)
    n_peaks = len(peaks)
    peak_amp_mean = float(np.mean(x[peaks])) if n_peaks > 0 else 0.0
    peak_prom_mean = float(np.mean(props.get("prominences", [0.0]))) if n_peaks > 0 else 0.0
    return {
        "n_peaks": float(n_peaks),
        "peak_amp_mean": peak_amp_mean,
        "peak_prom_mean": peak_prom_mean
    }

def acf_features(x, max_lag=10):
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    acf = [1.0]
    for lag in range(1, max_lag+1):
        if len(x) > lag:
            ac = np.corrcoef(x[:-lag], x[lag:])[0,1]
        else:
            ac = 0.0
        acf.append(float(ac))
    return {f"acf_lag{lag}": acf[lag] for lag in range(1, max_lag+1)}

def extract_features_matrix(beats, fs=FS):
    rows = []
    for i in range(beats.shape[0]):
        x = beats[i]
        feats = {}
        feats.update(time_features(x))
        feats.update(welch_features(x, fs=fs))
        feats.update(simple_morph_ecg(x))
        feats.update(acf_features(x, max_lag=10))
        rows.append(feats)
    return pd.DataFrame(rows)

# ----------------- Orchestration -----------------
def process_record(record_name: str, data_dir: str | None = None, pn_dir: str | None = None,
                   use_notch=False, pre_ms=200, post_ms=400,
                   per_record_zscore=True):
    # Read signal and annotations
    sig, meta = read_record_signal(record_name, data_dir=data_dir, pn_dir=pn_dir)
    fs = int(meta['fs']) if 'fs' in meta else FS  # wfdb.rdrecord has fs
    ann_samp, ann_sym = read_annotations(record_name, data_dir=data_dir, pn_dir=pn_dir)

    # Filter
    x = bandpass_filter(sig, fs=fs, low=0.5, high=40.0, order=4)
    if use_notch:
        x = notch_filter(x, fs=fs, f0=60.0, Q=30.0)

    # Optional per-record standardization (helps stabilize features)
    if per_record_zscore:
        mu, sd = np.mean(x), np.std(x) + 1e-8
        x = (x - mu) / sd

    # Segment beats
    beats, beat_syms, beat_locs = segment_beats(
        x, ann_samp, ann_sym, fs=fs, pre_ms=pre_ms, post_ms=post_ms
    )

    # Map symbols to AAMI classes and to binary anomaly labels
    aami_labels = [AAMI_MAP.get(s, 'Q') for s in beat_syms]
    # Example binary: normal (N) vs anomaly (non-N)
    y_binary = np.array([1 if lab == 'N' else 0 for lab in aami_labels], dtype=int)

    # Features
    feat_df = extract_features_matrix(beats, fs=fs)
    feat_df['symbol'] = beat_syms
    feat_df['aami'] = aami_labels
    feat_df['y_binary_normal_is_1'] = y_binary
    feat_df['record'] = record_name
    feat_df['r_sample'] = beat_locs
    return beats, feat_df

def run_mitbih_pipeline(record_list, out_dir="artifacts/mitbih",
                        data_dir: str | None = None, pn_dir: str | None = None, use_notch=False,
                        pre_ms=200, post_ms=400):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_feats = []
    beats_by_rec = {}
    for rec in record_list:
        beats, feat = process_record(
            rec, data_dir=data_dir, pn_dir=pn_dir, use_notch=use_notch,
            pre_ms=pre_ms, post_ms=post_ms
        )
        beats_by_rec[rec] = beats
        all_feats.append(feat)
        # Save per-record arrays
        np.save(out / f"{rec}_beats.npy", beats)
        feat.to_csv(out / f"{rec}_features.csv", index=False)
    feat_all = pd.concat(all_feats, axis=0, ignore_index=True) if all_feats else pd.DataFrame()
    feat_all.to_csv(out / "all_records_features.csv", index=False)
    return beats_by_rec, feat_all

if __name__ == "__main__":
    # Example usage with local dataset stored under data/mit-bih-arrhythmia-database-1.0.0
    # Choose a subset of records, e.g., '100','101','103','105'
    records = ['100', '101', '103', '105']
    beats_by_record, features_all = run_mitbih_pipeline(
        records,
        out_dir="artifacts/mitbih",
        data_dir="data/mit-bih-arrhythmia-database-1.0.0",
        pn_dir=None,  # Or set to 'mitdb' to fetch from PhysioNet if not local
        use_notch=False,
    )
