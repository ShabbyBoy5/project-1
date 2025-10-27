# mitbih_wavelet_delineation_features.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import wfdb
from scipy import signal
import pywt

FS = 360  # MIT-BIH sampling rate

# ------------------- RR / HR / HRV -------------------
def load_annotations(record: str, pb_dir='mitdb/'):
    ann = wfdb.rdann(record, 'atr', pn_dir=pb_dir if pb_dir else None)
    return ann.sample.astype(int), list(ann.symbol)

def compute_rr_series(ann_samples: np.ndarray) -> np.ndarray:
    return np.diff(ann_samples).astype(float)

def instantaneous_hr_from_rr(rr_samples: float, fs=FS) -> float:
    rr_sec = rr_samples / fs
    return 60.0 / rr_sec if rr_sec > 0 else np.nan

def hrv_time_metrics(rr_series_samples: np.ndarray, fs=FS) -> Dict[str, float]:
    if rr_series_samples.size < 2:
        return {"SDNN": np.nan, "RMSSD": np.nan}
    rr_ms = (rr_series_samples / fs) * 1000.0
    sdnn = float(np.std(rr_ms, ddof=1)) if rr_ms.size >= 2 else np.nan
    diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(diff**2))) if diff.size >= 1 else np.nan
    return {"SDNN": sdnn, "RMSSD": rmssd}

# ------------------- Wavelet delineation -------------------
# Approach:
# 1) Band-limit beat for stability (optional if upstream filtered) [0.5,40] Hz.
# 2) CWT with a mother wavelet suitable for QRS (e.g., Mexican hat) across scales spanning ~10–60 ms.
# 3) Use modulus maxima and zero-crossings to estimate QRS onset/offset around the R center; search earlier for P, later for T.
# References: wavelet delineation literature and validated algorithms on PhysioNet [web:106][web:86].

def bandpass(x, fs=FS, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
    return signal.filtfilt(b, a, x)

def wavelet_delineate_beat(beat: np.ndarray, fs=FS, pre_ms=200, post_ms=400) -> Dict[str, float]:
    W = len(beat)
    # Ensure filtered
    xb = bandpass(beat, fs=fs, low=0.5, high=40.0, order=4)
    # Choose mother wavelet: Mexican hat ('mexh') is commonly used for QRS [web:106].
    # Build scales to capture 10–60 ms widths: scale ~ width / dt; dt=1/fs.
    dt = 1.0 / fs
    widths = np.arange(int(0.01/dt), int(0.06/dt))  # samples: 10–60 ms
    # Use CWT via convolution approximation: pywt.cwt expects scales, returns coefficients per scale.
    # Translate widths to scales; for 'mexh', scale is proportional to width; use widths directly as scales proxy.
    cwt_coeffs, freqs = pywt.cwt(xb, scales=widths, wavelet='mexh', sampling_period=dt)
    # Aggregate across scales (energy)
    cwt_energy = np.mean(np.abs(cwt_coeffs), axis=0)
    # R index assumed at center of window
    pre = int(round(pre_ms * 1e-3 * fs))
    R_idx = pre
    # QRS boundaries: search around R for strong energy transitions
    left = max(0, R_idx - int(0.12*fs))
    right = min(W-1, R_idx + int(0.12*fs))
    seg = cwt_energy[left:right+1]
    # Threshold relative to segment stats
    thr = np.percentile(seg, 60)
    # QRS onset: last index before R where energy rises above thr
    qrs_on = None
    for i in range(R_idx - left, -1, -1):
        if seg[i] > thr:
            qrs_on = left + i
    if qrs_on is None:
        qrs_on = max(0, R_idx - int(0.04*fs))
    # QRS offset: first index after R where energy drops below thr
    qrs_off = None
    for i in range(R_idx - left, len(seg)):
        if seg[i] < thr:
            qrs_off = left + i
            break
    if qrs_off is None:
        qrs_off = min(W-1, R_idx + int(0.06*fs))
    qrs_on = max(0, min(W-1, qrs_on))
    qrs_off = max(0, min(W-1, qrs_off))
    qrs_dur = (qrs_off - qrs_on) / fs

    # P wave: search before QRS onset using lower scales (broader waves) for P detection
    p_left = max(0, qrs_on - int(0.24*fs))
    p_right = max(0, qrs_on - int(0.04*fs))
    if p_right > p_left:
        # Favor slightly larger scales to capture P morphology; use median across top scales.
        top_scales = cwt_coeffs[-10:, p_left:p_right]
        p_energy = np.mean(np.abs(top_scales), axis=0)
        P_rel = int(np.argmax(p_energy))
        P_idx = p_left + P_rel
        P_amp = float(xb[P_idx])
    else:
        P_idx = max(0, qrs_on - int(0.12*fs))
        P_amp = float(xb[P_idx])

    # T wave: search after QRS offset
    t_left = min(W-1, qrs_off + int(0.04*fs))
    t_right = min(W-1, qrs_off + int(0.4*fs))
    if t_right > t_left:
        top_scales = cwt_coeffs[-10:, t_left:t_right]
        t_energy = np.mean(np.abs(top_scales), axis=0)
        T_rel = int(np.argmax(t_energy))
        T_idx = t_left + T_rel
        T_amp = float(xb[T_idx])
    else:
        T_idx = min(W-1, qrs_off + int(0.2*fs))
        T_amp = float(xb[T_idx])

    # Intervals
    PR = (qrs_on - P_idx) / fs if qrs_on >= P_idx else np.nan
    QT = (T_idx - qrs_on) / fs if T_idx >= qrs_on else np.nan
    # J-point ~ QRS offset; ST deviation = value at J minus baseline (pre-P median)
    base_start = max(0, P_idx - int(0.08*fs))
    base_end = max(1, P_idx - int(0.02*fs))
    baseline = float(np.median(xb[base_start:base_end])) if base_end > base_start else float(np.median(xb[:max(1,int(0.04*fs))]))
    ST_dev = float(xb[qrs_off] - baseline)
    R_amp = float(xb[R_idx])

    return {
        "PR_s": PR,
        "QRS_s": qrs_dur,
        "QT_s": QT,
        "ST_dev": ST_dev,
        "R_amp": R_amp,
        "T_amp": T_amp,
        "P_idx": P_idx,
        "QRS_on": qrs_on,
        "QRS_off": qrs_off,
        "T_idx": T_idx
    }


def wavelet_delineate_beat_unaligned(beat: np.ndarray, fs=FS) -> Dict[str, float]:
    """
    Wavelet delineation for a single beat segment without assuming the R wave is centered.
    Automatically estimates R index from the filtered signal and then reuses the logic
    from wavelet_delineate_beat by setting pre/post around the detected R.

    Note: Without absolute time calibration, returned intervals (PR/QRS/QT) are in seconds
    under the assumed fs.
    """
    xb = bandpass(beat, fs=fs, low=0.5, high=40.0, order=4)
    # Detect R index as the most prominent peak
    peaks, props = signal.find_peaks(xb, prominence=np.std(xb) * 0.5)
    if len(peaks) == 0:
        R_idx = int(np.argmax(np.abs(xb)))
    else:
        prom = props.get("prominences", np.zeros(len(peaks)))
        R_idx = int(peaks[int(np.argmax(prom))])

    # Define window around R to emulate centered-beat delineation
    pre_ms = 200
    post_ms = 400
    pre = int(round(pre_ms * 1e-3 * fs))
    post = int(round(post_ms * 1e-3 * fs))
    start = max(0, R_idx - pre)
    end = min(len(xb), R_idx + post)
    win = xb[start:end]
    # If window too small, pad
    if len(win) < (pre + post):
        pad_left = pre - min(pre, R_idx - start)
        pad_right = pre + post - len(win) - pad_left
        win = np.pad(win, (pad_left, max(0, pad_right)), mode='edge')

    # Now call the centered delineation on the window
    d = wavelet_delineate_beat(win, fs=fs, pre_ms=pre_ms, post_ms=post_ms)
    return d

def qtc_bazett(qt_sec: float, rr_samples: float, fs=FS) -> float:
    if qt_sec is None or np.isnan(qt_sec) or rr_samples is None or np.isnan(rr_samples) or rr_samples <= 0:
        return np.nan
    rr_sec = rr_samples / fs
    return float(qt_sec / np.sqrt(rr_sec))

# ------------------- Augmentation driver -------------------
def augment_with_wavelet_clinical(
    features_csv: str,
    pb_dir='mitdb/',
    pre_ms=200,
    post_ms=400,
    hrv_window_beats=30
) -> pd.DataFrame:
    df = pd.read_csv(features_csv)
    df['record'] = df['record'].astype(str)

    PR_list, QRS_list, QT_list, QTc_list, ST_list, R_amp_list, T_amp_list = [], [], [], [], [], [], []
    RR_list, HR_list, SDNN_list, RMSSD_list = [], [], [], []

    for rec, g in df.groupby('record'):
        ann_samples, ann_symbols = load_annotations(rec, pb_dir=pb_dir)
        idx_map = {int(s): i for i, s in enumerate(ann_samples)}
        # Load raw signal once for delineation
        recsig = wfdb.rdrecord(rec, pn_dir=pb_dir if pb_dir else None)
        # Prefer MLII or first channel
        ch = 0
        if 'MLII' in recsig.sig_name:
            ch = recsig.sig_name.index('MLII')
        elif 'V5' in recsig.sig_name:
            ch = recsig.sig_name.index('V5')
        sig = recsig.p_signal[:, ch].astype(np.float32)

        for _, row in g.iterrows():
            r_samp = int(row['r_sample'])
            # RR/HRV
            if r_samp in idx_map and idx_map[r_samp] > 0:
                i = idx_map[r_samp]
                rr_curr = float(ann_samples[i] - ann_samples[i-1])
                RR_list.append(rr_curr)
                HR_list.append(instantaneous_hr_from_rr(rr_curr, fs=FS))
                start_i = max(1, i - hrv_window_beats)
                rr_win = np.diff(ann_samples[start_i:i+1])
                hrv = hrv_time_metrics(rr_win, fs=FS)
                SDNN_list.append(hrv["SDNN"])
                RMSSD_list.append(hrv["RMSSD"])
            else:
                RR_list.append(np.nan); HR_list.append(np.nan); SDNN_list.append(np.nan); RMSSD_list.append(np.nan)

            # Build beat window around r_samp
            pre = int(round(pre_ms * 1e-3 * FS))
            post = int(round(post_ms * 1e-3 * FS))
            start = r_samp - pre
            end = r_samp + post
            if start >= 0 and end <= len(sig):
                beat = sig[start:end].copy()
                d = wavelet_delineate_beat(beat, fs=FS, pre_ms=pre_ms, post_ms=post_ms)
                PR_list.append(d["PR_s"])
                QRS_list.append(d["QRS_s"])
                QT_list.append(d["QT_s"])
                ST_list.append(d["ST_dev"])
                R_amp_list.append(d["R_amp"])
                T_amp_list.append(d["T_amp"])
                QTc_list.append(qtc_bazett(d["QT_s"], RR_list[-1], fs=FS))
            else:
                PR_list.append(np.nan); QRS_list.append(np.nan); QT_list.append(np.nan); ST_list.append(np.nan)
                R_amp_list.append(np.nan); T_amp_list.append(np.nan); QTc_list.append(np.nan)

    df['RR_samples'] = RR_list
    df['HR_bpm'] = HR_list
    df['SDNN_ms'] = SDNN_list
    df['RMSSD_ms'] = RMSSD_list
    df['PR_s'] = PR_list
    df['QRS_s'] = QRS_list
    df['QT_s'] = QT_list
    df['QTc_Bazett_s'] = QTc_list
    df['ST_dev'] = ST_list
    df['R_amp'] = R_amp_list
    df['T_amp'] = T_amp_list
    return df

if __name__ == "__main__":
    # Input from your existing pipeline
    in_csv = "artifacts/mitbih/all_records_features.csv"
    out_csv = "artifacts/mitbih/all_records_features_wavelet_clinical.csv"
    df_aug = augment_with_wavelet_clinical(in_csv, pb_dir='mitdb/')
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_aug.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
