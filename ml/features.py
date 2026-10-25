"""One feature vector per window of Muse-style data. Used for training (EPIStress) and live (your Muse 2).

Inputs for one window:
  eeg   (4, n) microvolts, channels TP9, AF7, AF8, TP10, at eeg_fs
  ppg   (k, m) optical pulse (1 channel for the Empatica wristband, 3 for the Muse), at ppg_fs; or None
  acc   (3, j) accelerometer in g, or None
Features:
  EEG    log relative power per band and channel (delta..gamma), log(alpha/beta), frontal alpha asymmetry
         Relative power (band / total) is used instead of absolute microvolts because amplitudes differ a lot
         between headsets, electrode contact and people.
  Heart  heart rate, RMSSD and SDNN (heart rate variability) from beats detected in the pulse wave
  Motion standard deviation of the accelerometer magnitude (head movement)
Missing signals give NaN; LightGBM handles NaN, logistic regression gets the training mean instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clutch.processing import BANDS, _band_power, filter_eeg  # noqa: E402  same filter as the live app

CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
WINDOW_S = 10.0
STEP_S = 5.0
MAX_EEG_STD_UV = 100.0   # a channel above this after filtering is off the skin: the window is dropped

EEG_FEATURES = [f"{b}_{c}" for c in CHANNELS for b in BANDS] + ["alpha_beta_log", "alpha_asym"]
HEART_FEATURES = ["hr_bpm", "hrv_rmssd_ms", "hrv_sdnn_ms"]
MOTION_FEATURES = ["motion_std_g"]
FEATURES = EEG_FEATURES + HEART_FEATURES + MOTION_FEATURES


def eeg_window_features(eeg: np.ndarray, fs: float, notch_hz: float) -> dict | None:
    filt = filter_eeg(eeg, fs, notch_hz)
    if np.median(np.std(filt, axis=1)) > MAX_EEG_STD_UV:
        return None  # most channels off the skin or moving: not usable
    freqs, psd = signal.welch(filt, fs=fs, nperseg=int(fs * 2), axis=-1)
    power = {b: _band_power(freqs, psd, lo, hi) + 1e-12 for b, (lo, hi) in BANDS.items()}
    total = sum(power.values())
    out = {}
    for i, c in enumerate(CHANNELS):
        for b in BANDS:
            out[f"{b}_{c}"] = float(np.log10(power[b][i] / total[i]))
    out["alpha_beta_log"] = float(np.median(np.log10(power["alpha"] / power["beta"])))
    # frontal alpha asymmetry: right (AF8) minus left (AF7) log alpha
    out["alpha_asym"] = float(np.log10(power["alpha"][2]) - np.log10(power["alpha"][1]))
    return out


def beat_intervals(ppg: np.ndarray, fs: float) -> np.ndarray:
    """Seconds between heartbeats, from the most periodic pulse channel."""
    ppg = np.atleast_2d(ppg)
    if ppg.shape[-1] < fs * 5:
        return np.array([])
    sos = signal.butter(3, [0.7, 3.5], btype="bandpass", fs=fs, output="sos")
    filt = signal.sosfiltfilt(sos, signal.detrend(ppg, axis=-1), axis=-1)
    freqs, psd = signal.welch(filt, fs=fs, nperseg=min(filt.shape[-1], int(fs * 8)), axis=-1)
    band = (freqs >= 0.7) & (freqs <= 3.5)
    best = int(np.argmax(psd[:, band].max(axis=-1) / (psd[:, band].mean(axis=-1) + 1e-12)))
    x = filt[best]
    peaks, _ = signal.find_peaks(x, distance=max(1, int(fs * 0.33)), prominence=0.3 * np.std(x))
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]
    # missed or doubled beats make huge jumps that look like extreme HRV: keep beats within 25% of the median
    return ibi[np.abs(ibi - np.median(ibi)) < 0.25 * np.median(ibi)] if len(ibi) else ibi


def heart_window_features(ppg: np.ndarray | None, fs: float | None) -> dict:
    out = dict.fromkeys(HEART_FEATURES, np.nan)
    if ppg is None or fs is None:
        return out
    ibi = beat_intervals(ppg, fs)
    if len(ibi) < 5:
        return out
    out["hr_bpm"] = float(60.0 / np.median(ibi))
    out["hrv_rmssd_ms"] = float(np.sqrt(np.mean(np.diff(ibi) ** 2)) * 1000)
    out["hrv_sdnn_ms"] = float(np.std(ibi) * 1000)
    return out


def motion_window_features(acc: np.ndarray | None) -> dict:
    if acc is None or acc.shape[-1] < 5:
        return {"motion_std_g": np.nan}
    return {"motion_std_g": float(np.std(np.linalg.norm(acc, axis=0)))}


def window_features(eeg, eeg_fs, ppg=None, ppg_fs=None, acc=None, notch_hz=60.0) -> dict | None:
    e = eeg_window_features(eeg, eeg_fs, notch_hz)
    if e is None:
        return None
    return {**e, **heart_window_features(ppg, ppg_fs), **motion_window_features(acc)}


def sliding_windows(duration_s: float, window_s: float = WINDOW_S, step_s: float = STEP_S):
    """Start times (s) of windows that fit in a recording of duration_s."""
    t = 0.0
    while t + window_s <= duration_s + 1e-9:
        yield t
        t += step_s


def featurize_recording(eeg, eeg_fs, ppg=None, ppg_fs=None, acc=None, acc_fs=None, notch_hz=60.0):
    """Cut a continuous recording into 10 s windows (5 s step). Returns (list of feature dicts, start times)."""
    rows, starts = [], []
    for t in sliding_windows(eeg.shape[-1] / eeg_fs):
        cut = lambda x, fs: None if x is None else x[..., int(t * fs): int((t + WINDOW_S) * fs)]  # noqa: E731
        f = window_features(cut(eeg, eeg_fs), eeg_fs, cut(ppg, ppg_fs), ppg_fs, cut(acc, acc_fs), notch_hz)
        if f is not None:
            rows.append(f)
            starts.append(t)
    return rows, starts
