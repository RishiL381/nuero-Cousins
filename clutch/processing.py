"""Signal processing for Muse 2 data: EEG band powers, contact quality, heart rate, motion.

All functions are pure (numpy in, numbers out) so they can be tested on recorded or
synthetic data without a headset.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 44.0),
}
# Relative 30-44 Hz power above which a channel window is treated as muscle-contaminated. Set on the first
# real session: clean temporal channels sat around 0.01, a tense forehead channel around 0.13.
MUSCLE_GAMMA = 0.10


def _band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Integrate a PSD (channels x freqs) between lo and hi Hz."""
    mask = (freqs >= lo) & (freqs < hi)
    df = freqs[1] - freqs[0]
    return psd[..., mask].sum(axis=-1) * df


def filter_eeg(eeg: np.ndarray, fs: float, notch_hz: float = 60.0) -> np.ndarray:
    """Detrend, band-pass 1-45 Hz and notch mains noise. eeg: (channels, samples) in uV."""
    x = signal.detrend(eeg, axis=-1)
    sos = signal.butter(4, [1.0, 45.0], btype="bandpass", fs=fs, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=-1)
    if notch_hz and notch_hz < fs / 2:
        b, a = signal.iirnotch(notch_hz, 30.0, fs)
        x = signal.filtfilt(b, a, x, axis=-1)
    return x


def channel_quality(eeg_raw: np.ndarray, eeg_filt: np.ndarray, fs: float, notch_hz: float = 60.0) -> list[dict]:
    """Heuristic contact quality per channel: 'good', 'fair' or 'bad'.

    - A well-seated Muse electrode gives roughly 2-40 uV std after 1-45 Hz filtering.
    - Off-skin or loose electrodes show huge amplitudes and strong mains (50/60 Hz) noise.
    - A flat line (std < 0.5 uV) means no signal at all.
    """
    out = []
    nperseg = min(eeg_raw.shape[-1], int(fs * 2))
    freqs, psd = signal.welch(signal.detrend(eeg_raw, axis=-1), fs=fs, nperseg=nperseg, axis=-1)
    line = _band_power(freqs, psd, notch_hz - 2, notch_hz + 2) if notch_hz < fs / 2 else np.zeros(len(eeg_raw))
    brain = _band_power(freqs, psd, 1.0, 45.0) + 1e-12
    for i in range(eeg_raw.shape[0]):
        std = float(np.std(eeg_filt[i]))
        ptp = float(np.ptp(eeg_filt[i]))
        line_ratio = float(line[i] / brain[i])
        saturated = float(np.mean(np.abs(eeg_raw[i]) >= 990))  # Muse range is about +-1000 uV
        if std < 0.5 or std > 100 or ptp > 800 or line_ratio > 2.0 or saturated > 0.01:
            q = "bad"
        elif std > 45 or ptp > 400 or line_ratio > 0.5:
            q = "fair"
        else:
            q = "good"
        # Plain-language reason, most specific first
        if std < 0.5:
            why = "flat: no signal"
        elif saturated > 0.01:
            why = "saturating: not touching skin"
        elif line_ratio > 0.5:
            why = "electrical noise: not touching skin"
        elif q == "bad":
            why = "signal far too large: loose contact or moving"
        elif q == "fair":
            why = "borderline: press it on, keep still"
        else:
            why = "good contact"
        out.append({"q": q, "std": round(std, 1), "ptp": round(ptp, 1), "line": round(line_ratio, 2),
                    "sat": round(saturated, 3), "why": why})
    return out


def eeg_features(eeg_raw: np.ndarray, fs: float, names: list[str], notch_hz: float = 60.0) -> dict:
    """Band powers per channel and the calm feature log10(alpha/beta), median over usable channels.

    Blinks mostly live below 4 Hz and barely touch alpha/beta, so we do not reject windows
    for them. Channels with bad contact are excluded.
    """
    filt = filter_eeg(eeg_raw, fs, notch_hz)
    quality = channel_quality(eeg_raw, filt, fs, notch_hz)
    nperseg = min(filt.shape[-1], int(fs * 2))
    freqs, psd = signal.welch(filt, fs=fs, nperseg=nperseg, axis=-1)
    powers = {band: _band_power(freqs, psd, lo, hi) for band, (lo, hi) in BANDS.items()}

    channels = {}
    usable_ratios, usable_idx = [], []
    for i, name in enumerate(names):
        ch = dict(quality[i])
        total = sum(powers[b][i] for b in BANDS) + 1e-12
        ch.update({b: round(float(powers[b][i] / total), 3) for b in BANDS})  # relative power
        # Alpha share of 4-30 Hz power: leaves out <4 Hz, where blinks and eye movements put most of their
        # energy, so a blink on the forehead channels does not drag the alpha value down.
        mid = powers["theta"][i] + powers["alpha"][i] + powers["beta"][i] + 1e-12
        ch["alphashare"] = round(float(powers["alpha"][i] / mid), 4)
        # Absolute alpha power, log10(uV^2): blinks add almost nothing at 8-12 Hz, so this is the most
        # blink-robust alpha measure when a channel is compared with itself within one session.
        ch["alphalog"] = round(float(np.log10(powers["alpha"][i] + 1e-12)), 4)
        ch["thetalog"] = round(float(np.log10(powers["theta"][i] + 1e-12)), 4)  # same, for model training
        ch["betalog"] = round(float(np.log10(powers["beta"][i] + 1e-12)), 4)
        # Muscle (EMG) check: on scalp EEG, 30-44 Hz power comes mostly from forehead and jaw muscles, which
        # also spill into beta and alpha (Goncharova et al., 2003). Such a window says nothing about the brain.
        ch["muscle"] = bool(ch["gamma"] > MUSCLE_GAMMA)
        if ch["muscle"] and ch["q"] != "bad":
            ch["why"] = "muscle tension: relax forehead and jaw"
        channels[name] = ch
        if quality[i]["q"] != "bad":
            usable_ratios.append(np.log10((powers["alpha"][i] + 1e-12) / (powers["beta"][i] + 1e-12)))
            usable_idx.append(i)

    feats = {"channels": channels, "usable": len(usable_idx), "ratio_log": None}
    if usable_idx:
        feats["ratio_log"] = float(np.median(usable_ratios))
        for b in BANDS:
            rel = [channels[names[i]][b] for i in usable_idx]
            feats[b] = round(float(np.median(rel)), 3)
    return feats


def heart_rate(ppg: np.ndarray, fs: float) -> dict:
    """Heart rate from Muse PPG (channels x samples). Returns bpm and a 0-1 quality.

    The Muse 2 has three optical channels (ambient, infrared, red). We pick the channel with
    the most periodic signal in the 0.7-3.5 Hz band (42-210 bpm), detect beats, and check that
    the beat-to-beat estimate agrees with the spectral peak.
    """
    if ppg is None or ppg.shape[-1] < fs * 5:
        return {"bpm": None, "quality": 0.0}
    sos = signal.butter(3, [0.7, 3.5], btype="bandpass", fs=fs, output="sos")
    filt = signal.sosfiltfilt(sos, signal.detrend(ppg, axis=-1), axis=-1)
    nperseg = min(filt.shape[-1], int(fs * 8))
    freqs, psd = signal.welch(filt, fs=fs, nperseg=nperseg, axis=-1)
    band = (freqs >= 0.7) & (freqs <= 3.5)
    peakiness = psd[:, band].max(axis=-1) / (psd[:, band].mean(axis=-1) + 1e-12)
    best = int(np.argmax(peakiness))
    spec_bpm = float(freqs[band][np.argmax(psd[best, band])] * 60)

    x = filt[best]
    peaks, _ = signal.find_peaks(x, distance=max(1, int(fs * 0.33)), prominence=0.3 * np.std(x))
    ibi = np.diff(peaks) / fs
    ibi = ibi[(ibi > 0.33) & (ibi < 1.5)]
    if len(ibi) < 4:
        return {"bpm": None, "quality": 0.0, "spec_bpm": round(spec_bpm, 1)}

    # A missed or extra peak gives one wildly wrong interval; judge quality by the share of intervals that
    # agree with the median instead of their overall spread, which one bad interval ruins.
    med = float(np.median(ibi))
    good = np.abs(ibi - med) < 0.15 * med
    if good.sum() < 3:
        return {"bpm": None, "quality": 0.0, "spec_bpm": round(spec_bpm, 1)}
    bpm = float(60.0 / np.mean(ibi[good]))
    quality = max(0.0, (float(good.mean()) - 0.5) / 0.5)
    if abs(bpm - spec_bpm) > 12:  # beat detector and spectrum disagree: likely noise
        quality *= 0.4
    return {"bpm": round(bpm, 1), "quality": round(quality, 2), "spec_bpm": round(spec_bpm, 1)}


def motion_level(accel: np.ndarray) -> dict:
    """Head motion from accelerometer (3 x samples, in g). Std of the magnitude."""
    if accel is None or accel.shape[-1] < 5:
        return {"level": None, "flag": False}
    mag = np.linalg.norm(accel, axis=0)
    level = float(np.std(mag))
    return {"level": round(level, 4), "flag": level > 0.05}
