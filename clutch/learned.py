"""Learned stress model: features shared by offline training (tools/train_model.py) and live scoring.

One input per moment, from the same windows the live engine uses: the spectrum of each Muse channel
(4 s window), heart rate (+ variability from a watch that sends beat intervals) and the face indicators.
Features computed from it:
  brain   log10 theta / alpha / low-beta power per channel, in bands set on the person's individual alpha
          frequency (IAF, Klimesch 1999), estimated on the start of the rest phase. Empty when that channel
          has bad contact, head motion or muscle tension.
  muscle  log10 30-44 Hz power per channel: forehead / jaw / temple muscle tension (Goncharova et al., 2003).
          Kept apart from the brain so the models say which one they rely on. Empty on bad contact or motion.
  heart   heart rate in bpm (quality >= 0.5), RMSSD in ms when available
  face    MediaPipe indicators (1 s means)
Each feature is expressed relative to the person's own baseline (median of the first BASELINE_S seconds
of the calibration rest phase), then averaged over the last SMOOTH_S seconds. The model sees changes from
the person's own calm state, never absolute values, so it can transfer between people.
"""
from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path

import numpy as np

from .processing import IAF_DEFAULT, band_logs, individual_alpha_frequency

# Protocol phases with a label (0 = calm, 1 = stressed). Everything else is unlabelled, including
# study_math_easy, the in-between control block.
LABELS = {"calib_rest": 0, "calib_stress": 1, "study_relax": 0, "study_math_hard": 1, "study_stroop_hard": 1}
EEG_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
BRAIN_FEATURES = [f"{b}_iaf_{ch}" for ch in EEG_CHANNELS for b in ("theta", "alpha", "beta")]
MUSCLE_FEATURES = [f"muscle_{ch}" for ch in EEG_CHANNELS]
FACE_FEATURES = ["face_brow", "face_press", "face_squint", "face_frown", "face_smile", "blink_rate", "head_motion"]
HEART_FEATURES = ["hr", "hrv"]  # hrv: RMSSD from a watch or strap, when it sends beat-to-beat intervals
ALL_FEATURES = BRAIN_FEATURES + MUSCLE_FEATURES + HEART_FEATURES + FACE_FEATURES
FEATURE_SETS = {
    "full": ALL_FEATURES,                                     # everything the headset, watch and camera give
    "headset": BRAIN_FEATURES + MUSCLE_FEATURES,              # all the Muse gives
    "brain": BRAIN_FEATURES,                                  # the brain alone (teaches the body)
    "muscle": MUSCLE_FEATURES,
    "headset_free": HEART_FEATURES + FACE_FEATURES,           # the goal: watch + webcam
    "face": FACE_FEATURES,
    "heart": HEART_FEATURES,
}
WITH_HEADSET_SETS = ("brain", "headset", "full")             # candidates for the live "with headset" model
BASELINE_S = 45.0       # start of the rest phase used as the personal reference
SMOOTH_S = 10.0         # causal moving average
MIN_BASELINE_VALUES = 5
OUTPUT_TAU_S = 5.0      # smoothing of the live score, same as the rule-based score


def row_inputs(eeg_feats: dict, hr: dict, motion: dict, face: dict | None, min_hr_quality: float = 0.5) -> dict:
    """What one moment gives, before features: spectra per channel (None on bad contact or motion), the
    muscle flag per channel, heart and face values."""
    psd = eeg_feats.get("psd") or {}
    spectra, muscle = {}, {}
    for ch in EEG_CHANNELS:
        c = eeg_feats.get("channels", {}).get(ch)
        ok = c is not None and c["q"] != "bad" and not motion.get("flag") and ch in psd
        spectra[ch] = psd[ch] if ok else None
        muscle[ch] = bool(c and c.get("muscle"))
    return {"freqs": psd.get("freqs"), "psd": spectra, "muscle": muscle,
            "hr": hr.get("bpm") if (hr.get("quality") or 0) >= min_hr_quality else None, "hrv": hr.get("rmssd"),
            **{k: (face or {}).get(k) for k in FACE_FEATURES}}


def estimate_iaf(inputs: list[dict]) -> float:
    """Individual alpha frequency from resting inputs: median of the channels with a clear alpha peak
    (muscle-free windows only), clamped to 8-13 Hz; 10 Hz when no channel shows a peak."""
    freqs = next((x["freqs"] for x in inputs if x.get("freqs") is not None), None)
    if freqs is None:
        return IAF_DEFAULT
    peaks = []
    for ch in EEG_CHANNELS:
        spectra = [x["psd"][ch] for x in inputs if x["psd"].get(ch) is not None and not x["muscle"][ch]]
        f = individual_alpha_frequency(spectra, np.asarray(freqs))
        if f is not None:
            peaks.append(f)
    return float(np.clip(np.median(peaks), 8, 13)) if peaks else IAF_DEFAULT


def features_from(x: dict, iaf: float) -> dict:
    """Feature values for one moment (None = not available), with bands set on the IAF."""
    out = {}
    for ch in EEG_CHANNELS:
        spectrum = x["psd"].get(ch)
        bands = band_logs(np.asarray(spectrum), np.asarray(x["freqs"]), iaf) if spectrum is not None else None
        brain_ok = bands is not None and not x["muscle"][ch]
        for b in ("theta", "alpha", "beta"):
            out[f"{b}_iaf_{ch}"] = bands[b] if brain_ok else None
        out[f"muscle_{ch}"] = bands["muscle"] if bands is not None else None
    for k in HEART_FEATURES + FACE_FEATURES:
        out[k] = x.get(k)
    return out


def brain_values(x: dict, iaf: float) -> dict:
    """Brain signals of the live calibration rule, {band}_{channel} (theta, alpha, beta in IAF bands), only
    for the channels that are clean this moment."""
    f = features_from(x, iaf)
    return {f"{b}_{ch}": f[f"{b}_iaf_{ch}"] for ch in EEG_CHANNELS for b in ("theta", "alpha", "beta")
            if f[f"{b}_iaf_{ch}"] is not None}


def to_vector(row: dict, names=ALL_FEATURES) -> np.ndarray:
    return np.array([np.nan if row.get(k) is None else float(row[k]) for k in names])


def baseline_of(vectors) -> np.ndarray:
    """Per-feature median of the baseline rows; NaN where fewer than MIN_BASELINE_VALUES were clean."""
    X = np.vstack(vectors)
    ok = np.sum(~np.isnan(X), axis=0) >= MIN_BASELINE_VALUES
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X, axis=0)
    return np.where(ok, med, np.nan)


def window_mean(vectors) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.vstack(vectors), axis=0)


class LiveScorer:
    """Scores stress live with a model saved by tools/train_model.py.

    The baseline (and the IAF) come from the first BASELINE_S seconds of each calibration rest phase, exactly
    as in training, so the model only works after a calibration (headset or not).
    """

    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.idx = [ALL_FEATURES.index(k) for k in self.features]
        self.baseline_s = bundle.get("baseline_s", BASELINE_S)
        self.smooth_s = bundle.get("smooth_s", SMOOTH_S)
        self.kind, self.scale = bundle.get("kind", "label"), bundle.get("scale")
        self.info = {k: bundle.get(k) for k in ("name", "kind", "trained_on", "cv_auc", "validation")}
        self.iaf = None
        self._rest_t0 = None
        self._base_inputs = []
        self.baseline = None
        self._recent = deque()
        self._last_phase = None
        self._t = 0.0
        self._out = None

    @classmethod
    def load(cls, path: Path):
        if not Path(path).exists():
            return None
        import joblib  # only needed when a trained model exists
        bundle = joblib.load(path)
        missing = [k for k in bundle["features"] if k not in ALL_FEATURES]
        if missing:  # trained with an older feature set: retrain (step 4 of the app)
            print(f"{Path(path).name}: trained with older features ({', '.join(missing[:3])}...), retrain to use it")
            return None
        return cls(bundle)

    def update(self, dt: float, phase: str, inputs: dict):
        """Feed one moment (row_inputs), dt seconds after the previous one; returns the stress score 0-100, or
        None before the baseline exists. Time is counted in ticks so simulations can run faster than real time."""
        self._t += dt
        t = self._t
        if phase == "calib_rest":
            if self._last_phase != "calib_rest":  # a new calibration starts: new baseline
                self._rest_t0, self._base_inputs = t, []
            if t - self._rest_t0 <= self.baseline_s:
                self._base_inputs.append(inputs)
            elif self._base_inputs:
                self.iaf = estimate_iaf(self._base_inputs)
                self.info["iaf"] = self.iaf
                self.baseline = baseline_of([to_vector(features_from(x, self.iaf)) for x in self._base_inputs])
                self._base_inputs = []
                self._recent.clear()
        self._last_phase = phase
        if self.baseline is None:
            return None
        self._recent.append((t, to_vector(features_from(inputs, self.iaf)) - self.baseline))
        while self._recent and self._recent[0][0] <= t - self.smooth_s:
            self._recent.popleft()
        x = window_mean([r for _, r in self._recent])[self.idx]
        if np.all(np.isnan(x)):
            return None
        if self.kind == "teacher":  # regressor imitating the brain model, scaled between rest (0) and stress (1)
            r, s = self.scale
            pos = (float(self.model.predict(x.reshape(1, -1))[0]) - r) / (s - r)
        else:
            pos = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        # Trees trained on little data give near-0 or near-1 outputs: smooth like the app's own score (~5 s)
        raw = float(np.clip(100 * pos, 0, 100))
        self._out = raw if self._out is None else self._out + (1 - np.exp(-dt / OUTPUT_TAU_S)) * (raw - self._out)
        return round(self._out, 1)
