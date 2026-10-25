"""Learned stress model: features shared by offline training (tools/train_model.py) and live scoring.

One feature row per moment, from the same windows the live engine uses:
  EEG    log10 theta / alpha / beta power per Muse channel (4 s window), empty when that channel has bad
         contact, muscle tension or head motion
  heart  heart rate in bpm (10 s window), empty when the beat detection quality is below 0.5
  face   MediaPipe indicators (1 s means)
Each feature is expressed relative to the person's own baseline (median of the first BASELINE_S seconds
of the calibration rest phase), then averaged over the last SMOOTH_S seconds. The model sees changes from
the person's own calm state, never absolute values, so it can transfer between people.
"""
from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path

import numpy as np

# Protocol phases with a label (0 = calm, 1 = stressed). Everything else is unlabelled, including
# study_math_easy, the in-between control block.
LABELS = {"calib_rest": 0, "calib_stress": 1, "study_relax": 0, "study_math_hard": 1, "study_stroop_hard": 1}
EEG_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
EEG_FEATURES = [f"{b}log_{ch}" for ch in EEG_CHANNELS for b in ("theta", "alpha", "beta")]
FACE_FEATURES = ["face_brow", "face_press", "face_squint", "face_frown", "face_smile", "blink_rate", "head_motion"]
HEART_FEATURES = ["hr"]
ALL_FEATURES = EEG_FEATURES + HEART_FEATURES + FACE_FEATURES
FEATURE_SETS = {
    "full": EEG_FEATURES + HEART_FEATURES + FACE_FEATURES,   # with the headset (teacher)
    "headset_free": HEART_FEATURES + FACE_FEATURES,          # the goal: watch + webcam
    "eeg": EEG_FEATURES,
    "face": FACE_FEATURES,
    "heart": HEART_FEATURES,
}
BASELINE_S = 45.0       # start of the rest phase used as the personal reference
SMOOTH_S = 10.0         # causal moving average
MIN_BASELINE_VALUES = 5
OUTPUT_TAU_S = 5.0      # smoothing of the live score, same as the rule-based score


def row_features(eeg_feats: dict, hr: dict, motion: dict, face: dict | None, min_hr_quality: float = 0.5) -> dict:
    """Raw (not yet baseline-corrected) feature values for one moment; None = not available."""
    out = {}
    for ch in EEG_CHANNELS:
        c = eeg_feats.get("channels", {}).get(ch)
        clean = c is not None and c["q"] != "bad" and not c.get("muscle") and not motion.get("flag")
        for b in ("theta", "alpha", "beta"):
            out[f"{b}log_{ch}"] = c.get(f"{b}log") if clean else None
    out["hr"] = hr.get("bpm") if (hr.get("quality") or 0) >= min_hr_quality else None
    for k in FACE_FEATURES:
        out[k] = (face or {}).get(k)
    return out


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

    The baseline is taken from the first BASELINE_S seconds of each calibration rest phase, exactly as in
    training, so the model only works after a calibration (headset or not).
    """

    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.idx = [ALL_FEATURES.index(k) for k in self.features]
        self.baseline_s = bundle.get("baseline_s", BASELINE_S)
        self.smooth_s = bundle.get("smooth_s", SMOOTH_S)
        self.kind, self.scale = bundle.get("kind", "label"), bundle.get("scale")
        self.info = {k: bundle.get(k) for k in ("name", "kind", "trained_on", "cv_auc", "validation")}
        self._rest_t0 = None
        self._base_rows = []
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
        return cls(joblib.load(path))

    def update(self, dt: float, phase: str, row: dict):
        """Feed one moment, dt seconds after the previous one; returns the stress score 0-100, or None
        before the baseline exists. Time is counted in ticks so that simulations can run faster than real time."""
        self._t += dt
        t, v = self._t, to_vector(row)
        if phase == "calib_rest":
            if self._last_phase != "calib_rest":  # a new calibration starts: new baseline
                self._rest_t0, self._base_rows = t, []
            if t - self._rest_t0 <= self.baseline_s:
                self._base_rows.append(v)
            elif self._base_rows:
                self.baseline = baseline_of(self._base_rows)
                self._base_rows = []
                self._recent.clear()
        self._last_phase = phase
        if self.baseline is None:
            return None
        self._recent.append((t, v - self.baseline))
        while self._recent and self._recent[0][0] <= t - self.smooth_s:
            self._recent.popleft()
        x = window_mean([r for _, r in self._recent])[self.idx]
        if np.all(np.isnan(x)):
            return None
        if self.kind == "teacher":  # regressor imitating the EEG index, scaled between rest (0) and stress (1)
            r, s = self.scale
            pos = (float(self.model.predict(x.reshape(1, -1))[0]) - r) / (s - r)
        else:
            pos = float(self.model.predict_proba(x.reshape(1, -1))[0, 1])
        # Trees trained on little data give near-0 or near-1 outputs: smooth like the app's own score (~5 s)
        raw = float(np.clip(100 * pos, 0, 100))
        self._out = raw if self._out is None else self._out + (1 - np.exp(-dt / OUTPUT_TAU_S)) * (raw - self._out)
        return round(self._out, 1)
