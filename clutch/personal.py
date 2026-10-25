"""Personal model: trained automatically at the end of each person's calibration, used live without the headset.

Training table: one row per 0.5 s tick of that person's calibration (rest, stress test, recovery), trained
when the calibration ends; after a study protocol it is retrained on the calibration plus the study blocks
(relax, easy math, hard math, hard colour words), which also have the headset on.
  target    the EEG stress position from the calibration equation, nothing learned:
            per channel (value - rest median) / (stress median - rest median) on log alpha power,
            0 = as calm as at rest, 1 = as stressed as in the test, averaged over the channels that
            responded (weighted by Cohen's d, as in the live score)
  features  heart rate and the face indicators, each as (value - rest median) / rest spread, so the
            model sees changes from this person's calm state. A missing value counts as "at rest" (0).
Model: ridge regression (linear, with a penalty against large weights), fitted with numpy. A linear model
suits a few minutes of one person's data; trees would memorise it.
Live: the same features go through the person's weights; output 0-100 like the other scores.
"""
from __future__ import annotations

import numpy as np

MIN_ROWS = 20            # rows with an EEG target needed to train
MIN_PRESENT = 0.5        # a feature is used only if present in at least half of the rows
RIDGE = 0.1              # penalty per row, on features in rest-spread units


class PersonalModel:
    def __init__(self, features, centers, scales, weights, intercept, report):
        self.features, self.centers, self.scales = features, centers, scales
        self.weights, self.intercept, self.report = weights, intercept, report

    @staticmethod
    def eeg_target(tick: dict, anchors: dict):
        """EEG stress position of one tick: weighted mean of per-channel positions, None without EEG."""
        ps = [(float(np.clip((tick[k] - r) / (s - r), -0.5, 1.5)), w) for k, (r, s, w) in anchors.items() if k in tick]
        return sum(p * w for p, w in ps) / sum(w for _, w in ps) if ps else None

    @classmethod
    def fit(cls, ticks: dict, signals: dict, brain: list, candidates: list, min_sd: dict, default_sd: float):
        """ticks: phase -> list of {signal: value}, every phase is used (features are centred on calib_rest);
        signals: the calibration's per-signal report (finalize). Returns (model or None, report)."""
        used = {k: signals[k] for k in brain if k in signals and signals[k].get("used")}
        weak = not used  # no channel reached a medium effect: fall back to every channel that moved at all
        pool = used or {k: signals[k] for k in brain if k in signals and "stress" in signals[k]}
        anchors = {k: (v["rest"], v["stress"], min(max(v.get("effect", 1.0), 0.1), 2.0))
                   for k, v in pool.items() if v["stress"] != v["rest"]}
        if not anchors:
            return None, {"ok": False, "note": "no EEG target: the headset had no clean signal in both phases"}

        rows = [t for phase_ticks in ticks.values() for t in phase_ticks]
        y = [cls.eeg_target(t, anchors) for t in rows]
        rows = [t for t, v in zip(rows, y) if v is not None]
        y = np.array([v for v in y if v is not None])
        rest = ticks.get("calib_rest", [])
        feats = []
        for k in candidates:
            present = sum(k in t for t in rows)
            r = [t[k] for t in rest if k in t]
            if rows and present / len(rows) >= MIN_PRESENT and len(r) >= 5:
                feats.append(k)
        if len(rows) < MIN_ROWS:
            return None, {"ok": False, "note": f"only {len(rows)} ticks with a clean EEG target (need {MIN_ROWS})"}
        if not feats:
            return None, {"ok": False, "note": "no heart rate or face signal during the calibration"}

        centers = np.array([np.median([t[k] for t in rest if k in t]) for k in feats])
        scales = np.array([max(float(np.std([t[k] for t in rest if k in t])), min_sd.get(k, default_sd)) for k in feats])
        X = np.array([[((t[k] - c) / s) if k in t else 0.0 for k, c, s in zip(feats, centers, scales)] for t in rows])
        xm, ym = X.mean(axis=0), y.mean()
        Xc = X - xm
        w = np.linalg.solve(Xc.T @ Xc + RIDGE * len(y) * np.eye(len(feats)), Xc.T @ (y - ym))
        b = float(ym - xm @ w)
        pred = X @ w + b
        fit_r = float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 and y.std() > 0 else 0.0
        report = {"ok": True, "rows": len(y), "seconds": round(len(y) * 0.5), "features": feats,
                  "weights": {k: round(float(v), 3) for k, v in zip(feats, w)},
                  "phases": [p for p, ts in ticks.items() if ts], "target_channels": sorted(anchors), "weak_target": weak, "fit_r": round(fit_r, 2),
                  "note": ("EEG channels did not clearly respond: target is weak" if weak else
                           "trained on this calibration" if all(p.startswith("calib_") for p in ticks)
                           else "retrained on the calibration and the study blocks")}
        return cls(feats, centers, scales, w, b, report), report

    def predict(self, values: dict):
        """0-100 from heart rate and face values, None when none of the model's features is available."""
        if not any(values.get(k) is not None for k in self.features):
            return None
        x = np.array([((values[k] - c) / s) if values.get(k) is not None else 0.0
                      for k, c, s in zip(self.features, self.centers, self.scales)])
        return float(np.clip(100 * (x @ self.weights + self.intercept), 0, 100))
