"""Real-time loop: reads the source, computes features, turns them into a 0-100 stress score,
and logs sessions to disk.

Stress score: two-anchor personal calibration (protocol and references: docs/methods.md)
-----------------------------------------------------------------------------------------
Calibration records each signal in known states:
  calib_rest      low-demand "vanilla" baseline task, eyes open       -> CALM anchor
                  (Jennings et al., 1992; Barry et al., 2007)
  calib_stress    serial subtraction from 1022 by 13 (TSST, Kirschbaum et al., 1993) with an adaptive
                  time limit and social comparison (MIST, Dedovic et al., 2005)   -> STRESS anchor
  calib_recovery  quiet rest after the stressor (reported, not an anchor)
The client sends phase "calibration_check" when the protocol is over; the calibration is computed then.

For each signal (EEG calm feature log10(alpha/beta), heart rate) we compute Cohen's d between the
stress and rest samples, signed so that positive = moved the expected way (EEG ratio down, heart up).
A signal is used only with at least a medium effect (d >= 0.5, Cohen, 1988), weighted by d.

Live, each used signal is placed on its own calm -> stress line:
  position = (value - rest) / (stress - rest)      0 = as calm as at rest, 1 = as stressed as in the test
score = 100 * weighted mean position, clipped to 0-100, smoothed over ~5 s. If no signal responded,
a weaker fallback uses distance from the rest anchor only (3 rest SDs = 100), flagged low confidence.
Note: samples come from overlapping windows, so d describes the size of the change, not significance.
"""
from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .learned import LiveScorer, brain_values, estimate_iaf, row_inputs
from .personal import PersonalModel
from .processing import eeg_features, heart_rate, motion_level
from .watch import WatchHR

CALIB_PHASES = ("calib_rest", "calib_stress", "calib_recovery")  # phases whose data are collected
PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"  # saved calibrations, per person (not in git)


def _json_default(o):
    """numpy numbers and booleans in a calibration -> plain JSON."""
    return o.item() if hasattr(o, "item") else str(o)

# Study protocol blocks (headset on): added to the personal model's training data, retrained at study_done
STUDY_PHASES = ("study_relax", "study_math_easy", "study_math_hard", "study_stroop_hard")
# Expected direction under stress: +1 goes up, -1 goes down, 0 = learned per person from the calibration.
#   <band>_<ch>  brain: log10 theta, alpha or low-beta power of ONE Muse channel, compared with itself between
#                phases, in bands set on the person's individual alpha frequency (IAF, Klimesch 1999) measured
#                on the calibration rest; muscle-free, clean windows only. Direction learned: on P01 session 2,
#                alpha rose under stress on most channels, so "alpha must drop" rejected the whole EEG, while
#                theta + alpha + beta with a learned direction separated the unseen study blocks (AUC 0.86,
#                docs/methods.md). Alpha suppression remains the average finding (Vanhollebeke et al., 2022).
#   hr           heart rate in bpm (Muse PPG now, a Garmin watch later), up with stress
#   face_*       MediaPipe blendshape means (0-1): brows lowered, lips pressed, eyes squinted, frown, smile
#   blink_rate   blinks per minute, head_motion: head movement (face widths per second)
EEG_CHANNELS = ("TP9", "AF7", "AF8", "TP10")
BRAIN_BANDS = ("theta", "alpha", "beta")
SIGNALS = {
    **{f"{b}_{ch}": 0 for ch in EEG_CHANNELS for b in BRAIN_BANDS}, "hr": +1, "hrv": -1,
    "face_brow": +1, "face_press": +1, "face_squint": +1, "face_frown": +1, "face_smile": 0,
    "blink_rate": 0, "head_motion": +1,
}
# Each modality counts once in the score, whatever its number of signals (so 4 EEG channels or 7 face
# indicators cannot outvote the heart). Inside a group, signals are averaged weighted by their effect size.
GROUPS = {
    "brain": [f"{b}_{ch}" for ch in EEG_CHANNELS for b in BRAIN_BANDS],
    "heart": ["hr", "hrv"],
    "face": ["face_brow", "face_press", "face_squint", "face_frown", "face_smile", "blink_rate", "head_motion"],
}
HEADSET_GROUPS = {"brain"}  # left out of the "without headset" score
HEADSET_SIGNALS = set(GROUPS["brain"])
MIN_SD = {"hr": 2.0, "hrv": 3.0, "blink_rate": 2.0, "head_motion": 0.005, **{s: 0.05 for s in GROUPS["brain"]}}  # brain: log10 units
MIN_SD_FACE = 0.02


@dataclass
class Config:
    eeg_window_s: float = 4.0
    ppg_window_s: float = 10.0
    motion_window_s: float = 2.0
    tick_s: float = 0.5
    smooth_tau_s: float = 5.0
    min_effect: float = 0.5            # Cohen's d needed to use a signal (medium effect, Cohen 1988)
    check_min_auc: float = 0.75        # flip check: second-half stress must outrank second-half rest...
    check_min_gap: float = 0.2         # ...by a visible gap on the 0-1 scale
    min_hr_quality: float = 0.5
    face_max_age_s: float = 2.0        # face features older than this are ignored
    notch_hz: float = 60.0             # mains frequency: 60 Hz in the US, 50 Hz in Europe


def _empty_samples():
    return {p: {k: [] for k in SIGNALS} for p in CALIB_PHASES}


def _empty_ticks():
    return {p: [] for p in CALIB_PHASES}


def _auc(pos, neg) -> float:
    """Probability that a random value of pos is above a random value of neg (ties count half)."""
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


@dataclass
class StressModel:
    cfg: Config
    calibration: dict | None = None
    _samples: dict = field(default_factory=_empty_samples)
    _ticks: dict = field(default_factory=_empty_ticks)   # same values, kept together per tick (flip check)
    _smoothed: dict = field(default_factory=dict)
    _dropped: dict = field(default_factory=dict)  # signal -> {reason: windows rejected during calibration}
    personal: PersonalModel | None = None  # heart + face -> EEG stress, trained on this person's calibration
    brain_rule: float | None = None  # last brain-only score of the calibration rule (0-100), set by score()
    _calib_ticks: dict = field(default_factory=dict)  # the last good calibration's ticks, per phase
    _study_ticks: dict = field(default_factory=dict)  # study phase -> ticks since the last calibration
    iaf: float | None = None  # individual alpha frequency of the last good calibration (Hz)
    _inputs: dict = field(default_factory=_empty_ticks)  # spectra per tick, aligned with _ticks (brain recompute)

    @property
    def baseline(self):  # kept for callers that only need to know whether we are calibrated
        return self.calibration

    def collect(self, phase, values: dict, dropped: dict | None = None, inputs: dict | None = None):
        """values: signal name -> current value (None or missing = not available this tick).
        dropped: signal name -> why its value was rejected this tick ("contact", "motion", "muscle").
        inputs: this tick's spectra (learned.row_inputs), kept during calibration so that the brain signals can
        be recomputed with the IAF once the rest is over."""
        if phase in STUDY_PHASES and self.calibration is not None:
            self._study_ticks.setdefault(phase, []).append(
                {k: float(x) for k, x in values.items() if k in SIGNALS and x is not None})
        if phase not in self._samples:
            return
        tick = {}
        for name, x in values.items():
            if name in SIGNALS and x is not None:
                self._samples[phase][name].append(float(x))
                tick[name] = float(x)
        for name in values:
            if name in SIGNALS:
                counts = self._dropped.setdefault(name, {})
                counts["ticks"] = counts.get("ticks", 0) + 1
        for name, why in (dropped or {}).items():
            counts = self._dropped.setdefault(name, {})
            counts[why] = counts.get(why, 0) + 1
        self._ticks[phase].append(tick)
        self._inputs[phase].append(inputs)

    def _brain_with_iaf(self):
        """Brain signals need the person's IAF, known only once the rest is recorded: estimate it on the
        calibration rest, then recompute every calibration tick's brain values with it. Returns the IAF, or
        None when no spectra were kept (values then stay as collected)."""
        rest = [x for x in self._inputs["calib_rest"] if x is not None]
        if not rest:
            return None
        iaf, brain = estimate_iaf(rest), GROUPS["brain"]
        for phase, ticks in self._ticks.items():
            for tick, x in zip(ticks, self._inputs[phase]):
                if x is not None:
                    for k in brain:
                        tick.pop(k, None)
                    tick.update(brain_values(x, iaf))
            for k in brain:
                self._samples[phase][k] = [t[k] for t in ticks if k in t]
        return iaf

    def split_half_check(self) -> dict:
        """Flip check: does this calibration point the right way for this person?
        Anchors are rebuilt from the FIRST half of rest and stress only (same rules as finalize: a signal
        needs a medium effect, weighted by Cohen's d), then the combined score is computed on the SECOND
        half, which the anchors never saw. Passes when second-half stress ranks above second-half rest
        (AUC >= check_min_auc) with a visible gap (median difference >= check_min_gap on the 0-1 scale).
        On EPIStress (17 people), demos worked for 90% of people whose check passed vs 82% overall."""
        rest, stress = self._ticks["calib_rest"], self._ticks["calib_stress"]
        if len(rest) < 20 or len(stress) < 20:
            return {"passed": None, "note": "calibration too short to check"}
        hr_, hs = len(rest) // 2, len(stress) // 2
        anchors = {}
        for name, expected in SIGNALS.items():
            r = [t[name] for t in rest[:hr_] if name in t]
            s = [t[name] for t in stress[:hs] if name in t]
            if len(r) < 5 or len(s) < 5:
                continue
            rm, sm = float(np.median(r)), float(np.median(s))
            pooled = max(math.sqrt((np.var(r) + np.var(s)) / 2), MIN_SD.get(name, MIN_SD_FACE))
            direction = expected if expected != 0 else (1 if sm >= rm else -1)
            d = direction * (sm - rm) / pooled
            if d >= self.cfg.min_effect and sm != rm:
                anchors[name] = (rm, sm, min(d, 2.0))
        if not anchors:
            return {"passed": False, "note": "no signal reacted in the first half of the calibration"}

        def position(t):
            ps = [(float(np.clip((t[k] - rm) / (sm - rm), -0.5, 1.5)), w) for k, (rm, sm, w) in anchors.items() if k in t]
            return sum(p * w for p, w in ps) / sum(w for _, w in ps) if ps else None

        pr = [p for p in map(position, rest[hr_:]) if p is not None]
        ps = [p for p in map(position, stress[hs:]) if p is not None]
        if len(pr) < 5 or len(ps) < 5:
            return {"passed": False, "note": "not enough clean signal in the second half"}
        auc, gap = _auc(ps, pr), float(np.median(ps) - np.median(pr))
        passed = auc >= self.cfg.check_min_auc and gap >= self.cfg.check_min_gap
        note = ("second half sorted correctly" if passed else
                "score points the wrong way on the second half" if gap < 0 else
                "second half not clearly separated")
        return {"passed": passed, "auc": round(auc, 2), "gap": round(gap, 2), "signals": sorted(anchors), "note": note}

    def finalize(self) -> dict:
        iaf = self._brain_with_iaf()
        check = self.split_half_check()
        rest, stress = self._samples["calib_rest"], self._samples["calib_stress"]
        signals = {}
        for name, expected in SIGNALS.items():
            r, s = rest[name], stress[name]
            floor = MIN_SD.get(name, MIN_SD_FACE)
            info = {"n_rest": len(r), "n_stress": len(s), "used": False}
            counts = dict(self._dropped.get(name, {}))
            ticks = counts.pop("ticks", 0)
            if counts and ticks:  # % of calibration windows rejected, by reason
                info["dropped"] = {why: round(100 * n / ticks) for why, n in counts.items()}
            if len(r) >= 10:
                info["rest"] = float(np.median(r))
                info["rest_sd"] = max(float(np.std(r)), floor)
            if len(r) >= 10 and len(s) >= 10:
                info["stress"] = float(np.median(s))
                pooled = max(math.sqrt((np.var(r) + np.var(s)) / 2), floor)
                d_raw = (info["stress"] - info["rest"]) / pooled
                # learned direction: whichever way it moved for this person
                info["direction"] = expected if expected != 0 else (1 if d_raw >= 0 else -1)
                d = info["direction"] * d_raw
                info["effect"] = d
                info["used"] = d >= self.cfg.min_effect
                info["note"] = ("large response" if d >= 0.8 else
                                "medium response" if d >= 0.5 else
                                "small response, not used" if d >= 0.2 else
                                "changed the opposite way, not used" if d <= -0.2 else
                                "no clear change, not used")
                rec = self._samples["calib_recovery"][name]
                if len(rec) >= 10 and info["stress"] != info["rest"]:  # how far back toward calm afterwards
                    pos = (float(np.median(rec)) - info["rest"]) / (info["stress"] - info["rest"])
                    info["recovered_pct"] = round(100 * (1 - pos), 0)
            elif len(r) or len(s) or "dropped" in info:
                worst = max(info.get("dropped", {"": 0}).items(), key=lambda kv: kv[1])[0]
                info["note"] = {"muscle": "not enough clean signal: muscle tension (forehead, jaw)",
                                "contact": "not enough clean signal: poor contact",
                                "motion": "not enough clean signal: head movement"}.get(worst, "not enough clean signal")
            else:
                info["note"] = "not recorded"
            signals[name] = info

        used = [k for k, v in signals.items() if v["used"]]
        has_rest = [k for k, v in signals.items() if "rest" in v and SIGNALS[k] != 0]
        if used:
            mode = "anchored"
            confidence = "good" if any(signals[k]["effect"] >= 0.8 for k in used) else "ok"
        elif has_rest:
            mode, confidence = "rest_only", "low"
        else:
            mode, confidence = None, None

        cal = {"ok": mode is not None, "mode": mode, "confidence": confidence, "used": used, "check": check,
               "iaf": iaf,
               "used_without_headset": [k for k in used if k not in HEADSET_SIGNALS],
               "signals": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                           for k, v in signals.items() if v["n_rest"] or v["n_stress"] or "dropped" in v}}
        if cal["ok"]:  # personal model: heart + face learn this person's EEG stress position
            model, cal["personal"] = self._fit_personal(self._ticks, signals)
            self._calib_ticks, self._study_ticks = self._ticks, {}
        self._samples, self._dropped = _empty_samples(), {}
        self._ticks, self._inputs = _empty_ticks(), _empty_ticks()
        if cal["ok"]:
            self.calibration = cal
            if iaf is not None:
                self.iaf = iaf
            self._signals = signals
            self.personal = model
            self._smoothed = {}
        return cal

    @staticmethod
    def _fit_personal(ticks, signals):
        return PersonalModel.fit(ticks, signals, GROUPS["brain"], [k for k in SIGNALS if k not in HEADSET_SIGNALS],
                                 MIN_SD, MIN_SD_FACE)

    def state(self) -> dict:
        """What a saved profile needs to score live again without recalibrating."""
        return {"calibration": self.calibration, "signals": self._signals, "iaf": self.iaf,
                "personal": self.personal.to_dict() if self.personal else None}

    def restore(self, saved: dict):
        """Back to a saved profile's calibration: EEG anchors and IAF (the teacher) and the personal model."""
        self.calibration, self._signals, self.iaf = saved["calibration"], saved["signals"], saved.get("iaf")
        self.personal = PersonalModel.from_dict(saved["personal"]) if saved.get("personal") else None
        self._smoothed, self._calib_ticks, self._study_ticks = {}, {}, {}

    def retrain_personal(self):
        """End of a study protocol: retrain the personal model on the calibration plus the study blocks.
        Returns the updated calibration, or None when there is nothing new to train on."""
        if self.calibration is None or not self._study_ticks:
            return None
        model, report = self._fit_personal({**self._calib_ticks, **self._study_ticks}, self._signals)
        self._study_ticks = {}
        if model is None:  # keep the calibration-only model rather than losing it
            return None
        self.personal = model
        self.calibration["personal"] = report
        return self.calibration

    def personal_score(self, values: dict, dt):
        """Personal model's 0-100 score from heart rate and face, None until it is trained."""
        if self.personal is None:
            return None
        return self._smooth("personal", self.personal.predict(values), dt)

    def _position(self, name, x):
        """Where a value sits for this person: 0 = calm (rest anchor), 1 = stressed (stress anchor)."""
        info, mode = self._signals[name], self.calibration["mode"]
        if mode == "anchored" and info["used"]:
            return float(np.clip((x - info["rest"]) / (info["stress"] - info["rest"]), -0.5, 1.5)), min(info["effect"], 2.0)
        if mode == "rest_only" and "rest" in info and SIGNALS[name] != 0:
            return float(np.clip(SIGNALS[name] * (x - info["rest"]) / info["rest_sd"] / 3, -0.5, 1.5)), 1.0
        return None, 0.0

    def _smooth(self, key, raw, dt):
        prev = self._smoothed.get(key)
        if raw is None:
            return round(prev, 1) if prev is not None else None
        if prev is None:
            self._smoothed[key] = raw
        else:
            self._smoothed[key] = prev + (1 - math.exp(-dt / self.cfg.smooth_tau_s)) * (raw - prev)
        return round(self._smoothed[key], 1)

    def score(self, values: dict, dt):
        """Returns (full score, score without headset signals, per-signal positions)."""
        if self.calibration is None:
            return None, None, {}
        parts, weights = {}, {}
        for name in SIGNALS:
            x = values.get(name)
            if x is None or name not in self._signals:
                continue
            pos, w = self._position(name, x)
            if pos is not None:
                parts[name], weights[name] = pos, w

        def combine(groups):
            num = den = 0.0
            for g in groups:
                keys = [k for k in GROUPS[g] if k in parts]
                if not keys:
                    continue
                pos = sum(parts[k] * weights[k] for k in keys) / sum(weights[k] for k in keys)
                gw = max(weights[k] for k in keys)  # the group's strongest responder sets its weight
                num, den = num + pos * gw, den + gw
            return float(np.clip(100 * num / den, 0, 100)) if den else None

        full = self._smooth("full", combine(GROUPS), dt)
        body = self._smooth("body", combine([g for g in GROUPS if g not in HEADSET_GROUPS]), dt)
        self.brain_rule = self._smooth("brain", combine(["brain"]), dt)  # the EEG alone: the teacher
        out = {k: round(v, 2) for k, v in parts.items()}
        brain = [k for k in GROUPS["brain"] if k in parts]
        if brain:  # group position kept under "eeg" for loggers that read one brain value (clutch/face.py)
            out["eeg"] = round(sum(parts[k] * weights[k] for k in brain) / sum(weights[k] for k in brain), 2)
        return full, body, out


class SessionLogger:
    """Writes one folder per session (sessions/<time>_<participant>/):
      meta.json      participant code, consent, protocol, device
      metrics.csv    every 0.5 s tick: phase, scores, EEG band powers, heart rate, face indicators
      events.csv     phase changes, answers, questionnaires, calibration report
      face.csv       face indicators as received from the browser (~1 per second)
      raw_eeg.csv    Muse EEG at 256 Hz (TP9, AF7, AF8, TP10, microvolts)
      raw_ppg.csv    Muse optical heart sensor at 64 Hz
      raw_accel.csv  Muse accelerometer at 52 Hz (g)
      summary.json   end-of-session summary
    Times: `t` = seconds since the app started (same clock as events); `lsl_ts` = device timestamp.
    """
    FACE_COLS = ["face_brow", "face_press", "face_squint", "face_frown", "face_smile", "blink_rate", "head_motion"]
    CHANNELS = ("TP9", "AF7", "AF8", "TP10")
    CHANNEL_COLS = ([f"{b}_{ch}" for ch in CHANNELS for b in ("alpha", "beta", "theta", "gamma", "alphashare", "alphalog")]
                    + [f"q_{ch}" for ch in CHANNELS] + [f"muscle_{ch}" for ch in CHANNELS])
    METRIC_COLS = (["t", "phase", "stress", "stress_brain", "stress_body", "stress_ml", "stress_ml_headset", "stress_personal", "pos_eeg", "pos_hr", "ratio_log", "alpha", "beta", "theta",
                    "hr_bpm", "hr_quality", "hr_source", "hrv_rmssd", "motion", "usable_channels", "contact"]
                   + FACE_COLS + CHANNEL_COLS)

    def __init__(self, root: Path):
        self.root = root
        self.dir = None
        self._files = {}
        self._writers = {}
        self._lock = threading.Lock()  # raw samples arrive from the source thread
        self._flushed = 0.0

    @property
    def active(self):
        return self.dir is not None

    def _open(self, name, header):
        f = open(self.dir / name, "w", newline="")
        w = csv.writer(f)
        w.writerow(header)
        self._files[name], self._writers[name] = f, w
        return w

    def start(self, meta: dict):
        self.stop()
        with self._lock:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            who = "".join(ch for ch in str(meta.get("participant", "anon")) if ch.isalnum() or ch in "-_") or "anon"
            self.dir = self.root / f"{stamp}_{who}"
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
            self._open("metrics.csv", self.METRIC_COLS)
            self._open("events.csv", ["t", "phase", "name", "data"])
            self._open("face.csv", ["t", "phase"] + self.FACE_COLS)
            return self.dir

    def metric(self, m: dict):
        with self._lock:
            if not self.active:
                return
            eeg, hr = m["eeg"], m["hr"]
            contact = "".join(c["q"][0] for c in eeg.get("channels", {}).values())  # e.g. "ggfb"
            face = m.get("face") or {}
            self._writers["metrics.csv"].writerow(
                [m["t"], m["phase"], m["stress"], m.get("stress_brain"), m.get("stress_body"), m.get("stress_ml"), m.get("stress_ml_headset"), m.get("stress_personal"), m["z"].get("eeg"), m["z"].get("hr"),
                 eeg.get("ratio_log"), eeg.get("alpha"), eeg.get("beta"), eeg.get("theta"),
                 hr.get("bpm"), hr.get("quality"), hr.get("source"), hr.get("rmssd"), m["motion"].get("level"),
                 eeg.get("usable"), contact] + [face.get(k) for k in self.FACE_COLS]
                + [self._channel(eeg, col) for col in self.CHANNEL_COLS])
            if time.time() - self._flushed > 2:  # keep the files on disk up to date
                for fh in self._files.values():
                    fh.flush()
                self._flushed = time.time()

    @staticmethod
    def _channel(eeg, col):
        """Per-channel value for a column like alpha_AF8 (relative power) or q_AF8 (contact quality)."""
        kind, ch = col.split("_", 1)
        c = eeg.get("channels", {}).get(ch)
        return None if c is None else c.get(kind)

    def event(self, t, phase, name, data):
        with self._lock:
            if self.active:
                self._writers["events.csv"].writerow([t, phase, name, json.dumps(data)])
                self._files["events.csv"].flush()

    def face(self, t, phase, features: dict):
        with self._lock:
            if self.active:
                self._writers["face.csv"].writerow([t, phase] + [features.get(k) for k in self.FACE_COLS])

    def watch(self, t, bpm, rr_ms):
        """One heart rate notification from the watch: bpm and the beat-to-beat intervals it carried (ms)."""
        with self._lock:
            if not self.active:
                return
            w = self._writers.get("raw_hr.csv") or self._open("raw_hr.csv", ["t", "bpm", "rr_ms"])
            w.writerow([round(t, 3), bpm, " ".join(f"{r:.1f}" for r in rr_ms)])

    def raw(self, stream: str, lsl_ts, block, t0: float, names):
        """block: (channels, samples) array; lsl_ts: one timestamp per sample (wall clock, seconds)."""
        with self._lock:
            if not self.active:
                return
            fname = f"raw_{stream}.csv"
            w = self._writers.get(fname) or self._open(fname, ["lsl_ts", "t"] + list(names))
            w.writerows([round(ts, 4), round(ts - t0, 4), *(round(float(v), 3) for v in block[:, i])]
                        for i, ts in enumerate(lsl_ts))

    def finish(self, summary: dict):
        with self._lock:
            if self.active:
                (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.stop()

    def stop(self):
        with self._lock:
            for f in self._files.values():
                f.close()
            self._files, self._writers = {}, {}
            self.dir = None


class Engine:
    def __init__(self, source, cfg: Config, sessions_dir: Path, learned: LiveScorer | None = None,
                 learned_headset: LiveScorer | None = None):
        self.source = source
        self.learned = learned                    # trained heart + face model
        self.learned_headset = learned_headset    # trained model that uses the EEG (needs the Muse)
        self.cfg = cfg
        self.model = StressModel(cfg)
        self.logger = SessionLogger(sessions_dir)
        self.phase = "idle"
        self.latest = None
        self.seq = 0
        self.t0 = time.time()
        self.error = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.pending_baseline = None  # set when calibration ends, picked up by the server
        self.profile = None           # name typed in step 2: the calibration is saved under it (profiles/)
        self._face, self._face_t = None, 0.0
        self.watch = WatchHR()  # Garmin or any BLE heart rate device; replaces the Muse PPG while it streams
        self.watch.on_sample = lambda wall, bpm, rr: self.logger.watch(round(wall - self.t0, 3), bpm, rr)
        if hasattr(source, "on_samples"):
            source.on_samples = self._on_samples  # raw samples go to the session files while recording

    RAW_NAMES = {"ppg": ["ppg1", "ppg2", "ppg3"], "accel": ["x", "y", "z"]}

    def _on_samples(self, stream, lsl_ts, block):
        if self.logger.active:
            names = self.source.eeg_names if stream == "eeg" else self.RAW_NAMES.get(stream, [])
            self.logger.raw(stream, lsl_ts, block, self.t0, names)

    def set_face(self, features: dict):
        """Face features computed in the browser (MediaPipe), about once per second."""
        with self._lock:
            self._face = {k: float(v) for k, v in features.items() if k in SIGNALS and v is not None}
            self._face_t = time.time()
            self.logger.face(self.now(), self.phase, self._face)

    def start(self):
        self.source.start()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.logger.stop()
        self.source.stop()

    def now(self):
        return round(time.time() - self.t0, 2)

    def set_phase(self, phase: str):
        with self._lock:
            self.phase = phase
            self.source.on_phase(phase)
            if phase == "calibration_check":  # the client says the calibration protocol is over
                self.pending_baseline = self.model.finalize()
                if self.pending_baseline.get("ok"):
                    self.save_profile()
            elif phase == "study_done":  # retrain the personal model on everything recorded since calibration
                updated = self.model.retrain_personal()
                if updated is not None:
                    self.pending_baseline = updated
                    self.save_profile()
            self.logger.event(self.now(), phase, "phase", {})

    # ---------- Profiles: one saved calibration per person, reloaded without recalibrating ----------
    @staticmethod
    def _profile_file(name: str) -> Path:
        safe = "".join(ch for ch in name.strip() if ch.isalnum() or ch in "-_ ").strip().replace(" ", "_")
        return PROFILES_DIR / f"{safe or 'anon'}.json"

    def save_profile(self):
        if not self.profile or self.model.calibration is None:
            return
        PROFILES_DIR.mkdir(exist_ok=True)
        data = {"name": self.profile, "saved_at": time.strftime("%Y-%m-%d %H:%M"), **self.model.state()}
        self._profile_file(self.profile).write_text(json.dumps(data, default=_json_default))

    def load_profile(self, name: str) -> dict | None:
        path = self._profile_file(name)
        if not path.exists():
            return None
        saved = json.loads(path.read_text())
        with self._lock:
            self.model.restore(saved)
            self.profile = saved["name"]
            self.pending_baseline = saved["calibration"]  # the page shows it as the current calibration
        return saved

    @staticmethod
    def list_profiles() -> list[dict]:
        out = []
        for path in sorted(PROFILES_DIR.glob("*.json")) if PROFILES_DIR.exists() else []:
            try:
                d = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            cal = d.get("calibration") or {}
            used = cal.get("used", [])
            out.append({"name": d.get("name", path.stem), "saved_at": d.get("saved_at"), "iaf": d.get("iaf"),
                        "brain_signals": sum(k.split("_")[0] in BRAIN_BANDS for k in used),
                        "student": bool((d.get("personal") or {}).get("features")),
                        "check": (cal.get("check") or {}).get("passed")})
        return sorted(out, key=lambda p: p["saved_at"] or "", reverse=True)

    def _run(self):
        src, cfg = self.source, self.cfg
        next_tick = time.time()
        while not self._stop.is_set():
            next_tick += cfg.tick_s
            try:
                self._tick(src, cfg)
                connected = src.status().get("connected", True)
                self.error = None if connected else "Headset not streaming: check it is on and close to the laptop (reconnecting automatically)."
            except Exception as exc:  # keep the loop alive; surface the error to the UI
                self.error = f"{type(exc).__name__}: {exc}"
            time.sleep(max(0.0, next_tick - time.time()))

    def _tick(self, src, cfg):
        eeg = src.get_eeg(src.eeg_fs * cfg.eeg_window_s)
        ppg = src.get_ppg(src.ppg_fs * cfg.ppg_window_s) if src.ppg_fs else None
        acc = src.get_accel(src.accel_fs * cfg.motion_window_s) if src.accel_fs else None

        psd = None
        if eeg is None or eeg.shape[1] < src.eeg_fs * 2:
            feats = {"channels": {}, "usable": 0, "ratio_log": None, "warming_up": True}
        else:
            feats = eeg_features(eeg, src.eeg_fs, src.eeg_names, cfg.notch_hz)
        psd = feats.pop("psd", None)  # spectra for the trained models; too big for the WebSocket and the logs
        hr = heart_rate(ppg, src.ppg_fs) if ppg is not None else {"bpm": None, "quality": 0.0}
        hr["source"] = "muse" if hr.get("bpm") is not None else None
        watch_bpm = self.watch.latest()
        if watch_bpm is not None:  # the watch is more reliable than the forehead PPG
            hr = {"bpm": float(watch_bpm), "quality": 1.0, "source": "watch", "rmssd": self.watch.rmssd(),
                  "muse_bpm": hr.get("bpm")}
        motion = motion_level(acc)

        with self._lock:
            ratio = feats.get("ratio_log")
            if motion["flag"]:
                ratio = None  # head movement corrupts EEG; skip this window
            face = self._face if (self._face and time.time() - self._face_t < cfg.face_max_age_s) else None
            # Brain signals: theta / alpha / beta per channel in the person's IAF bands (known after a calibration;
            # during the first one they are recomputed at its end), dropped for bad contact, head motion or muscle.
            inputs = row_inputs({**feats, "psd": psd}, hr, motion, face, cfg.min_hr_quality)
            brain, dropped = {}, {}
            for n, ch in feats.get("channels", {}).items():
                why = ("contact" if ch["q"] == "bad" else "motion" if motion["flag"]
                       else "muscle" if ch.get("muscle") else None)
                for b in BRAIN_BANDS:
                    brain[f"{b}_{n}"] = None
                    if why:
                        dropped[f"{b}_{n}"] = why
            if self.model.iaf is not None and brain:
                brain.update(brain_values(inputs, self.model.iaf))
            values = {**brain, "hr": hr["bpm"] if hr["quality"] >= cfg.min_hr_quality else None,
                      "hrv": hr.get("rmssd"), **(face or {})}
            self.model.collect(self.phase, values, dropped, inputs if self.phase in CALIB_PHASES else None)
            full, body, z = self.model.score(values, cfg.tick_s)
            personal = self.model.personal_score(values, cfg.tick_s)  # heart + face imitating the EEG, this calibration
            # Trained models (tools/train_model.py), when installed: heart + face, and with the headset (brain)
            ml = self.learned.update(cfg.tick_s, self.phase, inputs) if self.learned else None
            ml_headset = self.learned_headset.update(cfg.tick_s, self.phase, inputs) if self.learned_headset else None
            # Main score: the trained brain model while the headset streams, else the calibration rule with the
            # EEG; without the headset, the personal model from this calibration, the trained heart + face model,
            # else the calibration rule on heart + face.
            # Two scores, the pitch in one panel:
            #   brain = the teacher: the EEG part of this person's calibration, which the student learned from (so
            #           the two curves are comparable); the shared trained brain model only when there is none.
            #           Only while the headset streams.
            #   body  = the student: heart + face. The personal model, trained at the end of this calibration to
            #           follow the EEG; else the trained heart + face model; else the calibration rule.
            # The main score is the brain while the headset is on, the student once it is off.
            headset_on = eeg is not None and feats.get("usable", 0) > 0
            brain_score, brain_src = next(((v, n) for v, n in (
                (self.model.brain_rule if headset_on else None, "your calibration (EEG)"),
                (ml_headset if headset_on else None, "shared trained brain model")) if v is not None), (None, None))
            body_score, body_src = next(((v, n) for v, n in (
                (personal, "personal model, taught by your EEG"),
                (ml, "trained heart + face model"),
                (body, "your calibration (heart + face)")) if v is not None), (None, None))
            stress, source = ((brain_score, f"brain: {brain_src}") if brain_score is not None else
                              (body_score, f"heart + face: {body_src}") if body_score is not None else (None, None))
            self.seq += 1
            m = {
                "type": "metrics", "seq": self.seq, "t": self.now(), "phase": self.phase,
                "stress": stress, "stress_source": source,
                "stress_brain": brain_score, "stress_brain_source": brain_src,
                "stress_body": body_score, "stress_body_source": body_src, "stress_body_rule": body,
                "stress_ml": ml, "stress_ml_headset": ml_headset, "stress_personal": personal,
                "learned": self.learned.info if self.learned else None,
                "learned_headset": self.learned_headset.info if self.learned_headset else None,
                "calibrated": self.model.calibration is not None,
                "z": z, "used": sorted(z), "eeg": feats, "hr": hr, "motion": motion, "face": face,
                "source": src.status(), "watch": self.watch.status(),
            }
            self.latest = m
            self.logger.metric(m)
