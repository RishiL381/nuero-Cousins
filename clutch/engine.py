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

from .processing import eeg_features, heart_rate, motion_level

CALIB_PHASES = ("calib_rest", "calib_stress", "calib_recovery")  # phases whose data are collected
# Expected direction under stress: +1 goes up, -1 goes down, 0 = learned per person from the calibration.
#   eeg          log10(alpha/beta), down with stress (alpha suppression, Vanhollebeke et al., 2022)
#   hr           heart rate in bpm (Muse PPG now, a Garmin watch later), up with stress
#   face_*       MediaPipe blendshape means (0-1): brows lowered, lips pressed, eyes squinted, frown, smile
#   blink_rate   blinks per minute, head_motion: head movement (face widths per second)
SIGNALS = {
    "eeg": -1, "hr": +1,
    "face_brow": +1, "face_press": +1, "face_squint": +1, "face_frown": +1, "face_smile": 0,
    "blink_rate": 0, "head_motion": +1,
}
HEADSET_SIGNALS = {"eeg"}  # left out of the "without headset" score
MIN_SD = {"eeg": 0.05, "hr": 2.0, "blink_rate": 2.0, "head_motion": 0.005}  # floors on spreads
MIN_SD_FACE = 0.02


@dataclass
class Config:
    eeg_window_s: float = 4.0
    ppg_window_s: float = 10.0
    motion_window_s: float = 2.0
    tick_s: float = 0.5
    smooth_tau_s: float = 5.0
    min_effect: float = 0.5            # Cohen's d needed to use a signal (medium effect, Cohen 1988)
    min_hr_quality: float = 0.5
    face_max_age_s: float = 2.0        # face features older than this are ignored
    notch_hz: float = 60.0             # mains frequency: 60 Hz in the US, 50 Hz in Europe


def _empty_samples():
    return {p: {k: [] for k in SIGNALS} for p in CALIB_PHASES}


@dataclass
class StressModel:
    cfg: Config
    calibration: dict | None = None
    _samples: dict = field(default_factory=_empty_samples)
    _smoothed: dict = field(default_factory=dict)

    @property
    def baseline(self):  # kept for callers that only need to know whether we are calibrated
        return self.calibration

    def collect(self, phase, values: dict):
        """values: signal name -> current value (None or missing = not available this tick)."""
        if phase not in self._samples:
            return
        for name, x in values.items():
            if name in SIGNALS and x is not None:
                self._samples[phase][name].append(float(x))

    def finalize(self) -> dict:
        rest, stress = self._samples["calib_rest"], self._samples["calib_stress"]
        signals = {}
        for name, expected in SIGNALS.items():
            r, s = rest[name], stress[name]
            floor = MIN_SD.get(name, MIN_SD_FACE)
            info = {"n_rest": len(r), "n_stress": len(s), "used": False}
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
            elif len(r) or len(s):
                info["note"] = "not enough clean signal"
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

        cal = {"ok": mode is not None, "mode": mode, "confidence": confidence, "used": used,
               "used_without_headset": [k for k in used if k not in HEADSET_SIGNALS],
               "signals": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                           for k, v in signals.items() if v["n_rest"] or v["n_stress"]}}
        self._samples = _empty_samples()
        if cal["ok"]:
            self.calibration = cal
            self._signals = signals
            self._smoothed = {}
        return cal

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

        def combine(keys):
            keys = [k for k in keys if k in parts]
            if not keys:
                return None
            return float(np.clip(100 * sum(parts[k] * weights[k] for k in keys) / sum(weights[k] for k in keys), 0, 100))

        full = self._smooth("full", combine(parts), dt)
        body = self._smooth("body", combine([k for k in parts if k not in HEADSET_SIGNALS]), dt)
        return full, body, {k: round(v, 2) for k, v in parts.items()}


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
    METRIC_COLS = (["t", "phase", "stress", "stress_body", "pos_eeg", "pos_hr", "ratio_log", "alpha", "beta", "theta",
                    "hr_bpm", "hr_quality", "motion", "usable_channels", "contact"] + FACE_COLS)

    def __init__(self, root: Path):
        self.root = root
        self.dir = None
        self._files = {}
        self._writers = {}
        self._lock = threading.Lock()  # raw samples arrive from the source thread

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
                [m["t"], m["phase"], m["stress"], m.get("stress_body"), m["z"].get("eeg"), m["z"].get("hr"),
                 eeg.get("ratio_log"), eeg.get("alpha"), eeg.get("beta"), eeg.get("theta"),
                 hr.get("bpm"), hr.get("quality"), m["motion"].get("level"),
                 eeg.get("usable"), contact] + [face.get(k) for k in self.FACE_COLS])

    def event(self, t, phase, name, data):
        with self._lock:
            if self.active:
                self._writers["events.csv"].writerow([t, phase, name, json.dumps(data)])
                self._files["events.csv"].flush()

    def face(self, t, phase, features: dict):
        with self._lock:
            if self.active:
                self._writers["face.csv"].writerow([t, phase] + [features.get(k) for k in self.FACE_COLS])

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
    def __init__(self, source, cfg: Config, sessions_dir: Path):
        self.source = source
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
        self._face, self._face_t = None, 0.0
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
            self.logger.event(self.now(), phase, "phase", {})

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

        if eeg is None or eeg.shape[1] < src.eeg_fs * 2:
            feats = {"channels": {}, "usable": 0, "ratio_log": None, "warming_up": True}
        else:
            feats = eeg_features(eeg, src.eeg_fs, src.eeg_names, cfg.notch_hz)
        hr = heart_rate(ppg, src.ppg_fs) if ppg is not None else {"bpm": None, "quality": 0.0}
        motion = motion_level(acc)

        with self._lock:
            ratio = feats.get("ratio_log")
            if motion["flag"]:
                ratio = None  # head movement corrupts EEG; skip this window
            face = self._face if (self._face and time.time() - self._face_t < cfg.face_max_age_s) else None
            values = {"eeg": ratio, "hr": hr["bpm"] if hr["quality"] >= cfg.min_hr_quality else None, **(face or {})}
            self.model.collect(self.phase, values)
            full, body, z = self.model.score(values, cfg.tick_s)
            # Main score: with EEG when the headset streams, otherwise the headset-free score.
            stress = full if eeg is not None else body
            self.seq += 1
            m = {
                "type": "metrics", "seq": self.seq, "t": self.now(), "phase": self.phase,
                "stress": stress, "stress_body": body, "calibrated": self.model.calibration is not None,
                "z": z, "used": sorted(z), "eeg": feats, "hr": hr, "motion": motion, "face": face,
                "source": src.status(),
            }
            self.latest = m
            self.logger.metric(m)
