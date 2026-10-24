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
# Expected direction under stress: EEG calm ratio goes down (-1), heart rate goes up (+1).
SIGNALS = {"eeg": -1, "hr": +1}


@dataclass
class Config:
    eeg_window_s: float = 4.0
    ppg_window_s: float = 10.0
    motion_window_s: float = 2.0
    tick_s: float = 0.5
    smooth_tau_s: float = 5.0
    min_sd = {"eeg": 0.05, "hr": 2.0}  # floor on spreads (log10 units, bpm)
    min_effect: float = 0.5            # Cohen's d needed to use a signal (medium effect, Cohen 1988)
    min_hr_quality: float = 0.5
    notch_hz: float = 60.0             # mains frequency: 60 Hz in the US, 50 Hz in Europe


@dataclass
class StressModel:
    cfg: Config
    calibration: dict | None = None
    _samples: dict = field(default_factory=lambda: {p: {"eeg": [], "hr": []} for p in CALIB_PHASES})
    _smoothed: float | None = None

    @property
    def baseline(self):  # kept for callers that only need to know whether we are calibrated
        return self.calibration

    def collect(self, phase, ratio_log, hr_bpm, hr_q):
        if phase not in self._samples:
            return
        if ratio_log is not None:
            self._samples[phase]["eeg"].append(ratio_log)
        if hr_bpm is not None and hr_q >= self.cfg.min_hr_quality:
            self._samples[phase]["hr"].append(hr_bpm)

    def finalize(self) -> dict:
        rest, stress = self._samples["calib_rest"], self._samples["calib_stress"]
        signals = {}
        for name, direction in SIGNALS.items():
            r, s = rest[name], stress[name]
            info = {"n_rest": len(r), "n_stress": len(s), "used": False}
            if len(r) >= 10:
                info["rest"] = float(np.median(r))
                info["rest_sd"] = max(float(np.std(r)), self.cfg.min_sd[name])
            if len(r) >= 10 and len(s) >= 10:
                info["stress"] = float(np.median(s))
                pooled = max(math.sqrt((np.var(r) + np.var(s)) / 2), self.cfg.min_sd[name])
                info["effect"] = direction * (info["stress"] - info["rest"]) / pooled
                d = info["effect"]
                info["used"] = d >= self.cfg.min_effect
                info["note"] = ("large response" if d >= 0.8 else
                                "medium response" if d >= 0.5 else
                                "small response, not used" if d >= 0.2 else
                                "changed the opposite way, not used" if d <= -0.2 else
                                "no clear change, not used")
                rec = self._samples["calib_recovery"][name]
                if len(rec) >= 10:  # how far back toward calm the user came after the stressor
                    pos = (float(np.median(rec)) - info["rest"]) / (info["stress"] - info["rest"])
                    info["recovered_pct"] = round(100 * (1 - pos), 0)
            else:
                info["note"] = "not enough clean signal"
            signals[name] = info

        used = [k for k, v in signals.items() if v["used"]]
        has_rest = [k for k, v in signals.items() if "rest" in v]
        if used:
            mode = "anchored"
            confidence = "good" if any(signals[k]["effect"] >= 0.8 for k in used) else "ok"
        elif has_rest:
            mode, confidence = "rest_only", "low"
        else:
            mode, confidence = None, None

        cal = {"ok": mode is not None, "mode": mode, "confidence": confidence, "used": used,
               "signals": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                           for k, v in signals.items()}}
        self._samples = {p: {"eeg": [], "hr": []} for p in CALIB_PHASES}
        if cal["ok"]:
            self.calibration = cal
            self._signals = signals
            self._smoothed = None
        return cal

    def score(self, ratio_log, hr_bpm, hr_q, dt):
        cal = self.calibration
        if cal is None:
            return None, {}
        values = {"eeg": ratio_log,
                  "hr": hr_bpm if (hr_bpm is not None and hr_q >= self.cfg.min_hr_quality) else None}
        parts, weights = {}, {}
        for name, direction in SIGNALS.items():
            info, x = self._signals[name], values[name]
            if x is None:
                continue
            if cal["mode"] == "anchored" and info["used"]:
                parts[name] = float(np.clip((x - info["rest"]) / (info["stress"] - info["rest"]), -0.5, 1.5))
                weights[name] = min(info["effect"], 2.0)
            elif cal["mode"] == "rest_only" and "rest" in info:
                parts[name] = float(np.clip(direction * (x - info["rest"]) / info["rest_sd"] / 3, -0.5, 1.5))
                weights[name] = 1.0
        if not parts:
            return (round(self._smoothed, 1) if self._smoothed is not None else None), parts
        position = sum(parts[k] * weights[k] for k in parts) / sum(weights.values())
        raw = float(np.clip(100 * position, 0, 100))
        if self._smoothed is None:
            self._smoothed = raw
        else:
            a = 1 - math.exp(-dt / self.cfg.smooth_tau_s)
            self._smoothed += a * (raw - self._smoothed)
        return round(self._smoothed, 1), {k: round(v, 2) for k, v in parts.items()}



class SessionLogger:
    METRIC_COLS = ["t", "phase", "stress", "pos_eeg", "pos_hr", "ratio_log", "alpha", "beta", "theta",
                   "hr_bpm", "hr_quality", "motion", "usable_channels", "contact"]

    def __init__(self, root: Path):
        self.root = root
        self.dir = None
        self._mf = self._ef = None
        self._mw = self._ew = None

    @property
    def active(self):
        return self.dir is not None

    def start(self, meta: dict):
        self.stop()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        who = "".join(ch for ch in str(meta.get("participant", "anon")) if ch.isalnum() or ch in "-_") or "anon"
        self.dir = self.root / f"{stamp}_{who}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        self._mf = open(self.dir / "metrics.csv", "w", newline="")
        self._mw = csv.writer(self._mf)
        self._mw.writerow(self.METRIC_COLS)
        self._ef = open(self.dir / "events.csv", "w", newline="")
        self._ew = csv.writer(self._ef)
        self._ew.writerow(["t", "phase", "name", "data"])
        return self.dir

    def metric(self, m: dict):
        if not self.active:
            return
        eeg, hr = m["eeg"], m["hr"]
        contact = "".join(c["q"][0] for c in eeg.get("channels", {}).values())  # e.g. "ggfb"
        self._mw.writerow([m["t"], m["phase"], m["stress"], m["z"].get("eeg"), m["z"].get("hr"),
                           eeg.get("ratio_log"), eeg.get("alpha"), eeg.get("beta"), eeg.get("theta"),
                           hr.get("bpm"), hr.get("quality"), m["motion"].get("level"),
                           eeg.get("usable"), contact])

    def event(self, t, phase, name, data):
        if self.active:
            self._ew.writerow([t, phase, name, json.dumps(data)])
            self._ef.flush()

    def finish(self, summary: dict):
        if self.active:
            (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        self.stop()

    def stop(self):
        for f in (self._mf, self._ef):
            if f:
                f.close()
        self.dir = None
        self._mf = self._ef = self._mw = self._ew = None


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
            self.model.collect(self.phase, ratio, hr["bpm"], hr["quality"])
            stress, z = self.model.score(ratio, hr["bpm"], hr["quality"], cfg.tick_s)
            if eeg is None:
                stress = None  # headset not streaming: show nothing rather than a stale score
            self.seq += 1
            m = {
                "type": "metrics", "seq": self.seq, "t": self.now(), "phase": self.phase,
                "stress": stress, "calibrated": self.model.calibration is not None, "z": z, "used": sorted(z),
                "eeg": feats, "hr": hr, "motion": motion, "source": src.status(),
            }
            self.latest = m
            self.logger.metric(m)
