"""Real-time loop: reads the source, computes features, turns them into a 0-100 stress score,
and logs sessions to disk.

Stress score
------------
Both signals are compared to the user's own baseline, recorded during calibration:
  z_eeg = -(log10(alpha/beta) - baseline) / spread   (less alpha relative to beta -> more stress)
  z_hr  =  (heart_rate     - baseline) / spread      (faster heart -> more stress)
  z     = 0.6 * z_eeg + 0.4 * z_hr                   (weights renormalised if heart rate is missing)
  score = 50 + 12 * z, smoothed with a ~5 s exponential average, clipped to 0-100
50 means "same as your baseline". 65+ is clearly above it.
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


@dataclass
class Config:
    eeg_window_s: float = 4.0
    ppg_window_s: float = 10.0
    motion_window_s: float = 2.0
    tick_s: float = 0.5
    smooth_tau_s: float = 5.0
    w_eeg: float = 0.6
    w_hr: float = 0.4
    min_sd_eeg: float = 0.08   # floor on baseline spread, log10 units
    min_sd_hr: float = 3.0     # floor on baseline spread, bpm
    min_hr_quality: float = 0.5
    score_gain: float = 12.0   # score points per baseline standard deviation
    notch_hz: float = 60.0     # mains frequency: 60 Hz in the US, 50 Hz in Europe


@dataclass
class Baseline:
    eeg_mu: float | None = None
    eeg_sd: float | None = None
    hr_mu: float | None = None
    hr_sd: float | None = None
    n_eeg: int = 0
    n_hr: int = 0

    def as_dict(self):
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class StressModel:
    cfg: Config
    baseline: Baseline | None = None
    _eeg_samples: list = field(default_factory=list)
    _hr_samples: list = field(default_factory=list)
    _smoothed: float | None = None

    def collect(self, ratio_log, hr_bpm, hr_q):
        if ratio_log is not None:
            self._eeg_samples.append(ratio_log)
        if hr_bpm is not None and hr_q >= self.cfg.min_hr_quality:
            self._hr_samples.append(hr_bpm)

    def finalize(self) -> Baseline:
        b = Baseline(n_eeg=len(self._eeg_samples), n_hr=len(self._hr_samples))
        if b.n_eeg >= 10:
            b.eeg_mu = float(np.median(self._eeg_samples))
            b.eeg_sd = max(float(np.std(self._eeg_samples)), self.cfg.min_sd_eeg)
        if b.n_hr >= 10:
            b.hr_mu = float(np.median(self._hr_samples))
            b.hr_sd = max(float(np.std(self._hr_samples)), self.cfg.min_sd_hr)
        self._eeg_samples, self._hr_samples = [], []
        self._smoothed = None
        if b.eeg_mu is not None or b.hr_mu is not None:
            self.baseline = b
        return b

    def score(self, ratio_log, hr_bpm, hr_q, dt):
        b = self.baseline
        if b is None:
            return None, {}
        parts, weights = {}, {}
        if ratio_log is not None and b.eeg_mu is not None:
            parts["eeg"] = float(np.clip(-(ratio_log - b.eeg_mu) / b.eeg_sd, -3, 3))
            weights["eeg"] = self.cfg.w_eeg
        if hr_bpm is not None and hr_q >= self.cfg.min_hr_quality and b.hr_mu is not None:
            parts["hr"] = float(np.clip((hr_bpm - b.hr_mu) / b.hr_sd, -3, 3))
            weights["hr"] = self.cfg.w_hr
        if not parts:
            return (round(self._smoothed, 1) if self._smoothed is not None else None), parts
        z = sum(parts[k] * weights[k] for k in parts) / sum(weights.values())
        raw = float(np.clip(50 + self.cfg.score_gain * z, 0, 100))
        if self._smoothed is None:
            self._smoothed = raw
        else:
            a = 1 - math.exp(-dt / self.cfg.smooth_tau_s)
            self._smoothed += a * (raw - self._smoothed)
        return round(self._smoothed, 1), {k: round(v, 2) for k, v in parts.items()}


class SessionLogger:
    METRIC_COLS = ["t", "phase", "stress", "z_eeg", "z_hr", "ratio_log", "alpha", "beta", "theta",
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
            leaving_calibration = self.phase == "calibration" and phase != "calibration"
            self.phase = phase
            self.source.on_phase(phase)
            if leaving_calibration:
                self.pending_baseline = self.model.finalize().as_dict()
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
            if self.phase == "calibration":
                self.model.collect(ratio, hr["bpm"], hr["quality"])
            stress, z = self.model.score(ratio, hr["bpm"], hr["quality"], cfg.tick_s)
            if eeg is None:
                stress = None  # headset not streaming: show nothing rather than a stale score
            self.seq += 1
            m = {
                "type": "metrics", "seq": self.seq, "t": self.now(), "phase": self.phase,
                "stress": stress, "calibrated": self.model.baseline is not None, "z": z,
                "eeg": feats, "hr": hr, "motion": motion, "source": src.status(),
            }
            self.latest = m
            self.logger.metric(m)
