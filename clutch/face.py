"""Logs webcam facial features sent by the face tracker page (frontend/face.html).

The browser runs MediaPipe Face Landmarker and sends only numbers, never images. Two files:
  face_features.csv  one row per 0.5 s tick: facial features + the EEG targets at that moment
                     (stress, rest-to-stress positions, alpha/beta, heart rate), ready to train on
  face_frames.csv    every camera frame: head pose + the 52 raw blendshapes, to recompute features later
When a session from the main app is running, both files go into that session's folder;
otherwise into sessions/<time>_<participant>_face/.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

TARGET_COLS = ["stress", "pos_eeg", "pos_hr", "ratio_log", "hr_bpm", "eeg_motion", "eeg_usable"]
POSE_COLS = ["yaw", "pitch", "roll", "tx", "ty", "tz"]


class FaceLogger:
    def __init__(self, root: Path):
        self.root = root
        self.dir = None
        self.rows = 0
        self._ff = self._rf = None
        self._fw = self._rw = None
        self._feature_cols: list[str] = []
        self._blend_cols: list[str] = []

    @property
    def active(self):
        return self.dir is not None

    def start(self, session_dir: Path | None, meta: dict, feature_cols: list, blend_cols: list) -> Path:
        self.stop()
        if session_dir is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            who = "".join(ch for ch in str(meta.get("participant", "anon")) if ch.isalnum() or ch in "-_") or "anon"
            session_dir = self.root / f"{stamp}_{who}_face"
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "face_meta.json").write_text(json.dumps(meta, indent=2))
        self._feature_cols, self._blend_cols = list(feature_cols), list(blend_cols)
        self._ff = open(self.dir / "face_features.csv", "w", newline="")
        self._fw = csv.writer(self._ff)
        self._fw.writerow(["t", "phase"] + self._feature_cols + TARGET_COLS)
        self._rf = open(self.dir / "face_frames.csv", "w", newline="")
        self._rw = csv.writer(self._rf)
        self._rw.writerow(["t", "present"] + POSE_COLS + self._blend_cols)
        self.rows = 0
        return self.dir

    def write(self, now: float, phase: str, latest: dict | None, msg: dict):
        """msg: {"sent": ms, "row": {feature: value}, "frames": [{"t": ms, "s": [...], "p": [...]}]}.
        Browser times (ms) are mapped onto the engine clock so face and EEG rows line up on t."""
        if not self.active:
            return
        sent = msg.get("sent", 0)
        for f in msg.get("frames", []):
            t = round(now - (sent - f["t"]) / 1000, 3)
            scores, pose = f.get("s"), f.get("p") or [None] * len(POSE_COLS)
            self._rw.writerow([t, int(scores is not None)] + list(pose) + (list(scores) if scores else [None] * len(self._blend_cols)))
        row = msg.get("row")
        if row is not None:
            self._fw.writerow([now, phase] + [row.get(c) for c in self._feature_cols] + targets(latest))
            self.rows += 1
        self._ff.flush()
        self._rf.flush()

    def stop(self):
        for f in (self._ff, self._rf):
            if f:
                f.close()
        self.dir = None
        self._ff = self._rf = self._fw = self._rw = None


def targets(m: dict | None) -> list:
    """The EEG engine's latest outputs, used as training targets for the face features."""
    if m is None:
        return [None] * len(TARGET_COLS)
    eeg, z = m.get("eeg", {}), m.get("z", {})
    return [m.get("stress"), z.get("eeg"), z.get("hr"), eeg.get("ratio_log"), m.get("hr", {}).get("bpm"),
            m.get("motion", {}).get("level"), eeg.get("usable")]
