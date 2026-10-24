"""Checks that face tracker messages are written to CSV next to the EEG targets.

Run: .venv\\Scripts\\python tests\\test_face_logger.py   (or pytest)
"""

import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clutch.face import POSE_COLS, TARGET_COLS, FaceLogger  # noqa: E402


def read(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_face_rows_carry_eeg_targets_and_aligned_time():
    with tempfile.TemporaryDirectory() as tmp:
        log = FaceLogger(Path(tmp))
        out = log.start(None, {"participant": "P1"}, ["tension", "blink_rate_min"], ["browDownLeft", "jawOpen"])
        assert out.name.endswith("_P1_face")
        latest = {"stress": 61.5, "z": {"eeg": 0.8, "hr": 0.4}, "eeg": {"ratio_log": -0.2, "usable": 4},
                  "hr": {"bpm": 72.0}, "motion": {"level": 0.01}}
        log.write(10.0, "round1", latest, {
            "sent": 5000.0,
            "row": {"tension": 0.21, "blink_rate_min": None},
            "frames": [{"t": 4500.0, "s": [0.1, 0.3], "p": [1, 2, 3, 0, 0, -40]}, {"t": 4950.0, "s": None, "p": None}],
        })
        log.stop()

        feats = read(out / "face_features.csv")
        assert feats[0] == ["t", "phase", "tension", "blink_rate_min"] + TARGET_COLS
        assert feats[1] == ["10.0", "round1", "0.21", "", "61.5", "0.8", "0.4", "-0.2", "72.0", "0.01", "4"]

        frames = read(out / "face_frames.csv")
        assert frames[0] == ["t", "present"] + POSE_COLS + ["browDownLeft", "jawOpen"]
        assert frames[1][:2] == ["9.5", "1"] and frames[1][-2:] == ["0.1", "0.3"]  # 500 ms before the send
        assert frames[2][:2] == ["9.95", "0"]


def test_face_files_join_a_running_session():
    with tempfile.TemporaryDirectory() as tmp:
        session = Path(tmp) / "20261024-120000_P2"
        log = FaceLogger(Path(tmp))
        assert log.start(session, {}, ["tension"], ["jawOpen"]) == session
        log.write(1.0, "calibration", None, {"sent": 0, "row": {"tension": 0.1}, "frames": []})
        log.stop()
        assert read(session / "face_features.csv")[1][-len(TARGET_COLS):] == [""] * len(TARGET_COLS)


if __name__ == "__main__":
    test_face_rows_carry_eeg_targets_and_aligned_time()
    test_face_files_join_a_running_session()
    print("all checks passed")
