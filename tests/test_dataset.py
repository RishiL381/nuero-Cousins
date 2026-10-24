"""Record a simulated calibration (EEG + heart + face) and export it as a dataset.

Run: .venv\\Scripts\\python tests\\test_dataset.py
"""

import csv
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from clutch.engine import Config, Engine  # noqa: E402
from clutch.sources import MockSource  # noqa: E402
from test_pipeline import fake_face  # noqa: E402


def record(sessions_dir, participant, consent):
    src = MockSource(seed=1)
    src.start()
    eng = Engine(src, Config(), sessions_dir)
    rng = np.random.default_rng(2)
    eng.logger.start({"kind": "calibration", "participant": participant, "consent": consent, "protocol": "short"})
    for phase, seconds in (("calib_rest", 20), ("calib_survey", 2), ("calib_stress", 20), ("calibration_check", 1)):
        eng.set_phase(phase)
        for _ in range(int(seconds / 0.5)):
            src.step(0.5)
            eng.set_face(fake_face(src.stress, rng))
            eng._tick(src, eng.cfg)
    eng.logger.event(eng.now(), eng.phase, "calibration",
                     {"used": ["eeg", "hr"], "stai_before": {"score": 27}, "stai_after": {"score": 50}})
    eng.logger.finish({"kind": "calibration"})


def test_export():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        record(tmp / "sessions", "Alice", consent=True)
        record(tmp / "sessions", "Bob", consent=False)   # must be skipped
        out = tmp / "export"
        res = subprocess.run([sys.executable, str(ROOT / "tools" / "export_dataset.py"),
                              "--sessions", str(tmp / "sessions"), "--out", str(out)],
                             capture_output=True, text=True)
        print(res.stdout.strip())
        assert res.returncode == 0, res.stderr

        rows = list(csv.DictReader(open(out / "features.csv")))
        labels = [r["label"] for r in rows]
        assert labels.count("0") == 40 and labels.count("1") == 40, (labels.count("0"), labels.count("1"))
        assert {r["participant"] for r in rows} == {"P01"}             # anonymised, Bob excluded
        assert all(r["face_brow"] for r in rows if r["label"] != "")    # face recorded on labelled rows
        sess = out / "sessions" / "P01_S01"
        for name in ("raw_eeg.csv", "raw_ppg.csv", "raw_accel.csv", "face.csv", "labels.csv", "events.csv", "meta.json"):
            assert (sess / name).exists(), name
        eeg = list(csv.reader(open(sess / "raw_eeg.csv")))
        assert eeg[0] == ["lsl_ts", "t", "TP9", "AF7", "AF8", "TP10"] and len(eeg) > 256 * 40, len(eeg)
        assert json.loads((sess / "meta.json").read_text())["participant"] == "P01"   # no real name
        assert "Alice" not in (sess / "meta.json").read_text()
        zips = list(tmp.glob("clutch_dataset_*.zip"))
        assert zips and "README.md" in zipfile.ZipFile(zips[0]).namelist()
        assert "dataset_export_mapping.json" not in zipfile.ZipFile(zips[0]).namelist()
        print(f"raw EEG rows: {len(eeg) - 1} (~{(len(eeg) - 1) / 256:.0f} s at 256 Hz) | zip: {zips[0].name}")


if __name__ == "__main__":
    test_export()
    print("dataset export OK")
