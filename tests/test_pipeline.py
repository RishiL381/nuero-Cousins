"""End-to-end check of the stress pipeline on the simulated headset, faster than real time.

Run: .venv\\Scripts\\python tests\\test_pipeline.py   (or pytest)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clutch.engine import Config, Engine  # noqa: E402
from clutch.processing import eeg_features, heart_rate  # noqa: E402
from clutch.sources import MockSource  # noqa: E402


def test_alpha_feature_tracks_relaxation():
    fs, t = 256, np.arange(256 * 4) / 256
    rng = np.random.default_rng(1)
    relaxed = 15 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 3, t.size)
    tense = 3 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 6, t.size)
    names = ["TP9", "AF7", "AF8", "TP10"]
    r = eeg_features(np.tile(relaxed, (4, 1)), fs, names)["ratio_log"]
    s = eeg_features(np.tile(tense, (4, 1)), fs, names)["ratio_log"]
    assert r > s, (r, s)


def test_heart_rate_on_clean_pulse():
    fs, t = 64, np.arange(64 * 10) / 64
    pulse = np.sin(2 * np.pi * 1.25 * t)  # 75 bpm
    hr = heart_rate(np.vstack([np.zeros_like(t), -pulse, -0.5 * pulse]), fs)
    assert abs(hr["bpm"] - 75) < 3 and hr["quality"] > 0.8, hr


def simulate(phases):
    src = MockSource(seed=3)
    src.start()
    with tempfile.TemporaryDirectory() as tmp:
        eng = Engine(src, Config(), Path(tmp))
        out = {}
        for phase, seconds in phases:
            eng.set_phase(phase)
            rows = []
            for _ in range(int(seconds / 0.5)):
                src.step(0.5)
                eng._tick(src, eng.cfg)
                rows.append(eng.latest)
            out[phase] = rows
        return out, eng


def test_stress_score_follows_phases():
    out, eng = simulate([("calib_rest", 60), ("calib_stress", 60), ("calib_recovery", 30), ("calibration_check", 1),
                         ("round1", 60), ("reset", 60), ("round2", 60)])
    cal = eng.model.calibration
    assert cal is not None and cal["mode"] == "anchored", cal
    assert set(cal["used"]) == {"eeg", "hr"}, cal  # both signals respond in the simulation
    print("calibration:", {k: (v["note"], round(v["effect"], 2), v.get("recovered_pct")) for k, v in cal["signals"].items()})

    def late_mean(phase):  # last 30 s of each phase, after the smoothing has settled
        vals = [m["stress"] for m in out[phase][-60:] if m["stress"] is not None]
        return float(np.mean(vals))

    r1, rs, r2 = late_mean("round1"), late_mean("reset"), late_mean("round2")
    print(f"stress (0 = your calm, 100 = your stress test)  round1={r1:.1f}  reset={rs:.1f}  round2={r2:.1f}")
    assert r1 > 60 and rs < 25 and rs < r2 < r1

    hr = [m["hr"]["bpm"] for m in out["round1"][-20:] if m["hr"]["bpm"]]
    print(f"heart rate round1 ~{np.mean(hr):.0f} bpm (simulated ~73)")
    assert 68 < np.mean(hr) < 78

    contact = out["round1"][-1]["eeg"]["channels"]
    assert all(c["q"] != "bad" for c in contact.values()), contact


def test_rest_only_calibration_is_flagged_low_confidence():
    _, eng = simulate([("calib_rest", 30), ("calibration_check", 1)])
    cal = eng.model.calibration
    assert cal["mode"] == "rest_only" and cal["confidence"] == "low" and cal["used"] == [], cal


if __name__ == "__main__":
    test_rest_only_calibration_is_flagged_low_confidence()
    test_alpha_feature_tracks_relaxation()
    test_heart_rate_on_clean_pulse()
    test_stress_score_follows_phases()
    print("all checks passed")
