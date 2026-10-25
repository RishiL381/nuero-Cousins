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


def fake_face(stress, rng):
    """Face features that follow the simulated stress like a real face might (with noise)."""
    n = lambda sd: rng.normal(0, sd)
    return {"face_brow": 0.10 + 0.30 * stress + n(0.04), "face_press": 0.05 + 0.20 * stress + n(0.04),
            "face_squint": 0.20 + n(0.05),                      # does not react: must be left out
            "face_smile": 0.30 - 0.20 * stress + n(0.04),       # learned direction: goes DOWN with stress
            "blink_rate": 15 + n(3), "head_motion": 0.01 + 0.02 * stress + n(0.004)}


def simulate(phases, with_face=False):
    src = MockSource(seed=3)
    src.start()
    src._advance = lambda: None  # data only from src.step(): deterministic, independent of the PC's speed
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        eng = Engine(src, Config(), Path(tmp))
        out = {}
        for phase, seconds in phases:
            eng.set_phase(phase)
            rows = []
            for _ in range(int(seconds / 0.5)):
                src.step(0.5)
                if with_face:
                    eng.set_face(fake_face(src.stress, rng))
                eng._tick(src, eng.cfg)
                rows.append(eng.latest)
            out[phase] = rows
        return out, eng


def test_stress_score_follows_phases():
    out, eng = simulate([("calib_rest", 60), ("calib_stress", 60), ("calib_recovery", 30), ("calibration_check", 1),
                         ("round1", 60), ("reset", 60), ("round2", 60)])
    cal = eng.model.calibration
    assert cal is not None and cal["mode"] == "anchored", cal
    eeg = {"alpha_TP9", "alpha_AF7", "alpha_AF8", "alpha_TP10"}
    # the heart and most channels respond (simulated blinks add noise on the forehead channels, as in real data)
    assert "hr" in cal["used"] and len(eeg & set(cal["used"])) >= 3, cal["used"]
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


def test_headset_free_score_from_face_and_heart():
    out, eng = simulate([("calib_rest", 60), ("calib_stress", 60), ("calibration_check", 1),
                         ("round1", 60), ("reset", 60)], with_face=True)
    cal = eng.model.calibration
    print("used:", cal["used"], "| without headset:", cal["used_without_headset"])
    print("face smile direction learned:", cal["signals"]["face_smile"].get("direction"))
    assert {"face_brow", "face_press", "face_smile", "hr"} <= set(cal["used_without_headset"]), cal["used"]
    assert "face_squint" not in cal["used"]
    assert not any(k.startswith("alpha_") for k in cal["used_without_headset"])
    assert cal["signals"]["face_smile"]["direction"] == -1

    def late(phase, key):
        return float(np.mean([m[key] for m in out[phase][-60:] if m[key] is not None]))
    b1, bs = late("round1", "stress_body"), late("reset", "stress_body")
    f1, fs = late("round1", "stress"), late("reset", "stress")
    print(f"with EEG: round1={f1:.0f} reset={fs:.0f} | without headset: round1={b1:.0f} reset={bs:.0f}")
    assert b1 > 60 and bs < 25


if __name__ == "__main__":
    test_headset_free_score_from_face_and_heart()
    test_rest_only_calibration_is_flagged_low_confidence()
    test_alpha_feature_tracks_relaxation()
    test_heart_rate_on_clean_pulse()
    test_stress_score_follows_phases()
    print("all checks passed")
