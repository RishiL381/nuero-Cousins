"""End-to-end check of the stress pipeline on the simulated headset, faster than real time.

Run: .venv\\Scripts\\python tests\\test_pipeline.py   (or pytest)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clutch.engine import Config, Engine, StressModel  # noqa: E402
from clutch.learned import LiveScorer  # noqa: E402
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


def simulate(phases, with_face=False, learned=None):
    src = MockSource(seed=3)
    src.start()
    src._advance = lambda: None  # data only from src.step(): deterministic, independent of the PC's speed
    rng = np.random.default_rng(5)
    with tempfile.TemporaryDirectory() as tmp:
        eng = Engine(src, Config(), Path(tmp), learned)
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


def test_individual_alpha_frequency():
    from clutch.processing import individual_alpha_frequency, iaf_bands
    freqs = np.arange(1, 45.5, 0.5)
    rng = np.random.default_rng(4)
    spectra = [10 / freqs * (1 + 1.5 * np.exp(-((freqs - 11.5) ** 2) / 0.8)) * rng.uniform(0.9, 1.1, freqs.size)
               for _ in range(20)]                                       # 1/f background + alpha peak at 11.5 Hz
    assert individual_alpha_frequency(spectra, freqs) == 11.5
    assert individual_alpha_frequency([10 / freqs] * 20, freqs) is None  # no peak: no IAF
    assert iaf_bands(12.0)["alpha"] == (8.0, 14.0)                        # Klimesch: IAF-4 to IAF+2


def test_muscle_tension_is_flagged_and_excluded():
    fs, t = 256, np.arange(256 * 4) / 256
    rng = np.random.default_rng(2)
    brain = 10 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 3, t.size)
    b, a = signal.butter(4, [30, 44], btype="bandpass", fs=fs)
    emg = signal.filtfilt(b, a, rng.normal(0, 40, t.size))  # forehead muscle: broadband power above 30 Hz
    names = ["TP9", "AF7", "AF8", "TP10"]
    chans = eeg_features(np.vstack([brain, brain + emg, brain, brain]), fs, names)["channels"]
    assert chans["AF7"]["muscle"] and "muscle" in chans["AF7"]["why"], chans["AF7"]
    assert not any(chans[n]["muscle"] for n in ("TP9", "AF8", "TP10"))

    model = StressModel(Config())
    for _ in range(40):  # AF7 tense for the whole calibration: rejected every time
        model.collect("calib_rest", {"alpha_AF7": None, "hr": 60 + rng.normal(0, 1)}, {"alpha_AF7": "muscle"})
        model.collect("calib_stress", {"alpha_AF7": None, "hr": 70 + rng.normal(0, 1)}, {"alpha_AF7": "muscle"})
    cal = model.finalize()
    af7 = cal["signals"]["alpha_AF7"]
    assert "alpha_AF7" not in cal["used"] and af7["dropped"] == {"muscle": 100} and "muscle" in af7["note"], af7



def test_personal_model_trains_at_calibration_end():
    out, eng = simulate([("calib_rest", 60), ("calib_stress", 60), ("calibration_check", 1),
                         ("round1", 60), ("reset", 60)], with_face=True)
    pm = eng.model.calibration["personal"]
    print("personal model:", pm)
    assert pm["ok"] and "hr" in pm["features"] and "face_brow" in pm["features"], pm
    assert not any(k.startswith("alpha_") for k in pm["features"])
    assert all(m["stress_personal"] is None for m in out["calib_stress"])  # not trained before the check

    def late(phase):
        return float(np.mean([m["stress_personal"] for m in out[phase][-60:] if m["stress_personal"] is not None]))
    print(f"personal model: round1={late('round1'):.0f} reset={late('reset'):.0f}")
    assert late("round1") > 60 and late("reset") < 30


def test_personal_model_needs_heart_or_face():
    rng = np.random.default_rng(4)
    model = StressModel(Config())
    for _ in range(40):  # EEG only: there is a target but nothing to learn it from
        model.collect("calib_rest", {"alpha_AF8": 1.0 + rng.normal(0, 0.05)})
        model.collect("calib_stress", {"alpha_AF8": 0.7 + rng.normal(0, 0.05)})
    cal = model.finalize()
    assert cal["ok"] and not cal["personal"]["ok"] and model.personal is None, cal["personal"]
    assert model.personal_score({"hr": 80}, 0.5) is None


def test_trained_model_scores_live():
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    # features are changes from the person's rest baseline: heart rate up ~10 bpm under stress
    X = np.r_[rng.normal(0, 2, 200), rng.normal(10, 2, 200)].reshape(-1, 1)
    y = np.r_[np.zeros(200), np.ones(200)]
    bundle = {"model": HistGradientBoostingClassifier(max_iter=50).fit(X, y), "features": ["hr"],
              "name": "headset_free", "trained_on": ["synthetic"], "cv_auc": 1.0}
    out, _ = simulate([("calib_rest", 60), ("calib_stress", 60), ("calibration_check", 1),
                       ("round1", 60), ("reset", 60)], learned=LiveScorer(bundle))
    assert all(m["stress_ml"] is None for m in out["calib_rest"][:80])  # no baseline yet
    late = lambda ph: float(np.mean([m["stress_ml"] for m in out[ph][-40:] if m["stress_ml"] is not None]))
    print(f"trained model: round1={late('round1'):.0f} reset={late('reset'):.0f}")
    assert late("round1") > 60 and late("reset") < 40

    # teacher model: a regressor imitating an EEG index, scaled so rest = 0 and stress = 1
    from sklearn.ensemble import HistGradientBoostingRegressor
    teacher = {"model": HistGradientBoostingRegressor(max_iter=50).fit(X, 0.3 * y), "kind": "teacher",
               "scale": (0.0, 0.3), "features": ["hr"], "trained_on": ["synthetic"], "cv_auc": 1.0}
    out, _ = simulate([("calib_rest", 60), ("calib_stress", 60), ("calibration_check", 1),
                       ("round1", 60), ("reset", 60)], learned=LiveScorer(teacher))
    print(f"teacher model: round1={late('round1'):.0f} reset={late('reset'):.0f}")
    assert late("round1") > 60 and late("reset") < 40


if __name__ == "__main__":
    test_individual_alpha_frequency()
    test_muscle_tension_is_flagged_and_excluded()
    test_trained_model_scores_live()
    test_headset_free_score_from_face_and_heart()
    test_rest_only_calibration_is_flagged_low_confidence()
    test_alpha_feature_tracks_relaxation()
    test_heart_rate_on_clean_pulse()
    test_stress_score_follows_phases()
    print("all checks passed")
