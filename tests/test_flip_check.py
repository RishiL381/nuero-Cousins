"""Checks the calibration flip check: anchors from the first half, tested on the second half.

Run: .venv\\Scripts\\python tests\\test_flip_check.py   (or pytest)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clutch.engine import Config, StressModel  # noqa: E402


def calibrate(rest_hr, stress_hr, rest_eeg=None, stress_eeg=None):
    m = StressModel(Config())
    for phase, hr, eeg in (("calib_rest", rest_hr, rest_eeg), ("calib_stress", stress_hr, stress_eeg)):
        for i, h in enumerate(hr):
            m.collect(phase, {"hr": h, "alpha_AF8": None if eeg is None else eeg[i]})  # one EEG channel, log alpha
    return m.finalize()


def test_clear_responder_passes():
    rng = np.random.default_rng(0)
    cal = calibrate(rng.normal(62, 2, 120), rng.normal(76, 2, 120), rng.normal(0.1, 0.05, 120), rng.normal(-0.1, 0.05, 120))
    assert cal["check"]["passed"] is True, cal["check"]
    assert cal["check"]["gap"] > 0.5 and set(cal["check"]["signals"]) == {"alpha_AF8", "hr"}


def test_no_response_fails():
    rng = np.random.default_rng(1)
    cal = calibrate(rng.normal(62, 3, 120), rng.normal(62, 3, 120))
    assert cal["check"]["passed"] is False, cal["check"]


def test_response_only_in_first_half_is_caught():
    # heart rate jumps early in the stress test, then falls below rest: the second half points the wrong way
    rng = np.random.default_rng(2)
    stress = np.r_[rng.normal(80, 2, 60), rng.normal(58, 2, 60)]
    cal = calibrate(rng.normal(62, 2, 120), stress)
    assert cal["check"]["passed"] is False and cal["check"]["gap"] < 0, cal["check"]
    assert "wrong way" in cal["check"]["note"]


def test_short_calibration_is_not_checked():
    rng = np.random.default_rng(3)
    cal = calibrate(rng.normal(62, 2, 12), rng.normal(76, 2, 12))
    assert cal["check"]["passed"] is None


if __name__ == "__main__":
    test_clear_responder_passes()
    test_no_response_fails()
    test_response_only_in_first_half_is_caught()
    test_short_calibration_is_not_checked()
    print("all checks passed")
