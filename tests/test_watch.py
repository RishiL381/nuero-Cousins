"""Checks the Bluetooth heart rate decoding and RMSSD (no device needed).

Run: .venv\Scripts\python tests\test_watch.py   (or pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clutch.watch import parse_measurement, rmssd_of  # noqa: E402


def test_parse_heart_rate_and_rr():
    assert parse_measurement(bytes([0x00, 72])) == (72, [])                     # 8-bit heart rate only
    bpm, rr = parse_measurement(bytes([0x10, 64, 0x00, 0x04, 0x00, 0x02]))    # + RR 1024/1024 s, 512/1024 s
    assert bpm == 64 and rr == [1000.0, 500.0], (bpm, rr)
    bpm, rr = parse_measurement(bytes([0x19, 0x2C, 0x01, 0x10, 0x00, 0x00, 0x04]))  # 16-bit HR, energy, RR
    assert bpm == 300 and rr == [1000.0], (bpm, rr)


def test_rmssd():
    assert rmssd_of([800, 820] * 3) is None                                    # too few beats
    assert rmssd_of([800, 820] * 10) == 20.0                                   # alternating by 20 ms
    assert rmssd_of([800, 820] * 10 + [2500]) == 20.0                          # implausible beat ignored


if __name__ == "__main__":
    test_parse_heart_rate_and_rr()
    test_rmssd()
    print("all checks passed")
