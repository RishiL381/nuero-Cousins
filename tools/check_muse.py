"""Muse diagnostic: connect, show what the headset streams, check contact and heart rate,
and optionally run the eyes-open / eyes-closed alpha test that proves the EEG is real.

Usage (from the project folder):
    .venv\\Scripts\\python tools\\check_muse.py --eyes-test                  (starts muselsl itself)
    .venv\\Scripts\\python tools\\check_muse.py --name Muse-4A2B --eyes-test
    .venv\\Scripts\\python tools\\check_muse.py --source lsl                 (muselsl already running)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clutch.processing import eeg_features, heart_rate, motion_level  # noqa: E402
from clutch.sources import MUSE_BOARDS, LslSource, MuseSource  # noqa: E402


def section(title):
    print(f"\n=== {title} ===")


def contact_report(src, seconds=4):
    eeg = src.get_eeg(src.eeg_fs * seconds)
    if eeg is None:
        print("  No EEG data yet.")
        return
    f = eeg_features(eeg, src.eeg_fs, src.eeg_names)
    for name, ch in f["channels"].items():
        print(f"  {name:5s} {ch['q']:5s}  std={ch['std']:6.1f} uV  ptp={ch['ptp']:6.1f} uV  mains={ch['line']:.2f}"
              f"  alpha={ch['alpha']:.2f} beta={ch['beta']:.2f}")


def mean_alpha(src, seconds, label):
    print(f"\n>>> {label} for {seconds} s ...")
    vals = []
    end = time.time() + seconds
    time.sleep(4)  # let the 4 s window fill with the new condition
    while time.time() < end:
        f = eeg_features(src.get_eeg(src.eeg_fs * 4), src.eeg_fs, src.eeg_names)
        if f.get("alpha") is not None:
            vals.append(f["alpha"])
        time.sleep(1)
    v = float(np.mean(vals)) if vals else float("nan")
    print(f"    mean relative alpha: {v:.3f}")
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["muselsl", "lsl", "brainflow"], default="muselsl")
    p.add_argument("--name", "--serial-number", dest="name", default="",
                   help="e.g. Muse-4A2B (shown in Bluetooth / printed inside the band)")
    p.add_argument("--board", choices=list(MUSE_BOARDS), default="muse2", help="brainflow only")
    p.add_argument("--ppg-preset", default="p50", help="brainflow only")
    p.add_argument("--seconds", type=int, default=12)
    p.add_argument("--eyes-test", action="store_true")
    args = p.parse_args()

    section("Connecting")
    print("Muse on (lights blinking), Bluetooth on, headset NOT paired to a phone. Searching...")
    if args.source == "brainflow":
        src = MuseSource(args.board, args.name, "", args.ppg_preset)
    else:
        src = LslSource(args.name, start_muselsl=args.source == "muselsl")
    t0 = time.time()
    src.start()
    print(f"Streaming after {time.time() - t0:.1f} s")

    try:
        section("What the headset streams")
        print(f"EEG   : {src.eeg_names} at {src.eeg_fs} Hz (microvolts)")
        print(f"Accel : {f'{src.accel_fs} Hz' if src.accel_fs else 'not found'}")
        print(f"PPG   : {f'{src.ppg_fs} Hz' if src.ppg_fs else 'not found'}")

        counts0 = {k: src.sample_count(k) for k in ("eeg", "ppg", "accel")}
        t1 = time.time()
        print(f"\nRecording {args.seconds} s, sit still...")
        time.sleep(args.seconds)

        section("Data rate (Bluetooth drops show up here)")
        dt = time.time() - t1
        for key, fs in (("eeg", src.eeg_fs), ("ppg", src.ppg_fs), ("accel", src.accel_fs)):
            if fs:
                got = src.sample_count(key) - counts0[key]
                print(f"  {key:5s} {got} samples in {dt:.1f} s = {got / dt:.0f} Hz (expected {fs}, {100 * got / (fs * dt):.0f}%)")

        section("Electrode contact (want 'good' on all four)")
        contact_report(src)
        print("  Tips: TP9/TP10 behind the ears need skin contact (move hair, wet slightly);"
              " AF7/AF8 on the forehead, band snug.")

        section("Heart rate (PPG)")
        ppg = src.get_ppg(src.ppg_fs * 10) if src.ppg_fs else None
        if ppg is None:
            print("  No PPG data. The app still works on EEG alone.")
        else:
            hr = heart_rate(ppg, src.ppg_fs)
            print(f"  {hr}  (quality > 0.5 is usable; compare with your pulse on the wrist)")

        section("Head motion")
        acc = src.get_accel(src.accel_fs * 2) if src.accel_fs else None
        print(f"  {motion_level(acc)}")

        if args.eyes_test:
            section("Eyes test: alpha should rise clearly with eyes closed")
            a_open = mean_alpha(src, 15, "Eyes OPEN, look at a fixed point")
            print("\a")
            a_closed = mean_alpha(src, 15, "Eyes CLOSED, relax")
            print("\a")
            ratio = a_closed / a_open if a_open else float("nan")
            verdict = "PASS: the headset is reading real brain activity" if ratio > 1.3 else \
                "WEAK: check contact behind the ears, then retry"
            print(f"\n  closed/open alpha = {ratio:.2f}  -> {verdict}")
    finally:
        src.stop()
    print("\nDone.")


if __name__ == "__main__":
    main()
