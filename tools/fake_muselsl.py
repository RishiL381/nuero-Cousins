"""Publish simulated Muse data on LSL exactly like `muselsl stream --ppg --acc` does.

Lets the team test the LSL path (python -m clutch --source lsl) without a headset.
Usage: .venv\\Scripts\\python tools\\fake_muselsl.py [--stress 0.3]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clutch.sources import MockSource  # noqa: E402  (imported first: it quiets liblsl logs)
from pylsl import StreamInfo, StreamOutlet  # noqa: E402


def outlet(stype, labels, rate, unit, chunk):
    info = StreamInfo("Muse", stype, len(labels), rate, "float32", "MuseFAKE")
    chans = info.desc().append_child("channels")
    for label in labels:
        ch = chans.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", unit)
    return StreamOutlet(info, chunk)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stress", type=float, default=None, help="force simulated stress 0-1 (default: drifts at 0.3)")
    args = p.parse_args()
    mock = MockSource(seed=7)
    mock.set_stress(args.stress)
    eeg = outlet("EEG", ["TP9", "AF7", "AF8", "TP10", "Right AUX"], 256, "microvolts", 12)
    ppg = outlet("PPG", ["PPG1", "PPG2", "PPG3"], 64, "mmHg", 6)
    acc = outlet("ACC", ["X", "Y", "Z"], 52, "g", 1)
    print("Fake Muse streaming EEG, PPG, ACC on LSL. Ctrl+C to stop.")
    counts = {"eeg": 0, "ppg": 0, "acc": 0}
    last = time.time()
    while True:
        time.sleep(0.05)
        now = time.time()
        mock.step(now - last)
        last = now
        for key, ring, out, extra in (("eeg", mock.eeg, eeg, 1), ("ppg", mock.ppg, ppg, 0), ("acc", mock.acc, acc, 0)):
            new = ring.count - counts[key]
            if new > 0:
                block = ring.last(new)
                if extra:  # Right AUX channel, unused
                    block = list(block) + [block[0] * 0]
                out.push_chunk([list(map(float, col)) for col in zip(*block)])
                counts[key] = ring.count


if __name__ == "__main__":
    main()
