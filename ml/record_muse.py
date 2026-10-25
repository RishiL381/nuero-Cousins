"""Record raw Muse 2 data (EEG, PPG, accelerometer) from muselsl to a .npz file for ml/predict_muse.py.

First, in another terminal:  muselsl stream --ppg --acc --name Muse-XXXX
     (or: python -m clutch.muselsl_runner stream --ppg --acc --name Muse-XXXX)
Then:  python ml/record_muse.py calm.npz --seconds 180      (sit still, eyes open, relaxed)
       python ml/record_muse.py stress.npz --seconds 180    (e.g. a Clutch round of timed arithmetic)
Ctrl+C stops early and still saves.
"""

import argparse
import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop


def open_stream(stype, timeout=15):
    found = resolve_byprop("type", stype, timeout=timeout)
    if not found:
        return None
    inlet = StreamInlet(found[0], max_buflen=60)
    return inlet, inlet.info().nominal_srate()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="output file, e.g. calm.npz")
    p.add_argument("--seconds", type=float, default=180)
    args = p.parse_args()

    streams = {k: open_stream(k) for k in ("EEG", "PPG", "ACC")}
    if streams["EEG"] is None:
        raise SystemExit("No EEG stream found. Start: muselsl stream --ppg --acc")
    for k, v in streams.items():
        print(f"{k}: {'%.0f Hz' % v[1] if v else 'not found (features will be missing)'}")

    data = {k: [] for k in streams}
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            for k, s in streams.items():
                if s:
                    chunk, _ = s[0].pull_chunk(timeout=0.0, max_samples=4096)
                    if chunk:
                        data[k].extend(chunk)
            time.sleep(0.05)
            print(f"\r{time.time() - t0:5.0f} s / {args.seconds:.0f}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()

    out = {}
    eeg = np.array(data["EEG"]).T
    out["eeg"], out["eeg_fs"] = eeg[:4], streams["EEG"][1]   # TP9, AF7, AF8, TP10 (drop Right AUX)
    if data["PPG"]:
        out["ppg"], out["ppg_fs"] = np.array(data["PPG"]).T, streams["PPG"][1]
    if data["ACC"]:
        out["acc"], out["acc_fs"] = np.array(data["ACC"]).T, streams["ACC"][1]
    np.savez(args.out, **out)
    print(f"saved {args.out}: {eeg.shape[1] / out['eeg_fs']:.0f} s of EEG"
          + (f", {out['ppg'].shape[1] / out['ppg_fs']:.0f} s of PPG" if "ppg" in out else ""))


if __name__ == "__main__":
    main()
