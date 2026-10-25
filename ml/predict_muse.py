"""Score your own Muse 2 recordings with the per-person centroid model (the best model on EPIStress).

Usage:  python ml/predict_muse.py --calm calm.npz --stress stress.npz session.npz [more.npz ...]
        python ml/predict_muse.py --calm calm.csv --stress math.csv session.csv     (Mind Monitor CSVs)
Inputs: .npz from ml/record_muse.py (EEG + PPG + accelerometer), or Mind Monitor CSVs
        (EEG + accelerometer only: Mind Monitor has no PPG, so heart features are missing).

How it scores
1. The calm and stress recordings are this person's calibration: their windows are averaged into a
   calm point and a stress point (after z-scoring every feature with the calibration windows).
2. Each 10 s window of the session is placed on the line between them: 0 = your calm, 1 = your stress.
   Above 0.5 = closer to your stress.
No trained model file is needed: everything is learned from the person's own calibration.
Motion is left out of the score (it mostly tells "typing vs sitting still") but printed next to it.
A proxy for stress compared with your own calibration, not a diagnosis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import EEG_FEATURES, HEART_FEATURES, featurize_recording  # noqa: E402

FEATURE_SETS = {"eeg+heart": EEG_FEATURES + HEART_FEATURES, "heart": HEART_FEATURES, "eeg": EEG_FEATURES}


def load_npz(path):
    z = np.load(path)
    get = lambda k: z[k] if k in z else None  # noqa: E731
    fs = lambda k: float(z[k]) if k in z else None  # noqa: E731
    return get("eeg"), fs("eeg_fs"), get("ppg"), fs("ppg_fs"), get("acc"), fs("acc_fs")


def load_mind_monitor(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"])
    t = pd.to_datetime(df["TimeStamp"])
    fs = round((len(df) - 1) / (t.iloc[-1] - t.iloc[0]).total_seconds())
    eeg = df[["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]].to_numpy().T
    acc = df[["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"]].ffill().to_numpy().T
    return eeg, float(fs), None, None, acc, float(fs)


def featurize(path, notch):
    loader = load_npz if str(path).endswith(".npz") else load_mind_monitor
    eeg, eeg_fs, ppg, ppg_fs, acc, acc_fs = loader(path)
    rows, starts = featurize_recording(eeg, eeg_fs, ppg, ppg_fs, acc, acc_fs, notch)
    print(f"{Path(path).name}: {eeg.shape[1] / eeg_fs:.0f} s, {len(rows)} usable windows"
          + ("" if ppg is not None else " (no PPG: heart features missing)"))
    if not rows:
        raise SystemExit(f"No usable windows in {path}: check electrode contact.")
    return pd.DataFrame(rows).assign(file=Path(path).name, t=starts)


def centroid_score(calm: pd.DataFrame, stress: pd.DataFrame, X: pd.DataFrame, cols) -> np.ndarray:
    """Position of each row of X between the calm (0) and stress (1) averages, after z-scoring with calibration."""
    cal = pd.concat([calm, stress])[cols]
    cols = [c for c in cols if cal[c].notna().mean() > 0.5]   # drop features the calibration mostly lacks
    mu, sd = cal[cols].mean(), cal[cols].std() + 1e-9
    z = lambda d: ((d[cols] - mu) / sd).fillna(0.0).to_numpy()  # noqa: E731
    c0, c1 = z(calm).mean(0), z(stress).mean(0)
    w = c1 - c0
    return (z(X) - c0) @ w / (w @ w + 1e-9)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calm", required=True, help="calibration recording while calm (e.g. relaxation video)")
    p.add_argument("--stress", required=True, help="calibration recording under stress (e.g. hard arithmetic)")
    p.add_argument("files", nargs="+", help="recordings to score")
    p.add_argument("--features", choices=list(FEATURE_SETS), default="eeg+heart")
    p.add_argument("--notch", type=float, default=60.0, help="mains frequency: 60 (US) or 50 (EU)")
    args = p.parse_args()

    cols = FEATURE_SETS[args.features]
    calm, stress = featurize(args.calm, args.notch), featurize(args.stress, args.notch)
    df = pd.concat([featurize(f, args.notch) for f in args.files], ignore_index=True)
    df["score"] = centroid_score(calm, stress, df, cols)

    print(f"\nScore per 10 s window ({args.features}; 0 = your calm, 1 = your stress):")
    for name, g in df.groupby("file", sort=False):
        bars = "".join(" ▁▂▃▄▅▆▇█"[int(np.clip(s, 0, 1) * 8)] for s in g.score)
        print(f"  {name:24s} {bars}")
    print("\nPer file:")
    summary = df.groupby("file", sort=False).agg(score=("score", "mean"), stressed_share=("score", lambda s: (s > 0.5).mean()),
                                                 hr_bpm=("hr_bpm", "median"), motion_g=("motion_std_g", "median"), windows=("score", "size"))
    print(summary.round(2).to_string())
    out = Path(args.files[0]).with_suffix(".scores.csv")
    df[["file", "t", "score", "alpha_beta_log", "hr_bpm", "motion_std_g"]].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
