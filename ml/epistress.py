"""Turn EPIStress lab tasks into labelled 10 s windows of Muse-style features.

EPIStress (Zenodo 10.5281/zenodo.16407549, CC BY 4.0): 20 adults with epilepsy wearing a Muse S headband
(TP9, AF7, AF8, TP10, dry electrodes, 256 Hz) and an Empatica E4 wristband (pulse wave at 64 Hz), doing a
relaxation video and mental tasks (arithmetic, Stroop, N-back, Sudoku) at easy and hard levels.
After each task (5-10 min) they rated their stress from 1 to 5. That rating labels every window of the task:
  1-2 -> 0 (calm), 4-5 -> 1 (stressed), 3 -> dropped (ambiguous)
Recorded in Germany, so mains noise is 50 Hz.

Run: python ml/epistress.py   -> writes data/epistress_windows.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_pickle  # noqa: E402
from features import CHANNELS, featurize_recording  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "data" / "epistress"
OUT = ROOT.parent / "epistress_windows.csv"
TASKS = ["relaxation_video", "arithmetix_easy", "arithmetix_hard", "stroop_easy", "stroop_hard",
         "n_back_easy", "n_back_hard", "sudoku_easy", "sudoku_hard"]
# Each labeled file is indexed by its device's own sample counter: Muse at 256 Hz (the accelerometer is
# repeated on every EEG row by Mind Monitor), Empatica E4 pulse wave at 64 Hz. A 600 s task = 153,600 Muse samples.
DEVICE_HZ = {"Muse": 256.0, "E4": 64.0}
NOTCH_HZ = 50.0


def stress_label(rating: float):
    if rating <= 2:
        return 0
    if rating >= 4:
        return 1
    return None


def read_series(path: Path):
    """Values and sampling rate of one labeled signal."""
    s = safe_pickle.load(path)
    step = float(np.median(np.diff(s.index.values)))  # 1 for raw signals, 64 for E4 heart rate (1 Hz)
    return s.values.astype(float), DEVICE_HZ[path.name.split("__")[0]] / step


def load_task(folder: Path):
    eeg, fs = zip(*(read_series(folder / f"Muse__RAW_{c}.pickle") for c in CHANNELS))
    n = min(len(x) for x in eeg)
    eeg = np.vstack([x[:n] for x in eeg])
    acc, acc_fs = None, None
    if all((folder / f"Muse__ACC_{a}.pickle").exists() for a in "XYZ"):
        parts = [read_series(folder / f"Muse__ACC_{a}.pickle") for a in "XYZ"]
        m = min(len(p[0]) for p in parts)
        acc, acc_fs = np.vstack([p[0][:m] for p in parts]), parts[0][1]
        # the labeled accelerometer is not in g (about 62 at rest); scale so gravity, the median magnitude, is 1 g
        acc = acc / np.median(np.linalg.norm(acc, axis=0))
    ppg, ppg_fs = (read_series(folder / "E4__BVP.pickle") if (folder / "E4__BVP.pickle").exists() else (None, None))
    if ppg is not None:
        ppg = ppg[None, :]
    return eeg, fs[0], ppg, ppg_fs, acc, acc_fs


def build():
    rows = []
    for subj in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        labels_file = subj / "Lab" / "Task_Labels.csv"
        if not labels_file.exists():
            continue
        ratings = {r["Task"]: r for r in csv.DictReader(open(labels_file))}
        for task in TASKS:
            folder = subj / "Lab" / "Labeled" / task
            if task not in ratings or not (folder / "Muse__RAW_AF7.pickle").exists():
                continue
            try:
                stress = float(ratings[task]["Mental stress level"])
            except ValueError:
                continue
            label = stress_label(stress)
            if label is None:
                continue
            eeg, eeg_fs, ppg, ppg_fs, acc, acc_fs = load_task(folder)
            feats, starts = featurize_recording(eeg, eeg_fs, ppg, ppg_fs, acc, acc_fs, NOTCH_HZ)
            for f, t in zip(feats, starts):
                rows.append({"subject": subj.name, "task": task, "t": t, "stress_rating": stress, "label": label,
                             "arousal": float(ratings[task]["Arousal Score"] or "nan"), **f})
            print(f"{subj.name} {task:17s} stress={stress:.0f} -> {len(feats):3d} windows (EEG {eeg_fs:.0f} Hz)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\n{len(df)} windows from {df.subject.nunique()} subjects -> {OUT}")
    print(df.groupby("label").size().rename({0: "calm", 1: "stressed"}).to_string())
    return df


if __name__ == "__main__":
    build()
