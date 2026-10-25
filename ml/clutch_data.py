"""Turn a Clutch dataset export (tools/export_dataset.py) into 10 s feature windows, same features as EPIStress.

Reads every <export>/sessions/<session>/ folder:
  raw_eeg.csv    lsl_ts, t, TP9, AF7, AF8, TP10   (256 Hz, microvolts)
  raw_ppg.csv    lsl_ts, t, ppg1, ppg2, ppg3      (64 Hz)            optional
  raw_accel.csv  lsl_ts, t, x, y, z               (52 Hz, g)         optional
  labels.csv     t_start, t_end, phase, label     (calib_rest = 0, calib_stress = 1, other phases empty)
  meta.json      participant code
`t` is the session clock shared by all files. Windows are cut by time, not by sample count, because
Bluetooth drops samples: a window is skipped when a stream has less than 90% of its expected samples.
Every phase is kept (label empty = unlabelled) so in-between activities can still be scored.

Run: python ml/clutch_data.py data/clutch_dataset   -> data/clutch_dataset_windows.csv
     --calm meditation --stress math_quiz   to label extra phase names (added to the export's labels)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CHANNELS, STEP_S, WINDOW_S, window_features  # noqa: E402

MIN_FILL = 0.9


def read_stream(path: Path, cols):
    """(t, signal (channels, samples), sampling rate) or (None, None, None) when the file is missing."""
    if not path.exists():
        return None, None, None
    df = pd.read_csv(path).dropna(subset=cols).sort_values("t")
    if len(df) < 10:
        return None, None, None
    t = df["t"].to_numpy()
    fs = float(round((len(t) - 1) / (t[-1] - t[0])))
    return t, df[cols].to_numpy().T, fs


def cut(t, x, fs, t0):
    """Samples of one window [t0, t0 + WINDOW_S), or None when too many are missing."""
    if t is None:
        return None
    i0, i1 = np.searchsorted(t, [t0, t0 + WINDOW_S])
    return x[:, i0:i1] if (i1 - i0) >= MIN_FILL * fs * WINDOW_S else None


def session_windows(sdir: Path, extra_labels: dict, notch: float):
    meta = json.loads((sdir / "meta.json").read_text()) if (sdir / "meta.json").exists() else {}
    eeg_t, eeg, eeg_fs = read_stream(sdir / "raw_eeg.csv", CHANNELS)
    if eeg_t is None:
        print(f"  {sdir.name}: no raw EEG, skipped")
        return []
    ppg_t, ppg, ppg_fs = read_stream(sdir / "raw_ppg.csv", ["ppg1", "ppg2", "ppg3"])
    acc_t, acc, acc_fs = read_stream(sdir / "raw_accel.csv", ["x", "y", "z"])
    labels = pd.read_csv(sdir / "labels.csv")

    rows = []
    for _, seg in labels.iterrows():
        label = seg.label if pd.notna(seg.label) else extra_labels.get(seg.phase, np.nan)
        t0 = float(seg.t_start)
        while t0 + WINDOW_S <= seg.t_end:
            e = cut(eeg_t, eeg, eeg_fs, t0)
            if e is not None:
                f = window_features(e, eeg_fs, cut(ppg_t, ppg, ppg_fs, t0), ppg_fs,
                                    cut(acc_t, acc, acc_fs, t0), notch)
                if f is not None:
                    rows.append({"subject": meta.get("participant", sdir.name.split("_")[0]), "session": sdir.name,
                                 "task": seg.phase, "t": round(t0 - seg.t_start, 2), "label": label, **f})
            t0 += STEP_S
    n_lab = sum(pd.notna(r["label"]) for r in rows)
    print(f"  {sdir.name}: EEG {eeg_fs:.0f} Hz, PPG {ppg_fs or '-'} Hz, accel {acc_fs or '-'} Hz -> "
          f"{len(rows)} windows ({n_lab} labelled)")
    return rows


def build(export: Path, extra_labels: dict | None = None, notch: float = 60.0) -> Path:
    sessions = export / "sessions"
    if not sessions.exists():
        raise SystemExit(f"{sessions} not found: point this at the export folder (it contains sessions/)")
    rows = []
    for sdir in sorted(d for d in sessions.iterdir() if d.is_dir()):
        rows += session_windows(sdir, extra_labels or {}, notch)
    df = pd.DataFrame(rows)
    out = export.parent / f"{export.name}_windows.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} windows from {df.subject.nunique() if len(df) else 0} people -> {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", type=Path)
    p.add_argument("--calm", default="", help="comma-separated extra phase names to label calm (0)")
    p.add_argument("--stress", default="", help="comma-separated extra phase names to label stressed (1)")
    p.add_argument("--notch", type=float, default=60.0)
    a = p.parse_args()
    extra = {**{k: 0 for k in a.calm.split(",") if k}, **{k: 1 for k in a.stress.split(",") if k}}
    build(a.export, extra, a.notch)


if __name__ == "__main__":
    main()
