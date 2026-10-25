"""One command for tomorrow: Clutch export -> features -> per-phase stress scores -> model comparison.

Run:  python ml/run.py data/clutch_dataset
      python ml/run.py dataset_export --calm meditation --stress math_hard --cal-seconds 60
  export         folder made by tools/export_dataset.py (it contains sessions/)
  --calm/--stress extra phase names to label (calib_rest = calm and calib_stress = stress already are;
                 the app's study protocol phases are labelled by default, see STUDY_LABELS)
  --cal-seconds  calibration length taken from the start of the calm and stress phases (default 60)
  --no-epistress do not add the EPIStress people (data/epistress_windows.csv) to the training folds

Prints
1. Per person, the average score of every phase (0 = their calm, 1 = their stress), from their own
   calibration only: in-between activities should land in between. Calibration windows are excluded.
2. With 2+ people: every model tested on held-out people (ml/train.py), results in ml/results_<export>.md.
   EPIStress people are added to training (never tested) when data/epistress_windows.csv exists, so the
   table also shows whether the epilepsy dataset helps on your own people.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clutch_data  # noqa: E402
import train  # noqa: E402
from features import EEG_FEATURES, FEATURES, HEART_FEATURES  # noqa: E402

EPISTRESS = Path(__file__).resolve().parent.parent / "data" / "epistress_windows.csv"
# Study protocol phases (frontend app.js): relaxation = calm, hard blocks = stressed, easy math stays in between
STUDY_LABELS = {"study_relax": 0, "study_math_hard": 1, "study_stroop_hard": 1}
SCORE_SETS = {"EEG + heart": EEG_FEATURES + HEART_FEATURES, "heart only": HEART_FEATURES}


def phase_table(windows: Path, cal_seconds: float):
    raw = pd.read_csv(windows)
    train.CAL_WINDOWS = int(round(cal_seconds / 5))
    cal = train.calibration(raw)
    if not raw.attrs["anchors"]:
        print("\nNobody has both a calm and a stress phase labelled: nothing to calibrate.")
        return
    df = train.normalise(raw, cal)
    for set_name, cols in {**SCORE_SETS, "app live score (alpha/beta + heart rate)": None}.items():
        cols = cols and [c for c in cols if df[c].notna().any()]
        scores = pd.Series(index=df.index, dtype=float)
        for subj, (calm, stress) in raw.attrs["anchors"].items():
            g = df.index[df.subject == subj]
            c = g[cal[g]]
            yc = df.label[c].astype(int).values
            if cols is None:   # exactly what the app shows on screen, from the same calibration
                scores[g] = train.app_scores(raw.loc[c, list(train.APP_SIGNALS)], yc, raw.loc[g, list(train.APP_SIGNALS)])
                continue
            X = df.loc[g, cols].fillna(0.0).values
            Xc = df.loc[c, cols].fillna(0.0).values
            scores[g] = train.centroid_scores(None, None, Xc, yc, X, use_lda=False)
        test = df.assign(score=scores)[~cal & scores.notna()]
        table = test.pivot_table(index="task", columns="subject", values="score", aggfunc="mean")
        table["all"] = test.groupby("task").score.mean()
        order = test.groupby("task").score.mean().sort_values().index
        print(f"\nAverage score per phase, {set_name} (0 = own calm, 1 = own stress; calibration windows excluded):")
        print(table.loc[order].round(2).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("export", type=Path)
    p.add_argument("--calm", default="")
    p.add_argument("--stress", default="")
    p.add_argument("--cal-seconds", type=float, default=60)
    p.add_argument("--notch", type=float, default=60.0)
    p.add_argument("--no-epistress", action="store_true")
    a = p.parse_args()

    extra = {**STUDY_LABELS, **{k: 0 for k in a.calm.split(",") if k}, **{k: 1 for k in a.stress.split(",") if k}}
    print("1) Features")
    windows = clutch_data.build(a.export, extra, a.notch)
    phase_table(windows, a.cal_seconds)

    people = pd.read_csv(windows).dropna(subset=["label"]).groupby("subject").label.nunique()
    if (people == 2).sum() >= 2:
        print("\n2) Models on held-out people")
        args = ["--data", str(windows), "--cal-seconds", str(a.cal_seconds)]
        if EPISTRESS.exists() and not a.no_epistress:
            args += ["--extra", str(EPISTRESS)]
        train.main(args)
    else:
        print("\n2) Model comparison skipped: needs at least 2 people with both calm and stress labelled.")


if __name__ == "__main__":
    main()
