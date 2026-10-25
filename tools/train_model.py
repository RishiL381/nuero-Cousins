"""Train stress classifiers (gradient-boosted trees) on the Clutch dataset and compare feature sets.

Usage (from the project folder):
    .venv\\Scripts\\python tools\\train_model.py                          -> evaluate on data/clutch_dataset
    .venv\\Scripts\\python tools\\train_model.py --data dataset_export --save models

Features are recomputed from the raw files of each session (raw_eeg.csv, raw_ppg.csv, raw_accel.csv,
face.csv) with the same code as the live app (clutch/learned.py), every second of the labelled phases
(rest = 0, stress = 1). Each feature is taken relative to the first 45 s of the person's rest phase;
those baseline seconds are then left out of training and evaluation.

Validation:
  - 2+ participants: leave-one-participant-out (the model never saw the person it is tested on).
  - 1 participant: each phase is cut into 5 contiguous blocks; the model is tested on one block of rest
    and one of stress, trained on the others, with 15 s removed around the test blocks so that
    overlapping windows do not leak. This only says whether the signals separate rest from stress for
    that person; time trends within the session can inflate it.
--save writes models/<set>.joblib for the "full" and "headset_free" sets, trained on all rows. The app
loads models/headset_free.joblib at start-up if it exists.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clutch.learned import (ALL_FEATURES, BASELINE_S, FEATURE_SETS, SMOOTH_S, baseline_of,  # noqa: E402
                            row_features, window_mean)
from clutch.processing import eeg_features, heart_rate, motion_level  # noqa: E402

FS = {"eeg": 256, "ppg": 64, "accel": 52}  # Muse 2
WINDOW_S = {"eeg": 4.0, "ppg": 10.0, "accel": 2.0}
STEP_S = 1.0
N_BLOCKS, GAP_S = 5, 15.0


def make_model():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)


def load_raw(path):
    if not path.exists():
        return None, None, []
    df = pd.read_csv(path)
    names = [c for c in df.columns if c not in ("lsl_ts", "t")]
    return df["t"].to_numpy(), df[names].to_numpy().T, names


def window(t, x, t_end, seconds, fs):
    if t is None:
        return None
    a, b = np.searchsorted(t, t_end - seconds, "right"), np.searchsorted(t, t_end, "right")
    return x[:, a:b] if b - a >= 0.85 * seconds * fs else None


def session_rows(sdir: Path, notch_hz: float):
    """One raw feature row per second of the labelled phases."""
    meta = json.loads((sdir / "meta.json").read_text())
    labels = [r for r in csv.DictReader(open(sdir / "labels.csv")) if r["label"] != ""]
    te, eeg, eeg_names = load_raw(sdir / "raw_eeg.csv")
    tp, ppg, _ = load_raw(sdir / "raw_ppg.csv")
    ta, acc, _ = load_raw(sdir / "raw_accel.csv")
    face = pd.read_csv(sdir / "face.csv") if (sdir / "face.csv").exists() else None
    keep = [i for i, n in enumerate(eeg_names) if n in ("TP9", "AF7", "AF8", "TP10")]
    rows = []
    for k, lab in enumerate(labels):
        t0, t1 = float(lab["t_start"]), float(lab["t_end"])
        for tt in np.arange(t0 + WINDOW_S["eeg"], t1, STEP_S):
            w = window(te, eeg, tt, WINDOW_S["eeg"], FS["eeg"])
            feats = eeg_features(w[keep], FS["eeg"], [eeg_names[i] for i in keep], notch_hz) if w is not None else {}
            w = window(tp, ppg, tt, WINDOW_S["ppg"], FS["ppg"])
            hr = heart_rate(w, FS["ppg"]) if w is not None else {}
            w = window(ta, acc, tt, WINDOW_S["accel"], FS["accel"])
            motion = motion_level(w) if w is not None else {}
            f = None
            if face is not None:
                recent = face[(face["t"] <= tt) & (face["t"] > tt - 2)]
                f = recent.iloc[-1].to_dict() if len(recent) else None
                f = {k_: (None if pd.isna(v) else v) for k_, v in f.items()} if f else None
            rows.append({"participant": meta.get("participant", sdir.name), "session": sdir.name, "t": float(tt),
                         "label": int(lab["label"]), "block": k, "since_start": float(tt - t0),
                         **row_features(feats, hr, motion, f)})
    return rows


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline-correct each session with its first BASELINE_S of rest, then smooth causally within each phase."""
    out = []
    for _, s in df.groupby("session"):
        s = s.sort_values("t").copy()
        is_base = (s["label"] == 0) & (s["since_start"] <= BASELINE_S)
        X = s[ALL_FEATURES].to_numpy(dtype=float)
        if is_base.sum() == 0:
            continue
        X = X - baseline_of(list(X[is_base.to_numpy()]))
        t, blocks = s["t"].to_numpy(), s["block"].to_numpy()
        smooth = np.vstack([window_mean(list(X[(blocks == blocks[i]) & (t > t[i] - SMOOTH_S) & (t <= t[i])]))
                            for i in range(len(s))])
        s[ALL_FEATURES] = smooth
        out.append(s[~is_base.to_numpy()])
    return pd.concat(out, ignore_index=True)


def folds(df: pd.DataFrame):
    """Yields (name, train mask, test mask)."""
    people = df["participant"].unique()
    if len(people) >= 2:
        for p in people:
            yield p, (df["participant"] != p).to_numpy(), (df["participant"] == p).to_numpy()
        return
    chunk = np.zeros(len(df), dtype=int)
    for _, g in df.groupby(["session", "block"]):
        chunk[g.index] = np.minimum((np.argsort(np.argsort(g["t"].to_numpy())) * N_BLOCKS) // len(g), N_BLOCKS - 1)
    for c in range(N_BLOCKS):
        test = chunk == c
        train = ~test
        for sess, g in df[test].groupby("session"):  # purge training rows close to the test rows
            tt = np.sort(g["t"].to_numpy())
            same = (df["session"] == sess).to_numpy()
            t = df["t"].to_numpy()
            pos = np.clip(np.searchsorted(tt, t), 1, len(tt) - 1)
            near = np.minimum(np.abs(t - tt[pos - 1]), np.abs(t - tt[pos])) <= GAP_S
            train &= ~(same & near)
        yield f"block {c + 1}", train, test


def evaluate(df: pd.DataFrame, features, importance=False):
    y = df["label"].to_numpy()
    X = df[features].to_numpy(dtype=float)
    prob = np.full(len(df), np.nan)
    fold_auc, imp = [], np.zeros(len(features))
    for _, train, test in folds(df):
        cols = ~np.all(np.isnan(X[train]), axis=0)  # a feature never recorded in training cannot be used
        if len(set(y[train])) < 2 or len(set(y[test])) < 2 or not cols.any():
            continue
        m = make_model().fit(X[train][:, cols], y[train])
        prob[test] = m.predict_proba(X[test][:, cols])[:, 1]
        fold_auc.append(roc_auc_score(y[test], prob[test]))
        if importance:
            r = permutation_importance(m, X[test][:, cols], y[test], scoring="roc_auc", n_repeats=10, random_state=0)
            imp[cols] += r.importances_mean
    if not fold_auc:
        return {"auc": None, "auc_folds": [], "balanced_acc": None, "note": "no usable data"}
    ok = ~np.isnan(prob)
    res = {"auc": round(float(roc_auc_score(y[ok], prob[ok])), 3),
           "auc_folds": [round(a, 2) for a in fold_auc],
           "balanced_acc": round(float(balanced_accuracy_score(y[ok], prob[ok] >= 0.5)), 3)}
    if importance and fold_auc:
        res["importance"] = {k: round(float(v / len(fold_auc)), 3) for k, v in
                             sorted(zip(features, imp), key=lambda kv: -kv[1])}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "clutch_dataset"), help="exported dataset folder")
    p.add_argument("--notch", type=float, default=60.0, help="mains frequency (60 Hz US, 50 Hz Europe)")
    p.add_argument("--save", help="folder to save the trained models into (e.g. models)")
    args = p.parse_args()

    sdirs = sorted(d for d in (Path(args.data) / "sessions").iterdir() if (d / "labels.csv").exists())
    raw = pd.DataFrame([r for d in sdirs for r in session_rows(d, args.notch)])
    df = normalise(raw)
    n0, n1 = int((df["label"] == 0).sum()), int((df["label"] == 1).sum())
    people = df["participant"].nunique()
    validation = "leave-one-participant-out" if people >= 2 else f"{N_BLOCKS} time blocks within the session"
    print(f"{len(sdirs)} sessions, {people} participants, {len(df)} rows of 1 s (rest {n0}, stress {n1}) "
          f"after removing the first {BASELINE_S:.0f} s of rest (baseline)")
    print(f"Validation: {validation}\n")
    print("Missing values (share of rows):")
    miss = df[ALL_FEATURES].isna().mean()
    print("  " + (", ".join(f"{k} {v:.0%}" for k, v in miss.items() if v > 0) or "none"))

    report = {"validation": validation, "rows": {"rest": n0, "stress": n1}, "participants": people, "sets": {}}
    print(f"\n{'feature set':14s} {'AUC':>6s} {'bal. acc':>9s}   AUC per fold")
    for name, feats in FEATURE_SETS.items():
        res = evaluate(df, feats, importance=name in ("full", "headset_free"))
        report["sets"][name] = res
        if res["auc"] is None:
            print(f"{name:14s} {'-':>6s} {'-':>9s}   no usable data")
        else:
            print(f"{name:14s} {res['auc']:6.2f} {res['balanced_acc']:9.2f}   {res['auc_folds']}")
    for name in ("full", "headset_free"):
        imp = report["sets"][name].get("importance", {})
        print(f"\nMost useful features, {name} (drop in AUC when shuffled):")
        for k, v in list(imp.items())[:8]:
            print(f"  {k:16s} {v:+.3f}")
    print("\nAUC: 0.5 = chance, 1.0 = perfect separation of rest and stress.")

    if args.save:
        import joblib
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        for name in ("full", "headset_free"):
            feats = [k for k in FEATURE_SETS[name] if df[k].notna().any()]
            model = make_model().fit(df[feats].to_numpy(dtype=float), df["label"].to_numpy())
            joblib.dump({"name": name, "model": model, "features": feats, "baseline_s": BASELINE_S,
                         "smooth_s": SMOOTH_S, "trained_on": sorted(df["session"].unique()),
                         "cv_auc": report["sets"][name]["auc"], "validation": validation}, out / f"{name}.joblib")
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"\nSaved {out / 'full.joblib'}, {out / 'headset_free.joblib'} and {out / 'report.json'}")


if __name__ == "__main__":
    main()
