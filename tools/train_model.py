"""Train a stress model on Clutch recordings, two ways, and compare them on data the model never saw.

Usage (from the project folder):
    .venv\\Scripts\\python tools\\train_model.py                          -> evaluate on data/clutch_dataset
    .venv\\Scripts\\python tools\\train_model.py --data dataset_export --save models

Features are recomputed from the raw files of each session (raw_eeg.csv, raw_ppg.csv, raw_accel.csv,
face.csv) with the same code as the live app (clutch/learned.py), one row per second of every phase.
Each feature is taken relative to the first 45 s of the session's first rest block; those seconds are then
left out of training and evaluation.

Two ways to learn:
  A. label    the target is what the person was doing: rest = 0, stressor = 1 (labelled phases only).
  B. teacher  the target is the EEG's own stress index at every second: the drop in log alpha power from
              baseline, averaged over the channels whose alpha dropped during that session's stressor
              (Cohen's d >= 0.5). A heart + face model learns to imitate it, from every phase, labelled or not.
Both are judged the same way: does the heart + face score separate the labelled rest and stress seconds of
a session the model never saw (AUC)?

Validation, chosen automatically:
  - 2+ participants: leave-one-participant-out;
  - 1 participant, 2+ sessions: leave-one-session-out (a personal model tested on a new session);
  - 1 session: each phase cut into 5 contiguous blocks, 15 s removed around the test block (weak check).
--save writes models/label.joblib, models/teacher.joblib and models/headset_free.joblib (the better of the
two, which the app loads at start-up).
"""

import argparse
import csv
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clutch.learned import (ALL_FEATURES, BASELINE_S, EEG_CHANNELS, FEATURE_SETS, SMOOTH_S,  # noqa: E402
                            baseline_of, row_features, window_mean)
from clutch.processing import eeg_features, heart_rate, motion_level  # noqa: E402

FS = {"eeg": 256, "ppg": 64, "accel": 52}  # Muse 2
WINDOW_S = {"eeg": 4.0, "ppg": 10.0, "accel": 2.0}
STEP_S = 1.0
N_BLOCKS, GAP_S = 5, 15.0
# Screens where the person reads or fills in a form (report, questionnaires, ratings): alpha drops with any
# visual attention, so these would teach "reading" rather than stress.
SKIP_PHASES = {"idle", "ready", "results", "calibration_check", "calib_survey", "study_rating", "study_done"}
TEACHER_MIN_D = 0.5
PARAMS = dict(max_iter=150, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0, random_state=0)


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
    """One raw feature row per second of every phase (label NaN where the phase is unlabelled)."""
    meta = json.loads((sdir / "meta.json").read_text())
    phases = [r for r in csv.DictReader(open(sdir / "labels.csv")) if r["phase"] not in SKIP_PHASES]
    te, eeg, eeg_names = load_raw(sdir / "raw_eeg.csv")
    tp, ppg, _ = load_raw(sdir / "raw_ppg.csv")
    ta, acc, _ = load_raw(sdir / "raw_accel.csv")
    face = pd.read_csv(sdir / "face.csv") if (sdir / "face.csv").exists() else None
    keep = [i for i, n in enumerate(eeg_names) if n in EEG_CHANNELS]
    rows = []
    for k, ph in enumerate(phases):
        t0, t1 = float(ph["t_start"]), float(ph["t_end"])
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
                if len(recent):
                    f = {c: (None if pd.isna(v) else v) for c, v in recent.iloc[-1].to_dict().items()}
            rows.append({"participant": meta.get("participant", sdir.name), "session": sdir.name, "t": float(tt),
                         "phase": ph["phase"], "label": float(ph["label"]) if ph["label"] != "" else np.nan,
                         "block": k, "since_start": float(tt - t0), **row_features(feats, hr, motion, f)})
    return rows


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline-correct each session with the first BASELINE_S of its first rest block, smooth causally within
    each phase, and add the EEG teacher target."""
    out = []
    for sess, s in df.groupby("session"):
        s = s.sort_values("t").copy()
        rest_blocks = s.loc[s["label"] == 0, "block"]
        if rest_blocks.empty:
            print(f"  {sess}: no rest block, skipped")
            continue
        is_base = ((s["block"] == rest_blocks.min()) & (s["since_start"] <= BASELINE_S)).to_numpy()
        X = s[ALL_FEATURES].to_numpy(dtype=float)
        X = X - baseline_of(list(X[is_base]))
        t, blocks = s["t"].to_numpy(), s["block"].to_numpy()
        s[ALL_FEATURES] = np.vstack([window_mean(list(X[(blocks == blocks[i]) & (t > t[i] - SMOOTH_S) & (t <= t[i])]))
                                     for i in range(len(s))])
        s = s[~is_base]
        s["teacher"], chosen = teacher_index(s)
        print(f"  {sess}: teacher channels {', '.join(chosen) or 'none (no alpha drop under stress)'}")
        out.append(s)
    return pd.concat(out, ignore_index=True)


def teacher_index(s: pd.DataFrame):
    """EEG stress index: alpha drop from baseline, averaged over the channels whose alpha dropped under stress
    in this session (same rule as the app's calibration: Cohen's d >= 0.5)."""
    chosen = []
    for ch in EEG_CHANNELS:
        r = s.loc[s["label"] == 0, f"alphalog_{ch}"].dropna()
        x = s.loc[s["label"] == 1, f"alphalog_{ch}"].dropna()
        if len(r) >= 10 and len(x) >= 10:
            pooled = max(np.sqrt((r.var() + x.var()) / 2), 0.05)
            if (r.mean() - x.mean()) / pooled >= TEACHER_MIN_D:
                chosen.append(ch)
    if not chosen:
        return np.nan, chosen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return -np.nanmean(s[[f"alphalog_{ch}" for ch in chosen]].to_numpy(dtype=float), axis=1), chosen


def folds(df: pd.DataFrame):
    """Yields (name, train mask, test mask)."""
    for key in ("participant", "session"):
        groups = df[key].unique()
        if len(groups) >= 2:
            for g in groups:
                yield g, (df[key] != g).to_numpy(), (df[key] == g).to_numpy()
            return
    chunk = np.zeros(len(df), dtype=int)
    for _, g in df.groupby(["session", "block"]):
        chunk[g.index] = np.minimum((np.argsort(np.argsort(g["t"].to_numpy())) * N_BLOCKS) // len(g), N_BLOCKS - 1)
    t, sess = df["t"].to_numpy(), df["session"].to_numpy()
    for c in range(N_BLOCKS):
        test = chunk == c
        train = ~test
        for name in np.unique(sess[test]):  # purge training rows close to the test rows
            tt = np.sort(t[test & (sess == name)])
            pos = np.clip(np.searchsorted(tt, t), 1, len(tt) - 1)
            near = np.minimum(np.abs(t - tt[pos - 1]), np.abs(t - tt[pos])) <= GAP_S
            train &= ~((sess == name) & near)
        yield f"block {c + 1}", train, test


def fit(kind, X, y, lab):
    """kind 'label': classifier on labelled rows. kind 'teacher': regressor on every row with a teacher value,
    then scaled so that the median prediction at rest = 0 and under stress = 1 (like the app's anchors)."""
    if kind == "label":
        ok = ~np.isnan(lab)
        return HistGradientBoostingClassifier(**PARAMS).fit(X[ok], lab[ok]), None
    ok = ~np.isnan(y)
    m = HistGradientBoostingRegressor(**PARAMS).fit(X[ok], y[ok])
    p = m.predict(X)
    r, s = np.median(p[lab == 0]), np.median(p[lab == 1])
    return m, (float(r), float(s) if s != r else float(r) + 1e-6)


def score(kind, model, scale, X):
    if kind == "label":
        return model.predict_proba(X)[:, 1]
    r, s = scale
    return (model.predict(X) - r) / (s - r)


def evaluate(df, kind, features, importance=False):
    X = df[features].to_numpy(dtype=float)
    lab, y = df["label"].to_numpy(), df["teacher"].to_numpy(dtype=float)
    pred = np.full(len(df), np.nan)
    fold_auc, imp, n_imp = [], np.zeros(len(features)), 0
    for _, train, test in folds(df):
        cols = ~np.all(np.isnan(X[train]), axis=0)  # a feature never recorded in training cannot be used
        tl = lab[train]
        if not cols.any() or len(set(tl[~np.isnan(tl)])) < 2 or (kind == "teacher" and np.isnan(y[train]).all()):
            continue
        model, scale = fit(kind, X[train][:, cols], y[train], tl)
        pred[test] = score(kind, model, scale, X[test][:, cols])
        ev = test & ~np.isnan(lab)
        if len(set(lab[ev])) == 2:
            fold_auc.append(roc_auc_score(lab[ev], pred[ev]))
            if importance:
                Xe, le = X[ev][:, cols], lab[ev]
                sc = (lambda m, Xs, ys: roc_auc_score(ys, score(kind, m, scale, Xs)))
                r = permutation_importance(model, Xe, le, scoring=sc, n_repeats=10, random_state=0)
                imp[cols] += r.importances_mean
                n_imp += 1
    ok = ~np.isnan(pred) & ~np.isnan(lab)
    if not fold_auc:
        return {"auc": None, "note": "no usable data"}, pred
    res = {"auc": round(float(roc_auc_score(lab[ok], pred[ok])), 3), "auc_folds": [round(a, 2) for a in fold_auc],
           "balanced_acc": round(float(balanced_accuracy_score(lab[ok], pred[ok] >= 0.5)), 3)}
    if kind == "teacher":  # how well heart + face imitate the EEG on unseen data
        both = ~np.isnan(pred) & ~np.isnan(y)
        res["imitation_r"] = round(float(np.corrcoef(pred[both], y[both])[0, 1]), 3) if both.sum() > 10 else None
    if n_imp:
        res["importance"] = {k: round(float(v / n_imp), 3) for k, v in sorted(zip(features, imp), key=lambda kv: -kv[1])}
    return res, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "data" / "clutch_dataset"), help="exported dataset folder")
    p.add_argument("--notch", type=float, default=60.0, help="mains frequency (60 Hz US, 50 Hz Europe)")
    p.add_argument("--save", help="folder to save the trained models into (e.g. models)")
    args = p.parse_args()

    sdirs = sorted(d for d in (Path(args.data) / "sessions").iterdir() if (d / "labels.csv").exists())
    print("Computing features from the raw files...")
    df = normalise(pd.DataFrame([r for d in sdirs for r in session_rows(d, args.notch)]))
    lab = df["label"]
    validation = ("leave-one-participant-out" if df["participant"].nunique() >= 2 else
                  "leave-one-session-out" if df["session"].nunique() >= 2 else
                  f"{N_BLOCKS} time blocks within the session (weak: same session)")
    print(f"\n{df['session'].nunique()} sessions, {df['participant'].nunique()} participants, {len(df)} s of data "
          f"(rest {int((lab == 0).sum())}, stress {int((lab == 1).sum())}, unlabelled {int(lab.isna().sum())})")
    print(f"Validation: {validation}")
    print("Missing values: " + (", ".join(f"{k} {v:.0%}" for k, v in df[ALL_FEATURES].isna().mean().items() if v > 0.05)
                                or "none above 5%"))

    report = {"validation": validation, "sessions": sorted(df["session"].unique()), "results": {}}
    teacher_ok = ~df["teacher"].isna() & ~lab.isna()
    if teacher_ok.any() and lab[teacher_ok].nunique() == 2:
        t_auc = roc_auc_score(lab[teacher_ok], df.loc[teacher_ok, "teacher"])
        report["teacher_auc"] = round(float(t_auc), 3)
        print(f"\nThe EEG teacher itself separates rest and stress with AUC {t_auc:.2f} "
              "(channels chosen on the same data, so optimistic)")

    runs = [("A label", "label", name) for name in FEATURE_SETS] + [("B teacher", "teacher", "headset_free")]
    preds = {}
    print(f"\n{'model':10s} {'signals':13s} {'AUC':>5s} {'bal.acc':>8s} {'imitation r':>12s}   AUC per fold")
    for title, kind, name in runs:
        res, pred = evaluate(df, kind, FEATURE_SETS[name], importance=name == "headset_free")
        report["results"][f"{kind}:{name}"] = res
        preds[f"{kind}:{name}"] = pred
        if res["auc"] is None:
            print(f"{title:10s} {name:13s}  no usable data")
            continue
        imit = res.get("imitation_r")
        print(f"{title:10s} {name:13s} {res['auc']:5.2f} {res['balanced_acc']:8.2f} "
              f"{'' if imit is None else f'{imit:.2f}':>12s}   {res['auc_folds']}")

    for key in ("label:headset_free", "teacher:headset_free"):
        imp = report["results"][key].get("importance", {})
        if imp:
            print(f"\nMost useful features, {key} (drop in AUC when shuffled):")
            for k, v in list(imp.items())[:6]:
                print(f"  {k:16s} {v:+.3f}")

    # Score per phase on unseen data: relax should be low, hard tasks high, easy math in between
    print("\nMean heart + face score per phase, on unseen data (0 = calm, 1 = stress):")
    table = df[["phase"]].assign(A=preds["label:headset_free"], B=preds["teacher:headset_free"])
    print(table.groupby("phase", sort=False).mean().round(2).to_string())
    print("\nAUC: 0.5 = chance, 1.0 = perfect. imitation r: correlation with the EEG index on unseen data.")

    if args.save:
        import joblib
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        aucs = {}
        for kind in ("label", "teacher"):
            feats = [k for k in FEATURE_SETS["headset_free"] if df[k].notna().any()]
            if kind == "teacher" and df["teacher"].isna().all():
                continue
            model, scale = fit(kind, df[feats].to_numpy(dtype=float), df["teacher"].to_numpy(dtype=float),
                               lab.to_numpy())
            aucs[kind] = report["results"][f"{kind}:headset_free"].get("auc") or 0
            joblib.dump({"name": f"headset_free ({kind})", "kind": kind, "model": model, "scale": scale,
                         "features": feats, "baseline_s": BASELINE_S, "smooth_s": SMOOTH_S,
                         "trained_on": report["sessions"], "cv_auc": aucs[kind], "validation": validation},
                        out / f"{kind}.joblib")
        best = max(aucs, key=aucs.get)
        shutil.copy2(out / f"{best}.joblib", out / "headset_free.joblib")
        report["live_model"] = best
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(f"\nSaved {', '.join(f'{k}.joblib' for k in aucs)} in {out}; the app will use '{best}' "
              f"(AUC {aucs[best]:.2f}) as headset_free.joblib")


if __name__ == "__main__":
    main()
