"""Train a stress model on Clutch recordings, two ways, and compare them on data the model never saw.

Usage (from the project folder):
    .venv\\Scripts\\python tools\\train_model.py                          -> evaluate on data/clutch_dataset
    .venv\\Scripts\\python tools\\train_model.py --data dataset_export --save models

Features are recomputed from the raw files of each session (raw_eeg.csv, raw_ppg.csv, raw_accel.csv,
raw_hr.csv, face.csv) with the same code as the live app (clutch/learned.py), one row per second of every
phase: brain bands set on the individual alpha frequency, muscle (30-44 Hz), heart and face.
Each feature is taken relative to the first 45 s of the session's first rest block; those seconds are then
left out of training and evaluation.

Two ways to learn:
  A. label    the target is what the person was doing: rest = 0, stressor = 1 (labelled phases only).
  B. teacher  the brain teaches the body: a brain model (A on brain features) gives its stress estimate for
              every second, labelled or not, and a heart + face model learns to imitate it. The EEG estimate
              is cross-fitted (each second is scored by an EEG model that did not train on it) so the student
              learns what the brain generalises, not what it memorised.
Both are judged the same way: does the heart + face score separate the labelled rest and stress seconds of
a session the model never saw (AUC)? The brain, headset (brain + muscle) and full models are evaluated
alike: the best one is the "with headset" model the app uses while the Muse is on.

Validation, chosen automatically:
  - 2+ participants: leave-one-participant-out;
  - 1 participant, 2+ sessions: leave-one-session-out (a personal model tested on a new session);
  - 1 session: each phase cut into 5 contiguous blocks, 15 s removed around the test block (weak check).
--save writes models/label.joblib, models/teacher.joblib, models/headset_free.joblib (the better of the two)
and models/with_headset.joblib (the best of brain, headset and full); the app loads both at start-up.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clutch.learned import (ALL_FEATURES, BASELINE_S, EEG_CHANNELS, FEATURE_SETS, SMOOTH_S,  # noqa: E402
                            WITH_HEADSET_SETS, baseline_of, estimate_iaf, features_from, row_inputs, to_vector,
                            window_mean)
from clutch.processing import eeg_features, heart_rate, motion_level  # noqa: E402
from clutch.watch import MAX_AGE_S, RMSSD_WINDOW_S, rmssd_of  # noqa: E402

FS = {"eeg": 256, "ppg": 64, "accel": 52}  # Muse 2
WINDOW_S = {"eeg": 4.0, "ppg": 10.0, "accel": 2.0}
STEP_S = 1.0
N_BLOCKS, GAP_S = 5, 15.0
# Screens where the person reads or fills in a form (report, questionnaires, ratings): alpha drops with any
# visual attention, so these would teach "reading" rather than stress.
SKIP_PHASES = {"idle", "ready", "results", "calibration_check", "calib_survey", "study_rating", "study_done"}
MIN_LIVE_AUC = 0.6  # a model is installed in the app only above this AUC on unseen data
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


def session_rows(sdir: Path, notch_hz: float, session_id: str | None = None):
    """One row per second of every phase (label NaN where the phase is unlabelled), holding the raw inputs
    (spectra, heart, face) in column "inputs"; features are computed in normalise() once the IAF is known."""
    meta = json.loads((sdir / "meta.json").read_text())
    phases = [r for r in csv.DictReader(open(sdir / "labels.csv")) if r["phase"] not in SKIP_PHASES]
    te, eeg, eeg_names = load_raw(sdir / "raw_eeg.csv")
    tp, ppg, _ = load_raw(sdir / "raw_ppg.csv")
    ta, acc, _ = load_raw(sdir / "raw_accel.csv")
    face = pd.read_csv(sdir / "face.csv") if (sdir / "face.csv").exists() else None
    watch = pd.read_csv(sdir / "raw_hr.csv") if (sdir / "raw_hr.csv").exists() else None  # Garmin / BLE strap
    keep = [i for i, n in enumerate(eeg_names) if n in EEG_CHANNELS]
    rows = []
    for k, ph in enumerate(phases):
        t0, t1 = float(ph["t_start"]), float(ph["t_end"])
        for tt in np.arange(t0 + WINDOW_S["eeg"], t1, STEP_S):
            w = window(te, eeg, tt, WINDOW_S["eeg"], FS["eeg"])
            feats = eeg_features(w[keep], FS["eeg"], [eeg_names[i] for i in keep], notch_hz) if w is not None else {}
            w = window(tp, ppg, tt, WINDOW_S["ppg"], FS["ppg"])
            hr = heart_rate(w, FS["ppg"]) if w is not None else {}
            if watch is not None:  # same rule as the live engine: the watch replaces the Muse PPG when fresh
                recent = watch[(watch["t"] <= tt) & (watch["t"] > tt - MAX_AGE_S)]
                if len(recent):
                    beats = watch.loc[(watch["t"] <= tt) & (watch["t"] > tt - RMSSD_WINDOW_S), "rr_ms"].dropna()
                    rr = [float(x) for v in beats for x in str(v).split()]
                    hr = {"bpm": float(recent["bpm"].iloc[-1]), "quality": 1.0, "rmssd": rmssd_of(rr)}
            w = window(ta, acc, tt, WINDOW_S["accel"], FS["accel"])
            motion = motion_level(w) if w is not None else {}
            f = None
            if face is not None:
                recent = face[(face["t"] <= tt) & (face["t"] > tt - 2)]
                if len(recent):
                    f = {c: (None if pd.isna(v) else v) for c, v in recent.iloc[-1].to_dict().items()}
            rows.append({"participant": meta.get("participant", sdir.name), "session": session_id or sdir.name, "t": float(tt),
                         "phase": ph["phase"], "label": float(ph["label"]) if ph["label"] != "" else np.nan,
                         "block": k, "since_start": float(tt - t0), "inputs": row_inputs(feats, hr, motion, f)})
    return rows


def collect_sessions(folders: list[Path]) -> list[tuple[str, Path]]:
    """(session id, folder) for every session of the given datasets, each recording once: a session exported
    both here and in the team dataset has the same start time. Ids are made unique if two datasets reuse one."""
    seen, out = set(), []
    for folder in folders:
        for d in sorted((folder / "sessions").iterdir() if (folder / "sessions").exists() else []):
            if not (d / "labels.csv").exists():
                continue
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            key = meta.get("started_at") or f"{folder}/{d.name}"
            if key in seen:
                continue
            seen.add(key)
            sid = d.name
            while sid in {s for s, _ in out}:
                sid += "b"
            out.append((sid, d))
    print(f"{len(out)} sessions from {', '.join(str(f) for f in folders)}")
    return out


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Per session: estimate the IAF on the first BASELINE_S of the first rest block (as live), compute the
    features with bands set on it, baseline-correct with those same seconds, smooth causally within each phase."""
    out = []
    for sess, s in df.groupby("session"):
        s = s.sort_values("t").copy()
        rest_blocks = s.loc[s["label"] == 0, "block"]
        if rest_blocks.empty:
            print(f"  {sess}: no rest block, skipped")
            continue
        is_base = ((s["block"] == rest_blocks.min()) & (s["since_start"] <= BASELINE_S)).to_numpy()
        iaf = estimate_iaf(list(s.loc[is_base, "inputs"]))
        print(f"  {sess}: individual alpha frequency {iaf:.1f} Hz")
        X = np.vstack([to_vector(features_from(x, iaf)) for x in s["inputs"]])
        s = s.drop(columns="inputs").assign(iaf=iaf)
        X = X - baseline_of(list(X[is_base]))
        t, blocks = s["t"].to_numpy(), s["block"].to_numpy()
        s[ALL_FEATURES] = np.vstack([window_mean(list(X[(blocks == blocks[i]) & (t > t[i] - SMOOTH_S) & (t <= t[i])]))
                                     for i in range(len(s))])
        out.append(s[~is_base])
    return pd.concat(out, ignore_index=True)


def eeg_teacher(sub: pd.DataFrame) -> np.ndarray:
    """The brain's own stress estimate (0-1) for every row of sub, labelled or not: an EEG model trained on
    the labelled rows, cross-fitted with folds() so that no row is scored by a model that trained on it."""
    sub = sub.reset_index(drop=True)
    X, lab = sub[FEATURE_SETS["brain"]].to_numpy(dtype=float), sub["label"].to_numpy()
    out = np.full(len(sub), np.nan)
    for _, tr, te in folds(sub):
        ok = tr & ~np.isnan(lab)
        cols = ~np.all(np.isnan(X[ok]), axis=0)
        if len(set(lab[ok])) == 2 and cols.any():
            m = HistGradientBoostingClassifier(**PARAMS).fit(X[ok][:, cols], lab[ok])
            out[te] = m.predict_proba(X[te][:, cols])[:, 1]
    return out


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


def evaluate(df, kind, features, importance=False, train_frac=1.0):
    """Out-of-fold scores. train_frac < 1 keeps only the start of every training phase (learning curve)."""
    X = df[features].to_numpy(dtype=float)
    early = (df["block_pos"] <= train_frac).to_numpy()
    lab = df["label"].to_numpy()
    pred, brain = np.full(len(df), np.nan), np.full(len(df), np.nan)  # brain: EEG estimate on the test rows
    fold_auc, imp, n_imp = [], np.zeros(len(features)), 0
    for _, train, test in folds(df):
        train = train & early
        cols = ~np.all(np.isnan(X[train]), axis=0)  # a feature never recorded in training cannot be used
        tl = lab[train]
        if not cols.any() or len(set(tl[~np.isnan(tl)])) < 2:
            continue
        y = None
        if kind == "teacher":  # the brain teaches: EEG estimate on the training rows, then on the test rows
            y = eeg_teacher(df[train])
            if np.isnan(y).all():
                continue
            E, ok = df[FEATURE_SETS["brain"]].to_numpy(dtype=float), train & ~np.isnan(lab)
            ecols = ~np.all(np.isnan(E[ok]), axis=0)
            brain[test] = HistGradientBoostingClassifier(**PARAMS).fit(E[ok][:, ecols], lab[ok]).predict_proba(
                E[test][:, ecols])[:, 1]
        model, scale = fit(kind, X[train][:, cols], y, tl)
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
    if kind == "teacher":  # how well heart + face imitate the brain on unseen data
        both = ~np.isnan(pred) & ~np.isnan(brain)
        res["imitation_r"] = round(float(np.corrcoef(pred[both], brain[both])[0, 1]), 3) if both.sum() > 10 else None
    if n_imp:
        res["importance"] = {k: round(float(v / n_imp), 3) for k, v in sorted(zip(features, imp), key=lambda kv: -kv[1])}
    return res, pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", action="append", help="exported dataset folder; repeat it to combine several "
                   "(default: data/clutch_dataset). A session found in several folders is used once.")
    p.add_argument("--notch", type=float, default=60.0, help="mains frequency (60 Hz US, 50 Hz Europe)")
    p.add_argument("--save", help="folder to save the trained models into (e.g. models)")
    args = p.parse_args()

    sdirs = collect_sessions([Path(d) for d in (args.data or [ROOT / "data" / "clutch_dataset"])])
    print("Computing features from the raw files...")
    df = normalise(pd.DataFrame([r for sid, d in sdirs for r in session_rows(d, args.notch, sid)]))
    df["block_pos"] = df.groupby(["session", "block"])["t"].rank(pct=True)
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
    runs = [("A label", "label", name) for name in FEATURE_SETS] + [("B teacher", "teacher", "headset_free")]
    preds = {}
    print(f"\n{'model':10s} {'signals':13s} {'AUC':>5s} {'bal.acc':>8s} {'imitation r':>12s}   AUC per fold")
    for title, kind, name in runs:
        res, pred = evaluate(df, kind, FEATURE_SETS[name], importance=name in ("headset_free", "headset"))
        report["results"][f"{kind}:{name}"] = res
        preds[f"{kind}:{name}"] = pred
        if res["auc"] is None:
            print(f"{title:10s} {name:13s}  no usable data")
            continue
        imit = res.get("imitation_r")
        print(f"{title:10s} {name:13s} {res['auc']:5.2f} {res['balanced_acc']:8.2f} "
              f"{'' if imit is None else f'{imit:.2f}':>12s}   {res['auc_folds']}")

    report["teacher_auc"] = report["results"]["label:brain"].get("auc")  # the teacher: the brain on its own
    with_headset = max(WITH_HEADSET_SETS, key=lambda k: report["results"][f"label:{k}"].get("auc") or 0)
    report["with_headset"] = with_headset
    report["iaf"] = {sess: round(float(v), 2) for sess, v in df.groupby("session")["iaf"].first().items()}
    for key in ("label:headset_free", "teacher:headset_free", "label:headset"):
        imp = report["results"][key].get("importance", {})
        if imp:
            print(f"\nMost useful features, {key} (drop in AUC when shuffled):")
            for k, v in list(imp.items())[:6]:
                print(f"  {k:16s} {v:+.3f}")

    # Score per phase on unseen data: relax should be low, hard tasks high, easy math in between
    print("\nMean score per phase, on unseen data (0 = calm, 1 = stress). A, B: heart + face; H: with headset")
    table = df[["phase"]].assign(A=preds["label:headset_free"], B=preds["teacher:headset_free"],
                                 H=preds[f"label:{with_headset}"])
    print(table.groupby("phase", sort=False).mean().round(2).to_string())
    print("\nAUC: 0.5 = chance, 1.0 = perfect. imitation r: correlation with the EEG model on unseen data.")
    report["phase_means"] = {ph: {k: (None if pd.isna(v) else round(float(v), 3)) for k, v in row.items()}
                             for ph, row in table.groupby("phase", sort=False).mean().iterrows()}

    # Learning curve: the same validation, training on the first 25/50/75/100% of every phase
    print("\nLearning curve (AUC on unseen data vs share of each training phase used):")
    report["learning_curve"] = {"fractions": [0.25, 0.5, 0.75, 1.0], "label": [], "teacher": []}
    for frac in report["learning_curve"]["fractions"]:
        for kind in ("label", "teacher"):
            auc = (report["results"][f"{kind}:headset_free"]["auc"] if frac == 1.0
                   else evaluate(df, kind, FEATURE_SETS["headset_free"], train_frac=frac)[0]["auc"])
            report["learning_curve"][kind].append(auc)
        print(f"  {frac:4.0%}: label {report['learning_curve']['label'][-1]}  teacher {report['learning_curve']['teacher'][-1]}")

    # Held-out score over time, per session (every 2 s), for the charts in the app
    report["timeline"] = {}
    for sess, g in df.assign(A=preds["label:headset_free"], B=preds["teacher:headset_free"],
                             H=preds[f"label:{with_headset}"]).groupby("session"):
        g = g.sort_values("t").iloc[::2]
        t0 = float(g["t"].iloc[0])
        report["timeline"][sess] = [[round(r.t - t0, 1), r.phase, None if pd.isna(r.label) else int(r.label),
                                     None if pd.isna(r.A) else round(float(r.A), 3),
                                     None if pd.isna(r.B) else round(float(np.clip(r.B, -0.5, 1.5)), 3),
                                     None if pd.isna(r.H) else round(float(r.H), 3)]
                                    for r in g.itertuples()]
    report["seconds"] = {"rest": int((lab == 0).sum()), "stress": int((lab == 1).sum()), "unlabelled": int(lab.isna().sum())}

    if args.save:
        import joblib
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        aucs = {}
        brain = eeg_teacher(df)
        for kind in ("label", "teacher"):
            feats = [k for k in FEATURE_SETS["headset_free"] if df[k].notna().any()]
            if kind == "teacher" and np.isnan(brain).all():
                continue
            model, scale = fit(kind, df[feats].to_numpy(dtype=float), brain, lab.to_numpy())
            aucs[kind] = report["results"][f"{kind}:headset_free"].get("auc") or 0
            joblib.dump({"name": f"headset_free ({kind})", "kind": kind, "model": model, "scale": scale,
                         "features": feats, "baseline_s": BASELINE_S, "smooth_s": SMOOTH_S,
                         "trained_on": report["sessions"], "cv_auc": aucs[kind], "validation": validation},
                        out / f"{kind}.joblib")
        best = max(aucs, key=aucs.get)
        report["live_model"] = best
        # A model that does not beat chance clearly on unseen data must not drive the app: the app then falls
        # back to the personal model of the current calibration.
        if aucs[best] >= MIN_LIVE_AUC:
            shutil.copy2(out / f"{best}.joblib", out / "headset_free.joblib")
        else:
            (out / "headset_free.joblib").unlink(missing_ok=True)
            report["live_model"] = None
            print(f"\nNo heart + face model installed: best AUC {aucs[best]:.2f} < {MIN_LIVE_AUC} on unseen data")
        feats = [k for k in FEATURE_SETS[with_headset] if df[k].notna().any()]
        model, _ = fit("label", df[feats].to_numpy(dtype=float), None, lab.to_numpy())
        if (report["results"][f"label:{with_headset}"].get("auc") or 0) >= MIN_LIVE_AUC:
            joblib.dump({"name": f"with headset ({with_headset})", "kind": "label", "model": model, "scale": None,
                         "features": feats, "baseline_s": BASELINE_S, "smooth_s": SMOOTH_S,
                         "trained_on": report["sessions"], "cv_auc": report["results"][f"label:{with_headset}"]["auc"],
                         "validation": validation}, out / "with_headset.joblib")
        else:
            (out / "with_headset.joblib").unlink(missing_ok=True)
        report["trained_at"] = time.strftime("%Y-%m-%d %H:%M")
        (out / "report.json").write_text(json.dumps(report))
        hist_path = out / "history.json"  # one entry per training, to show progress as the dataset grows
        history = json.loads(hist_path.read_text()) if hist_path.exists() else []
        history.append({"trained_at": report["trained_at"], "sessions": len(report["sessions"]),
                        "seconds": report["seconds"], "validation": validation, "live_model": best,
                        "with_headset": with_headset,
                        "auc": {k: v.get("auc") for k, v in report["results"].items()}})
        hist_path.write_text(json.dumps(history, indent=2))
        print(f"\nSaved in {out}: without headset '{best}' (AUC {aucs[best]:.2f}), with headset "
              f"'{with_headset}' (AUC {report['results'][f'label:{with_headset}']['auc']:.2f})")


if __name__ == "__main__":
    main()
