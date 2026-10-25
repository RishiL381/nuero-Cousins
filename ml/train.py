"""Compare stress models on held-out people, tested the way Clutch is used live: calibrate, then score.

Run: python ml/epistress.py   (once, builds data/epistress_windows.csv)
     python ml/train.py [--labels rating|condition]                     EPIStress (default data)
     python ml/train.py --data data/clutch_dataset_windows.csv --cal-seconds 60 --extra data/epistress_windows.csv
         rating:    the person's own 1-5 stress rating per task (EPIStress default)
         condition: EPIStress only: relaxation video = calm, hard tasks = stressed
         --extra:   more people (EPIStress) added to every training fold, never tested
Writes ml/results_<name>.md.

Test protocol (for every held-out person)
- Folds hold out whole people (GroupKFold by subject): the model never saw them during training.
- Calibration = the first --cal-seconds (default 120) of their calm anchor (calib_rest, or the relaxation
  video) and of one stress anchor (calib_stress, or hard arithmetic if rated stressful, else their first stressful task).
- Features are z-scored with that person's calibration windows only (per-person normalisation).
- Scored on every other window: the rest of the anchors plus all their other tasks.

Models
- chance:              guesses with the class proportions
- app two-anchor:      exactly the live app's score: alpha/beta and heart rate each placed between the person's
                       rest and stress medians, kept only with a medium response (Cohen's d >= 0.5), weighted by d
- logistic regression: one model for everyone
- LightGBM:            gradient-boosted trees, one model for everyone
- nearest centroid:    the person's calm and stress averages; the closer one wins
- centroid + LDA:      same averages, compared along the calm->stress direction learned by shrinkage LDA
                       on the training people; score 0 = their calm, 1 = their stress
With --extra, logistic regression and LightGBM are also trained on the extra people only, and on both.
Macro-F1 averages the F1 of "calm" and "stressed" (always guessing "stressed" scores about 0.4).
AUC: chance that a random stressed window scores higher than a random calm one (0.5 = coin flip).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import EEG_FEATURES, FEATURES, HEART_FEATURES, MOTION_FEATURES, STEP_S  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "epistress_windows.csv"
N_FOLDS = 6
CAL_WINDOWS = 24   # 2 min of 10 s windows with a 5 s step (set from --cal-seconds)
CALM_ANCHORS = ["calib_rest", "relaxation_video", "n_back_easy", "sudoku_easy", "arithmetix_easy", "stroop_easy"]
STRESS_ANCHORS = ["calib_stress", "arithmetix_hard", "stroop_hard", "arithmetix_easy", "stroop_easy", "n_back_hard", "sudoku_hard"]
HIGH_FREQ = [f for f in EEG_FEATURES if f.startswith(("beta_", "gamma_"))]
# The live app's signals (clutch/engine.py): expected direction under stress and floor on the spread
APP_SIGNALS = {"alpha_beta_log": (-1, 0.05), "hr_bpm": (+1, 2.0)}
POPULATION_MODELS = ("logistic regression", "LightGBM")

FEATURE_SETS = {
    "all": FEATURES,
    "EEG + heart, no motion": EEG_FEATURES + HEART_FEATURES,
    "EEG only": EEG_FEATURES,
    "heart only": HEART_FEATURES,
    "motion only": MOTION_FEATURES,
    "EEG w/o beta+gamma, heart": [f for f in EEG_FEATURES if f not in HIGH_FREQ and f != "alpha_beta_log"] + HEART_FEATURES,
}


def calibration(df: pd.DataFrame) -> pd.Series:
    """Mark each person's calibration windows: first CAL_WINDOWS of a calm anchor and of a stress anchor."""
    cal = pd.Series(False, index=df.index)
    df.attrs["anchors"] = {}
    for subj, g in df.groupby("subject"):
        tasks = g.groupby("task").label.first()
        first = list(dict.fromkeys(g.task))   # tasks in recording order, as a fallback
        calm = next((t for t in CALM_ANCHORS + first if tasks.get(t) == 0), None)
        stress = next((t for t in STRESS_ANCHORS + first if tasks.get(t) == 1), None)
        if calm is None or stress is None:
            continue  # cannot calibrate this person: used for training only
        df.attrs["anchors"][subj] = (calm, stress)
        for t in (calm, stress):
            cal[g[g.task == t].sort_values("t").index[:CAL_WINDOWS]] = True
    return cal


def normalise(df: pd.DataFrame, cal: pd.Series) -> pd.DataFrame:
    """z-score each person's features with their calibration windows (all windows if not calibratable)."""
    out = df.copy()
    for subj, g in df.groupby("subject"):
        ref = g[cal[g.index]] if cal[g.index].any() else g
        out.loc[g.index, FEATURES] = (g[FEATURES] - ref[FEATURES].mean()) / (ref[FEATURES].std() + 1e-9)
    return out


def prepare(raw: pd.DataFrame):
    """Calibration mask and normalised copy, with the raw features kept for the app's own score."""
    cal = calibration(raw)
    df = normalise(raw, cal)
    df.attrs = {"anchors": raw.attrs["anchors"]}
    return df, cal, raw[list(APP_SIGNALS)]


def lgbm(n_rows: int = 10000):
    # each leaf needs enough windows to mean something, but not so many that small datasets cannot split
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=int(np.clip(n_rows / 50, 5, 40)),
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          class_weight="balanced", random_state=0, verbose=-1)


def centroid_scores(Xtr, ytr, Xcal, ycal, Xte, use_lda):
    """Position of each test window between the person's calm (0) and stress (1) centroids."""
    calm, stress = Xcal[ycal == 0].mean(0), Xcal[ycal == 1].mean(0)
    if use_lda:
        w = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtr, ytr).coef_[0]
    else:
        w = stress - calm   # plain straight line between the two averages
    denom = w @ (stress - calm)
    if abs(denom) < 1e-9:
        return np.full(len(Xte), 0.5)
    return (Xte - calm) @ w / denom


def app_scores(cal_rows: pd.DataFrame, ycal, test_rows: pd.DataFrame) -> np.ndarray:
    """The live app's two-anchor score (clutch/engine.py StressModel), on raw window features."""
    parts, weights = [], []
    for col, (direction, floor) in APP_SIGNALS.items():
        r, s = cal_rows[col][ycal == 0].dropna(), cal_rows[col][ycal == 1].dropna()
        if len(r) < 2 or len(s) < 2:   # the app needs 10 ticks of 0.5 s = 5 s; one 10 s window covers that
            continue
        rm, sm = r.median(), s.median()
        d = direction * (sm - rm) / max(np.sqrt((r.var(ddof=0) + s.var(ddof=0)) / 2), floor)
        if d >= 0.5 and sm != rm:
            parts.append(((test_rows[col] - rm) / (sm - rm)).clip(-0.5, 1.5).to_numpy())
            weights.append(min(d, 2.0))
    if not parts:   # the app's fallback: distance from rest only (3 rest SDs = 1)
        for col, (direction, floor) in APP_SIGNALS.items():
            r = cal_rows[col][ycal == 0].dropna()
            if len(r) >= 2:
                parts.append((direction * (test_rows[col] - r.median()) / max(r.std(ddof=0), floor) / 3).clip(-0.5, 1.5).to_numpy())
                weights.append(1.0)
    if not parts:
        return np.full(len(test_rows), 0.5)
    P = np.vstack(parts)
    W = np.array(weights)[:, None] * ~np.isnan(P)
    return np.where(W.sum(0) > 0, np.nansum(P * W, 0) / np.maximum(W.sum(0), 1e-9), 0.5)


def run(df, cal, raw_app, cols, model, extra=None):
    """Out-of-fold scores (higher = more stressed) for every non-calibration window of calibratable people.
    extra: (normalised extra people, their labels) added to population models' training ("+ EPIStress"),
    or used alone ("EPIStress only")."""
    test_mask = df.subject.isin(df.attrs["anchors"]) & ~cal
    scores = pd.Series(np.nan, index=df.index)
    X = df[cols].fillna(0.0)   # 0 = the person's calibration average after z-scoring
    base = next((m for m in POPULATION_MODELS if model.startswith(m)), model)
    for tr_idx, te_idx in GroupKFold(n_splits=min(N_FOLDS, df.subject.nunique())).split(df, df.label, df.subject):
        tr, te = df.index[tr_idx], df.index[te_idx]
        te_score = te[test_mask[te]]
        if base in POPULATION_MODELS:
            Xtr, ytr = X.loc[tr], df.label[tr]
            if extra is not None and model.endswith("+ EPIStress"):
                Xtr, ytr = pd.concat([Xtr, extra[0][cols].fillna(0.0)]), pd.concat([ytr, extra[1]])
            elif extra is not None and model.endswith("(EPIStress only)"):
                Xtr, ytr = extra[0][cols].fillna(0.0), extra[1]
            m = (LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000) if base == "logistic regression"
                 else lgbm(len(Xtr)))
            m.fit(Xtr, ytr)
            scores[te_score] = m.predict_proba(X.loc[te_score])[:, 1]
        elif model == "chance":
            rng = np.random.default_rng(0)
            scores[te_score] = rng.random(len(te_score)) < df.label[tr].mean()
        else:
            for subj in df.subject[te_score].unique():
                s = te_score[df.subject[te_score] == subj]
                c = df.index[(df.subject == subj) & cal]
                if model == "app two-anchor":
                    scores[s] = app_scores(raw_app.loc[c], df.label[c].values, raw_app.loc[s])
                else:
                    scores[s] = centroid_scores(X.loc[tr].values, df.label[tr].values, X.loc[c].values,
                                                df.label[c].values, X.loc[s].values, model == "centroid + LDA")
    return scores[test_mask]


def metrics(df, scores, threshold=0.5):
    y = df.label[scores.index]
    pred = (scores >= threshold).astype(int)
    other = ~df.loc[scores.index].apply(lambda r: r.task in df.attrs["anchors"][r.subject], axis=1)
    auc = roc_auc_score(y, scores) if scores.nunique() > 2 else float("nan")
    f1_other = f1_score(y[other], pred[other], average="macro") if y[other].nunique() == 2 else float("nan")
    return f1_score(y, pred, average="macro"), auc, f1_other, confusion_matrix(y, pred, labels=[0, 1])


def condition_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """Label by what the person was doing instead of their rating: relaxation video = calm, hard tasks = stressed."""
    out = raw[raw.task.eq("relaxation_video") | raw.task.str.endswith("_hard")].copy()
    out["label"] = out.task.str.endswith("_hard").astype(int)
    return out.reset_index(drop=True)


def per_person(df, scores):
    """AUC of each held-out person: does the score rank their stressed windows above their calm ones?"""
    out = {}
    for subj, s in scores.groupby(df.subject[scores.index]):
        y = df.label[s.index]
        if y.nunique() == 2:
            out[subj] = roc_auc_score(y, s)
    return pd.Series(out)


def load_labelled(path: Path, label_mode: str = "rating") -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw[raw.label.notna()].reset_index(drop=True)   # unlabelled phases are not used for training
    raw["label"] = raw.label.astype(int)
    if label_mode == "condition":
        raw = condition_labels(raw)
    return raw


def main(argv=None):
    global CAL_WINDOWS
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--labels", choices=["rating", "condition"], default="rating")
    p.add_argument("--cal-seconds", type=float, default=120)
    p.add_argument("--extra", type=Path, default=None, help="extra training people, e.g. data/epistress_windows.csv")
    a = p.parse_args(argv)
    CAL_WINDOWS = int(round(a.cal_seconds / STEP_S))
    name = a.data.stem.replace("_windows", "") + (f"_{a.labels}" if a.data == DATA else "")

    raw = load_labelled(a.data, a.labels)
    df, cal, raw_app = prepare(raw)
    extra = None
    if a.extra:
        is_epistress = "relaxation_video" in set(pd.read_csv(a.extra, usecols=["task"]).task)
        ex_df, _, _ = prepare(load_labelled(a.extra, "condition" if is_epistress else "rating"))
        extra = (ex_df, ex_df.label)
        print(f"extra training people: {ex_df.subject.nunique()} ({len(ex_df)} windows) from {a.extra.name}")

    n_test = df.subject.isin(df.attrs["anchors"]).sum() - cal.sum()
    lines = [f"# Stress model comparison ({name}, held-out people, calibrated like the live app)", "",
             f"{len(df)} windows of 10 s from {df.subject.nunique()} people. "
             f"{len(df.attrs['anchors'])} people have both a calm and a stress anchor and are tested: "
             f"{n_test} test windows after removing {a.cal_seconds:.0f} s + {a.cal_seconds:.0f} s of calibration each."
             + (f" Extra training people: {extra[0].subject.nunique()} from {a.extra.name} (never tested)." if extra else ""), "",
             "Macro-F1 'other tasks' = only tasks not used for calibration (new activities for that person).", "",
             "| Features | Model | Macro-F1 | AUC | Macro-F1 other tasks |", "|---|---|---|---|---|"]
    results = {}
    model_names = ["chance", "app two-anchor", "logistic regression", "LightGBM", "nearest centroid", "centroid + LDA"]
    if extra is not None:
        model_names += [f"{m} {v}" for m in POPULATION_MODELS for v in ("+ EPIStress", "(EPIStress only)")]
    for set_name, cols in FEATURE_SETS.items():
        for model in model_names:
            if model in ("chance", "app two-anchor") and set_name != "all":
                continue  # these do not depend on the feature set
            s = run(df, cal, raw_app, cols, model, extra)
            f1, auc, f1o, cm = metrics(df, s)
            results[(set_name, model)] = (f1, auc, f1o, cm)
            label = "alpha/beta + heart rate" if model == "app two-anchor" else set_name
            lines.append(f"| {label} | {model} | {f1:.2f} | {auc:.2f} | {f1o:.2f} |")
            print(f"{label:26s} {model:36s} F1 {f1:.2f}  AUC {auc:.2f}  F1(other tasks) {f1o:.2f}", flush=True)

    pp = per_person(df, run(df, cal, raw_app, FEATURES, "nearest centroid"))
    lines += ["", "## Per person (nearest centroid, all features)", "",
              f"AUC per held-out person: median {pp.median():.2f}; {(pp >= 0.7).sum()} of {len(pp)} people at 0.7 or more, "
              f"{(pp < 0.5).sum()} below 0.5 (score points the wrong way).", "",
              " ".join(f"{k} {v:.2f}" for k, v in pp.sort_values(ascending=False).items())]
    print(lines[-3]); print(lines[-1])

    best = max(((k, v) for k, v in results.items() if k[0] == "all"), key=lambda kv: kv[1][0])
    (set_name, model), (f1, auc, f1o, cm) = best
    lines += ["", f"## Confusion matrix: {model}, all features (rows = true, columns = predicted)", "",
              "| | calm | stressed |", "|---|---|---|",
              f"| **calm** | {cm[0, 0]} | {cm[0, 1]} |", f"| **stressed** | {cm[1, 0]} | {cm[1, 1]} |"]

    corr = raw[HIGH_FREQ + MOTION_FEATURES].corr(method="spearman")["motion_std_g"].drop("motion_std_g")
    lines += ["", "## Motion check", "",
              f"Spearman correlation of head motion with beta/gamma power: median {corr.median():.2f}, "
              f"strongest {corr.abs().max():.2f} ({corr.abs().idxmax()}). "
              "Compare the 'motion only' and 'EEG w/o beta+gamma' rows: if motion alone scores close to 'all', "
              "the model is mostly reading movement."]

    X = df[FEATURES].fillna(0.0)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, df.label)
    w = pd.Series(lda.coef_[0] * X.std().values, index=FEATURES)   # weight x spread = influence
    final = lgbm(len(X)).fit(X, df.label)
    gain = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES)
    lines += ["", "## What the models lean on", "",
              "LDA direction (positive = higher when stressed), top 10 by influence:", ""]
    lines += [f"- `{k}` {'+' if w[k] > 0 else '-'}{abs(w[k]) / w.abs().sum():.0%}" for k in w.abs().sort_values(ascending=False).index[:10]]
    lines += ["", "LightGBM gain, top 10:", ""]
    lines += [f"- `{k}` {v / max(gain.sum(), 1e-9):.0%}" for k, v in gain.sort_values(ascending=False).head(10).items()]

    (HERE / "models").mkdir(exist_ok=True)
    joblib.dump({"lgbm": final, "lda_weights": pd.Series(lda.coef_[0], index=FEATURES), "features": FEATURES,
                 "normalize": "z-score with the person's calibration windows", "labels": {0: "calm", 1: "stressed"}},
                HERE / "models" / f"stress_model_{name}.joblib")
    (HERE / f"results_{name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[lines.index("## Motion check") - 1:]))
    print(f"\nsaved {HERE / 'models' / f'stress_model_{name}.joblib'} and {HERE / f'results_{name}.md'}")


if __name__ == "__main__":
    main()
