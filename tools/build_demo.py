"""Build the data of the static demo site (demo/): a real recording replayed through the app's own code.

Usage (from the project folder):
    .venv\\Scripts\\python tools\\build_demo.py                     -> demo/data.json from P01_S02

The session goes through the same steps as live:
  1. its calibration (rest, stress test, recovery) feeds StressModel: at the end, finalize() sets the EEG teacher
     (bands on the person's alpha frequency, direction learned) and trains the personal heart + face student;
  2. the whole session is then scored second by second with that calibration: brain score (teacher) and heart +
     face score (student). The study blocks after the calibration are data neither model ever saw.
Also written: the sensors' state per second (contact, muscle), band power per sensor relative to rest, heart rate,
the 7 face indicators, the calibration report, and the model results (aggregates only, from models/report.json and
from replaying the other sessions). Only the chosen session's own data is written; other people appear as numbers.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import train_model as tm  # noqa: E402
from clutch.engine import BRAIN_BANDS, CALIB_PHASES, EEG_CHANNELS, Config, StressModel  # noqa: E402
from clutch.learned import FACE_FEATURES, brain_values, features_from  # noqa: E402

HARD = {"study_relax": 0, "study_math_hard": 1, "study_stroop_hard": 1}


def values_of(x, iaf):
    v = {f"{b}_{ch}": None for ch in EEG_CHANNELS for b in BRAIN_BANDS}
    if iaf is not None:
        v.update(brain_values(x, iaf))
    v.update({"hr": x["hr"], "hrv": x["hrv"], **{k: x[k] for k in FACE_FEATURES}})
    return v


def replay(sdir: Path):
    """(rows, calibration, per-second scores) for one session, through the app's code."""
    with contextlib.redirect_stdout(io.StringIO()):
        rows = tm.session_rows(sdir, 60)
    model = StressModel(Config(tick_s=1.0))
    for r in rows:
        if r["phase"] in CALIB_PHASES:
            model.collect(r["phase"], values_of(r["inputs"], None), None, r["inputs"])
    cal = model.finalize()
    model._smoothed = {}
    scores = []
    for r in rows:
        v = values_of(r["inputs"], model.iaf)
        model.score(v, 1.0)
        scores.append((model.brain_rule, model.personal_score(v, 1.0)))
    return rows, cal, scores, model.iaf


def unseen_auc(rows, scores):
    """Teacher and student AUC on the study blocks after the calibration: relax (0) vs hard tasks (1)."""
    y, b, h = [], [], []
    for r, (brain, body) in zip(rows, scores):
        if r["phase"] in HARD and brain is not None and body is not None:
            y.append(HARD[r["phase"]]); b.append(brain); h.append(body)
    if len(set(y)) < 2:
        return None
    return {"teacher": round(roc_auc_score(y, b), 2), "student": round(roc_auc_score(y, h), 2),
            "follow_r": round(float(np.corrcoef(b, h)[0, 1]), 2)}


def r1(v, n=1):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", default=str(ROOT / "data" / "clutch_dataset" / "sessions" / "P01_S02"))
    p.add_argument("--others", nargs="*", default=[str(ROOT / "data" / "clutch_dataset" / "sessions" / "P02_S01")],
                   help="other people's sessions: only their aggregate results are written")
    p.add_argument("--out", default=str(ROOT / "demo" / "data.json"))
    args = p.parse_args()

    rows, cal, scores, iaf = replay(Path(args.session))
    # rest medians of each band, to show band power as a change from the person's calm
    rest = {}
    for ch in EEG_CHANNELS:
        for b in BRAIN_BANDS:
            vals = [f[f"{b}_iaf_{ch}"] for f in (features_from(r["inputs"], iaf) for r in rows if r["phase"] == "calib_rest")
                    if f[f"{b}_iaf_{ch}"] is not None]
            rest[f"{b}_{ch}"] = float(np.median(vals)) if vals else None

    ticks = []
    for r, (brain, body) in zip(rows, scores):
        x = r["inputs"]
        f = features_from(x, iaf)
        sensors = {}
        for ch in EEG_CHANNELS:
            state = "bad" if x["psd"].get(ch) is None else "muscle" if x["muscle"][ch] else "good"
            bands = {b: (r1(100 * (10 ** (f[f"{b}_iaf_{ch}"] - rest[f"{b}_{ch}"]) - 1), 0)
                         if f[f"{b}_iaf_{ch}"] is not None and rest[f"{b}_{ch}"] is not None else None) for b in BRAIN_BANDS}
            sensors[ch] = {"s": state, **bands}
        ticks.append({"p": r["phase"], "b": r1(brain), "h": r1(body), "hr": r1(x["hr"], 0),
                      "f": [r1(x[k], 3) if k not in ("blink_rate",) else r1(x[k], 1) for k in FACE_FEATURES],
                      "e": sensors})

    sig = cal["signals"]
    report = {
        "iaf": iaf, "check": (cal.get("check") or {}).get("note"), "check_passed": (cal.get("check") or {}).get("passed"),
        "brain": {k: {"d": r1(v.get("effect"), 2), "dir": v.get("direction"), "used": bool(v.get("used"))}
                  for k, v in sig.items() if k.split("_")[0] in BRAIN_BANDS},
        "body": {k: {"d": r1(v.get("effect"), 2), "dir": v.get("direction"), "used": bool(v.get("used"))}
                 for k, v in sig.items() if k in ["hr", *FACE_FEATURES]},
        "student": {k: (cal.get("personal") or {}).get(k) for k in ("features", "weights", "fit_r", "seconds")},
    }
    own = unseen_auc(rows, scores)
    people = [{"who": "Person 1 (this recording)", **(own or {})}]
    for i, other in enumerate(args.others, 2):
        o_rows, _, o_scores, _ = replay(Path(other))
        res = unseen_auc(o_rows, o_scores)
        if res:
            people.append({"who": f"Person {i}", **res})
    shared = {}
    rep_path = ROOT / "models" / "report.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        shared = {"validation": rep.get("validation"), "sessions": len(rep.get("sessions", [])),
                  "seconds": rep.get("seconds"),
                  "auc": {k: rep["results"][k]["auc"] for k in ("label:brain", "label:muscle", "label:heart", "label:face",
                                                                 "label:headset_free") if k in rep.get("results", {})}}

    out = {"session": Path(args.session).name, "step_s": 1, "face_keys": FACE_FEATURES, "channels": list(EEG_CHANNELS),
           "ticks": ticks, "report": report, "results": {"per_person": people, "shared": shared}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(ticks)} seconds from {out['session']} -> {args.out} ({Path(args.out).stat().st_size // 1024} KB)")
    print("unseen study blocks:", people)


if __name__ == "__main__":
    main()
