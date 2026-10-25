"""Build a shareable, labelled dataset from recorded sessions.

Usage (from the project folder):
    .venv\\Scripts\\python tools\\export_dataset.py              -> dataset_export/ + clutch_dataset_<date>.zip
    .venv\\Scripts\\python tools\\export_dataset.py --no-zip

Only sessions whose participant gave consent are exported. Participant codes are replaced by
P01, P02... (the mapping stays on this computer, in dataset_export_mapping.json, outside the zip).
No images are ever recorded: the face data are numbers computed in the browser.
"""

import argparse
import csv
import json
import os
import shutil
import stat
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from clutch.learned import LABELS  # noqa: E402  (0 = calm, 1 = stressed, other phases unlabelled)
FEATURE_COLS = ["t", "phase", "label", "alpha", "beta", "theta", "ratio_log", "hr_bpm", "hr_quality", "motion",
                "usable_channels", "contact", "face_brow", "face_press", "face_squint", "face_frown", "face_smile",
                "blink_rate", "head_motion", "stress", "stress_body", "stress_ml"] + \
               [f"{b}_{ch}" for ch in ("TP9", "AF7", "AF8", "TP10")
                for b in ("alpha", "beta", "theta", "gamma", "alphashare", "alphalog")] + \
               [f"{k}_{ch}" for k in ("q", "muscle") for ch in ("TP9", "AF7", "AF8", "TP10")]
RAW_FILES = ["raw_eeg.csv", "raw_ppg.csv", "raw_accel.csv", "face.csv", "events.csv"]


def remove_tree(path):
    """shutil.rmtree that also removes read-only folders (OneDrive marks synced folders read-only on Windows)."""
    def retry(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)


def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def phase_intervals(events, end_t):
    """[(start, end, phase)] from the phase-change events."""
    changes = [(float(e["t"]), e["phase"]) for e in events if e["name"] == "phase"]
    out = []
    for i, (t, phase) in enumerate(changes):
        t_end = changes[i + 1][0] if i + 1 < len(changes) else end_t
        if t_end > t:
            out.append((t, t_end, phase))
    return out


def calibration_info(events):
    for e in reversed(events):
        if e["name"] == "calibration":
            try:
                return json.loads(e["data"])
            except json.JSONDecodeError:
                return {}
    return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sessions", default=str(ROOT / "sessions"))
    p.add_argument("--out", default=str(ROOT / "dataset_export"))
    p.add_argument("--no-zip", action="store_true")
    args = p.parse_args()

    sessions_dir, out = Path(args.sessions), Path(args.out)
    if out.exists():
        remove_tree(out)
    (out / "sessions").mkdir(parents=True)

    mapping, rows_all, table = {}, [], []
    skipped = []
    for sdir in sorted(d for d in sessions_dir.iterdir() if d.is_dir()) if sessions_dir.exists() else []:
        meta_path = sdir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if not meta.get("consent"):
            skipped.append(f"{sdir.name} (no consent)")
            continue
        metrics = read_csv(sdir / "metrics.csv")
        events = read_csv(sdir / "events.csv")
        if not metrics:
            skipped.append(f"{sdir.name} (empty)")
            continue
        who = str(meta.get("participant", "anon"))
        code = mapping.setdefault(who, f"P{len(mapping) + 1:02d}")
        session_id = f"{code}_S{sum(1 for r in table if r['participant'] == code) + 1:02d}"

        intervals = phase_intervals(events, float(metrics[-1]["t"]) + 0.5)
        labelled = defaultdict(float)  # seconds per label, from the 0.5 s ticks
        for r in metrics:
            lab = LABELS.get(r["phase"], "")
            if lab != "":
                labelled[lab] += 0.5
            rows_all.append({"participant": code, "session": session_id,
                             **{k: (lab if k == "label" else r.get(k, "")) for k in FEATURE_COLS}})

        # per-session folder: raw signals, events, labels, anonymised meta
        dst = out / "sessions" / session_id
        dst.mkdir()
        for name in RAW_FILES:
            if (sdir / name).exists():
                shutil.copy2(sdir / name, dst / name)
        with open(dst / "labels.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_start", "t_end", "phase", "label"])
            for t0, t1, phase in intervals:
                w.writerow([round(t0, 2), round(t1, 2), phase, LABELS.get(phase, "")])
        clean_meta = {k: v for k, v in meta.items() if k != "participant"}
        clean_meta["participant"] = code
        (dst / "meta.json").write_text(json.dumps(clean_meta, indent=2))

        cal = calibration_info(events)
        stai_b = (cal.get("stai_before") or {}).get("score")
        stai_a = (cal.get("stai_after") or {}).get("score")
        table.append({"participant": code, "session": session_id, "kind": meta.get("kind", "session"),
                      "protocol": meta.get("protocol") or cal.get("protocol"), "stressor": meta.get("stressor"),
                      "rest_s": round(labelled[0]), "stress_s": round(labelled[1]),
                      "has_raw_eeg": (sdir / "raw_eeg.csv").exists(),
                      "has_face": any(r.get("face_brow") for r in metrics),
                      "stai_before": stai_b, "stai_after": stai_a,
                      "signals_used": ",".join(cal.get("used", []))})

    if not table:
        print("No exportable session found (record a calibration with consent ticked first).")
        if skipped:
            print("Skipped:", ", ".join(skipped))
        sys.exit(1)

    with open(out / "features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["participant", "session"] + FEATURE_COLS)
        w.writeheader()
        w.writerows(rows_all)
    with open(out / "sessions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    shutil.copy2(ROOT / "docs" / "dataset.md", out / "README.md")
    shutil.copy2(ROOT / "docs" / "methods.md", out / "methods.md")
    (out.parent / "dataset_export_mapping.json").write_text(json.dumps(mapping, indent=2))  # stays local, not zipped

    n_lab = sum(1 for r in rows_all if r["label"] != "")
    print(f"Exported {len(table)} sessions from {len(mapping)} participants -> {out}")
    print(f"features.csv: {len(rows_all)} rows (0.5 s each), {n_lab} labelled (0 = rest, 1 = stress)")
    for r in table:
        print(f"  {r['session']}: rest {r['rest_s']} s, stress {r['stress_s']} s, face={r['has_face']}, "
              f"raw EEG={r['has_raw_eeg']}, STAI {r['stai_before']} -> {r['stai_after']}")
    if skipped:
        print("Skipped:", ", ".join(skipped))
    if not args.no_zip:
        zip_base = out.parent / f"clutch_dataset_{time.strftime('%Y%m%d')}"
        path = shutil.make_archive(str(zip_base), "zip", root_dir=out)
        print(f"Zip ready to send: {path}")


if __name__ == "__main__":
    main()
