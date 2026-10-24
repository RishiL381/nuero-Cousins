"""Pilot results: compare the breathing reset against quiet rest (control) across all sessions.

Usage:  .venv\\Scripts\\python analysis\\summarize_sessions.py [--include-quick]
Writes analysis/results.csv and prints per-session changes and per-condition averages.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["session", "participant", "condition", "stress_r1", "stress_r2", "d_stress",
          "correct_r1", "correct_r2", "d_correct", "acc_r1", "acc_r2", "d_acc", "rt_r1", "rt_r2", "d_rt"]


def diff(a, b):
    return None if a is None or b is None else b - a


def load(include_quick):
    rows = []
    for summary in sorted((ROOT / "sessions").glob("*/summary.json")):
        s = json.loads(summary.read_text())
        if s.get("aborted") or (s.get("quick") and not include_quick):
            continue
        r1, r2 = s["rounds"].get("round1", {}), s["rounds"].get("round2", {})
        p1, p2 = s["phases"]["round1"], s["phases"]["round2"]
        rows.append({
            "session": summary.parent.name, "participant": s["participant"], "condition": s["condition"],
            "stress_r1": p1["stress"], "stress_r2": p2["stress"], "d_stress": diff(p1["stress"], p2["stress"]),
            "correct_r1": r1.get("correct"), "correct_r2": r2.get("correct"),
            "d_correct": diff(r1.get("correct"), r2.get("correct")),
            "acc_r1": r1.get("accuracy"), "acc_r2": r2.get("accuracy"), "d_acc": diff(r1.get("accuracy"), r2.get("accuracy")),
            "rt_r1": r1.get("mean_rt"), "rt_r2": r2.get("mean_rt"), "d_rt": diff(r1.get("mean_rt"), r2.get("mean_rt")),
        })
    return rows


def fmt(v, d=1):
    return "—" if v is None else f"{v:+.{d}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--include-quick", action="store_true", help="include ?quick test sessions")
    args = p.parse_args()
    rows = load(args.include_quick)
    if not rows:
        print("No finished sessions found in sessions/.")
        return

    print(f"{'session':28s} {'cond':10s} {'d_stress':>9s} {'d_correct':>10s} {'d_acc%':>7s} {'d_rt s':>7s}")
    for r in rows:
        acc = None if r["d_acc"] is None else r["d_acc"] * 100
        print(f"{r['session']:28s} {r['condition']:10s} {fmt(r['d_stress']):>9s} {fmt(r['d_correct'], 0):>10s}"
              f" {fmt(acc):>7s} {fmt(r['d_rt'], 2):>7s}")

    print("\nAverage change from round 1 to round 2 (negative stress / rt = better):")
    for cond in ("breathing", "rest"):
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue

        def avg(key, scale=1.0):
            vals = [r[key] * scale for r in sub if r[key] is not None]
            return mean(vals) if vals else None
        print(f"  {cond:10s} n={len(sub):2d}  stress {fmt(avg('d_stress'))}  correct {fmt(avg('d_correct'))}"
              f"  accuracy {fmt(avg('d_acc', 100))} pts  response time {fmt(avg('d_rt'), 2)} s")
    print("\nThe breathing effect is the breathing change minus the rest change (rest captures practice effects).")
    print("With a handful of volunteers this is a pilot, not proof.")

    out = ROOT / "analysis" / "results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
