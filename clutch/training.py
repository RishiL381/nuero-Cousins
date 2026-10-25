"""Dataset and model steps run from the web app: build the anonymised dataset, train the model, report.

The work is done by the same scripts as on the command line (tools/export_dataset.py, tools/train_model.py),
run one at a time in a background process; their output is shown live in the app.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .learned import LABELS

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset_export"
MODELS = ROOT / "models"


class Jobs:
    """Runs one script at a time and keeps its output."""

    def __init__(self):
        self._lock = threading.Lock()
        self.state = {"name": None, "running": False, "ok": None, "log": [], "started": None, "ended": None}

    def start(self, name: str, args: list[str], on_done=None) -> bool:
        with self._lock:
            if self.state["running"]:
                return False
            self.state = {"name": name, "running": True, "ok": None, "log": [], "started": time.time(), "ended": None}
        threading.Thread(target=self._run, args=(args, on_done), daemon=True).start()
        return True

    def _run(self, args, on_done):
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore",
                   PYTHON_COLORS="0", NO_COLOR="1")
        try:
            proc = subprocess.Popen([sys.executable, *args], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                with self._lock:
                    self.state["log"] = (self.state["log"] + [line.rstrip()])[-300:]
            ok = proc.wait() == 0
        except Exception as exc:  # e.g. the script is missing
            ok = False
            self.state["log"].append(f"{type(exc).__name__}: {exc}")
        if ok and on_done:
            on_done()
        with self._lock:
            self.state.update(running=False, ok=ok, ended=time.time())

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.state, log=list(self.state["log"]))


def recorded_sessions(sessions_dir: Path) -> list[dict]:
    """Recorded sessions with their consent and seconds of labelled data (rest / stress)."""
    out = []
    for d in sorted(sessions_dir.iterdir()) if sessions_dir.exists() else []:
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            continue
        rest = stress = 0.0
        if (d / "metrics.csv").exists():
            with open(d / "metrics.csv", newline="") as f:
                for row in csv.DictReader(f):
                    lab = LABELS.get(row.get("phase"))
                    rest += 0.5 if lab == 0 else 0
                    stress += 0.5 if lab == 1 else 0
        out.append({"folder": d.name, "participant": str(meta.get("participant", "")), "kind": meta.get("kind", "session"),
                    "consent": bool(meta.get("consent")), "rest_s": round(rest), "stress_s": round(stress),
                    "has_raw_eeg": (d / "raw_eeg.csv").exists(), "has_face": (d / "face.csv").exists()})
    return out


def dataset_info() -> dict | None:
    table = DATASET / "sessions.csv"
    if not table.exists():
        return None
    with open(table, newline="") as f:
        rows = list(csv.DictReader(f))
    zips = sorted(ROOT.glob("clutch_dataset_*.zip"), key=lambda p: p.stat().st_mtime)
    return {"sessions": len(rows), "participants": len({r["participant"] for r in rows}),
            "rest_s": sum(int(r["rest_s"] or 0) for r in rows), "stress_s": sum(int(r["stress_s"] or 0) for r in rows),
            "built": time.strftime("%Y-%m-%d %H:%M", time.localtime(table.stat().st_mtime)),
            "folder": str(DATASET), "zip": str(zips[-1]) if zips else None}


def model_report() -> tuple[dict | None, list]:
    def load(name):
        p = MODELS / name
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except json.JSONDecodeError:
            return None
    return load("report.json"), load("history.json") or []
