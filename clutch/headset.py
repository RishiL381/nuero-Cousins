"""Headset connection manager: runs muselsl in a child process and tracks its state from its output,
so the web app can connect, show progress, disconnect, and reconnect automatically.

States: idle, searching, connecting, streaming, retrying, not_found, disconnected, failed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_ATTEMPTS = 5

# (text printed by muselsl, new state, message for the user; None = keep a message built from the line)
PATTERNS = [
    ("Searching for Muses", "searching", "Searching for a Muse nearby (about 10 s)…"),
    ("No Muses found", "not_found", "No Muse found. Is it on (lights blinking), close, and not connected to a phone?"),
    ("Found device", "searching", None),     # printed once per Muse seen in the scan, not the one we pick
    ("Connecting to", "connecting", None),   # the headset actually chosen (matches --name when given)
    ("BLE connected", "connecting", "Bluetooth link up, setting up the headband…"),
    ("Streaming", "streaming", "Streaming"),
    ("Disconnected.", "disconnected", "The headset stopped sending data."),
    ("Failed to connect", "failed", "Could not connect to the headset."),
]


class HeadsetManager:
    def __init__(self, default_name: str = ""):
        self.name = default_name
        self.proc: subprocess.Popen | None = None
        self.state = "idle"
        self.message = "Not connected"
        self.device = None          # headset muselsl is connecting / connected to, e.g. Muse-3724
        self.seen = set()           # all Muses seen in the last scan
        self.log = deque(maxlen=40)
        self.want = False           # user asked to be connected: keep reconnecting
        self.attempts = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    # ---- public API ----
    def connect(self, name: str | None = None):
        with self._lock:
            if name is not None:
                self.name = name.strip()
            self.want = True
            self.attempts = 0
            self._launch()

    def disconnect(self):
        with self._lock:
            self.want = False
            self._kill()
            self._set("idle", "Disconnected")

    def mark_streaming(self):
        """Called by the data source once samples actually arrive."""
        if self.state != "streaming":
            self._set("streaming", "Streaming")
        self.attempts = 0

    def stop(self):
        self._stop.set()
        self.want = False
        self._kill()

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def status(self):
        return {"state": self.state, "message": self.message, "device": self.device, "name": self.name,
                "managed": self.running, "want": self.want, "attempts": self.attempts,
                "log": list(self.log)[-3:]}

    # ---- internals ----
    def _set(self, state, message):
        self.state, self.message = state, message

    def _launch(self):
        self._kill()
        self.attempts += 1
        cmd = [sys.executable, "-u", "-m", "clutch.muselsl_runner", "stream", "--ppg", "--acc", "--retries", "1"]
        if self.name:
            cmd += ["--name", self.name]
        env = dict(os.environ, PYTHONUNBUFFERED="1",
                   LSLAPICFG=str(Path(__file__).with_name("lsl_api.cfg")))  # quiet liblsl
        self.log.append(f"$ muselsl {' '.join(cmd[4:])}")
        self.seen, self.device = set(), None
        self._set("searching", "Starting the Bluetooth connection…")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, encoding="utf-8", errors="replace")
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _kill(self):
        p = self.proc
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.proc = None

    def _read(self, proc):
        for line in proc.stdout:
            line = line.rstrip()
            if not line or "netinterfaces.cpp" in line or "api_config.cpp" in line or "common.cpp" in line:
                continue
            self.log.append(line)
            if proc is not self.proc:
                continue
            for text, state, message in PATTERNS:
                if text in line:
                    if text == "Found device":
                        m = re.search(r"Found device (\S+)", line)
                        if m:
                            self.seen.add(m.group(1).rstrip(","))
                        message = f"Muses nearby: {', '.join(sorted(self.seen))}…"
                    elif text == "Connecting to":
                        m = re.search(r"Connecting to (Muse\S*?):", line)  # skip the "Connecting to <MAC>" line
                        self.device = m.group(1) if m else self.device
                        message = f"Connecting to {self.device} (can take up to 30 s)…"
                    self._set(state, message)
                    break
            else:
                if "Error" in line or "Exception" in line:
                    self.message = f"muselsl error: {line[-160:]}"

    def _watch(self):
        """Relaunch muselsl when it exits while the user still wants to be connected."""
        while not self._stop.is_set():
            time.sleep(0.5)
            with self._lock:
                p = self.proc
                if p is None or p.poll() is None:
                    continue
                self.proc = None
                if not self.want:
                    continue
                if self.attempts >= MAX_ATTEMPTS:
                    self.want = False
                    self._set("failed", f"Gave up after {MAX_ATTEMPTS} attempts. Check the headset is on and close "
                                        "to the laptop, then click Connect.")
                    continue
                reason = self.message if self.state in ("not_found", "disconnected", "failed") else "Connection lost."
                self._set("retrying", f"{reason} Retrying ({self.attempts + 1}/{MAX_ATTEMPTS})…")
            time.sleep(3)
            with self._lock:
                if self.want and self.proc is None:
                    self._launch()
