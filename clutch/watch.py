"""Heart rate from a watch or chest strap over Bluetooth LE (standard Heart Rate Service, 0x180D).

Works with Garmin watches that broadcast heart rate over Bluetooth (watch menu: "Broadcast Heart Rate"),
and with any standard BLE heart rate strap (Polar, Wahoo...). Older Garmin watches broadcast over ANT+ only,
which needs an ANT+ USB stick and is not supported here.

Each notification carries the heart rate and, when the device sends them, the beat-to-beat (RR) intervals,
from which we compute RMSSD, a standard short-term heart rate variability measure (Task Force, 1996).
Runs its own asyncio loop in a background thread; the engine only reads latest() and rmssd().
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque

HR_SERVICE = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT = "00002a37-0000-1000-8000-00805f9b34fb"
SCAN_S = 8.0
MAX_AGE_S = 5.0        # a heart rate older than this is not used
RMSSD_WINDOW_S = 30.0
MAX_TRIES = 3          # failed searches or lost connections in a row before letting the watch go
RETRY_PAUSE_S = 3.0


def parse_measurement(data: bytes) -> tuple[int, list[float]]:
    """Heart Rate Measurement characteristic (Bluetooth SIG): returns (bpm, RR intervals in ms)."""
    flags = data[0]
    i = 1
    if flags & 0x01:  # 16-bit heart rate
        bpm = int.from_bytes(data[i:i + 2], "little")
        i += 2
    else:
        bpm = data[i]
        i += 1
    if flags & 0x08:  # energy expended present
        i += 2
    rr = []
    if flags & 0x10:  # RR intervals present, units of 1/1024 s
        while i + 1 < len(data):
            rr.append(int.from_bytes(data[i:i + 2], "little") * 1000 / 1024)
            i += 2
    return bpm, rr


def rmssd_of(rr_ms: list[float]):
    """RMSSD (ms) of successive RR intervals, or None with fewer than 10 plausible beats. Used live and in training."""
    rr = [r for r in rr_ms if 300 < r < 2000]
    if len(rr) < 10:
        return None
    diffs = [b - a for a, b in zip(rr, rr[1:])]
    diffs = [d for d in diffs if abs(d) < 250]  # drop jumps from missed or extra beats
    return round((sum(d * d for d in diffs) / len(diffs)) ** 0.5, 1) if len(diffs) >= 8 else None


class WatchHR:
    def __init__(self):
        self.state = "off"          # off, searching, connecting, streaming, error
        self.message = "Not connected"
        self.name = None
        self.address = None
        self.on_sample = None       # callback(wall_time, bpm, rr_ms_list), set by the engine for logging
        self._want = False
        self._hint = ""
        self._bpm, self._bpm_t = None, 0.0
        self._rr = deque()          # (wall_time, rr_ms)
        self._lock = threading.Lock()
        self._thread = None

    # ---------- control ----------
    def connect(self, name_hint: str | None = None):
        self._hint = (name_hint or "").strip().lower()
        self._want = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
            self._thread.start()

    def disconnect(self):
        self._want = False

    # ---------- readings ----------
    def latest(self):
        with self._lock:
            return self._bpm if self._bpm is not None and time.time() - self._bpm_t < MAX_AGE_S else None

    def rmssd(self):
        """RMSSD (ms) over the last 30 s of RR intervals, or None without enough beats."""
        with self._lock:
            now = time.time()
            rr = [r for t, r in self._rr if now - t < RMSSD_WINDOW_S]
        return rmssd_of(rr)

    def status(self) -> dict:
        return {"state": self.state, "message": self.message, "name": self.name, "address": self.address,
                "bpm": self.latest(), "rmssd": self.rmssd(), "rr": bool(self._rr)}

    # ---------- BLE loop ----------
    def _set(self, state, message):
        self.state, self.message = state, message

    def _on_notify(self, _sender, data: bytearray):
        bpm, rr = parse_measurement(bytes(data))
        now = time.time()
        with self._lock:
            if bpm > 0:
                self._bpm, self._bpm_t = bpm, now
            self._rr.extend((now, r) for r in rr)
            while self._rr and now - self._rr[0][0] > 120:
                self._rr.popleft()
        if self.on_sample:
            self.on_sample(now, bpm, rr)

    async def _find(self):
        from bleak import BleakScanner
        self._set("searching", "Searching for a watch broadcasting heart rate (about 8 s)…")
        found = await BleakScanner.discover(timeout=SCAN_S, return_adv=True)
        candidates = []
        for device, adv in found.values():
            name = adv.local_name or device.name or ""
            has_hr = HR_SERVICE in [u.lower() for u in (adv.service_uuids or [])]
            if (self._hint and self._hint in name.lower()) or (not self._hint and has_hr):
                candidates.append((adv.rssi or -999, device, name))
        if not candidates:
            return None, None
        _, device, name = max(candidates, key=lambda c: c[0])  # the closest one
        return device, name

    async def _run(self):
        """Connects on demand; after MAX_TRIES failures in a row it stops, so the app never keeps the watch
        busy and another computer can connect to it."""
        from bleak import BleakClient
        tries = 0
        while self._want:
            if tries >= MAX_TRIES:
                self._want = False
                self._set("error", "Watch not found or taken by another computer. Click Retry to try again.")
                break
            tries += 1
            try:
                device, name = await self._find()
                if device is None:
                    self._set("searching", f"No watch found yet (try {tries}/{MAX_TRIES}). On the watch, start "
                              "\"Broadcast Heart Rate\".")
                    await asyncio.sleep(RETRY_PAUSE_S)
                    continue
                self.name, self.address = name or device.address, device.address
                self._set("connecting", f"Connecting to {self.name}…")
                async with BleakClient(device) as client:
                    await client.start_notify(HR_MEASUREMENT, self._on_notify)
                    self._set("streaming", f"Streaming from {self.name}")
                    tries = 0
                    while self._want and client.is_connected:
                        await asyncio.sleep(0.5)
                    if client.is_connected:
                        await client.stop_notify(HR_MEASUREMENT)
                if self._want:
                    self._set("searching", f"Connection lost, reconnecting (try {tries + 1}/{MAX_TRIES})…")
            except Exception as exc:  # BLE errors: show them and retry while the user wants the watch
                self._set("error", f"{type(exc).__name__}: {exc}"[:200])
                await asyncio.sleep(RETRY_PAUSE_S)
        if self.state != "error":
            self._set("off", "Not connected")
        with self._lock:
            self._bpm = None
