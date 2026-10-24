"""Data sources: a real Muse through muselsl/LSL or BrainFlow, or a synthetic Muse for development.

All expose the same interface so the rest of the app does not care which one is running:
    start(), stop(), get_eeg(n), get_ppg(n), get_accel(n), status()
Each get_* returns the most recent n samples as (channels, samples), or None.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

# Quiet liblsl's INFO logging (must be set before pylsl is first imported).
os.environ.setdefault("LSLAPICFG", str(Path(__file__).with_name("lsl_api.cfg")))

MUSE_BOARDS = {
    "muse2": "MUSE_2_BOARD",          # Muse 2, native Bluetooth (Windows 10+, macOS, Linux)
    "muses": "MUSE_S_BOARD",          # Muse S, native Bluetooth
    "muse2016": "MUSE_2016_BOARD",    # original Muse, no PPG sensor
    "muse2_bled": "MUSE_2_BLED_BOARD",  # Muse 2 through a BLED112 dongle
}


class MuseSource:
    """Live Muse headset through BrainFlow.

    BrainFlow splits Muse data into three presets (streams):
      DEFAULT    -> EEG: TP9, AF7, AF8, TP10 at 256 Hz, in microvolts
      AUXILIARY  -> accelerometer + gyroscope at 52 Hz
      ANCILLARY  -> PPG (ambient, infrared, red) at 64 Hz, only if enabled with config_board
    """

    kind = "muse"

    def __init__(self, board: str = "muse2", serial_number: str = "", mac_address: str = "",
                 ppg_preset: str = "p50", timeout: int = 15):
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams, BrainFlowPresets

        self._BoardShim = BoardShim
        self._presets = BrainFlowPresets
        self.board_id = getattr(BoardIds, MUSE_BOARDS[board]).value
        params = BrainFlowInputParams()
        params.serial_number = serial_number  # the Muse name, e.g. "Muse-4A2B"; picks the right headset in a busy room
        params.mac_address = mac_address
        params.timeout = timeout
        self.board = BoardShim(self.board_id, params)
        self.ppg_preset = ppg_preset

        d, aux, anc = BrainFlowPresets.DEFAULT_PRESET, BrainFlowPresets.AUXILIARY_PRESET, BrainFlowPresets.ANCILLARY_PRESET
        available = BoardShim.get_board_presets(self.board_id)
        self.eeg_fs = BoardShim.get_sampling_rate(self.board_id, d)
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id, d)
        self.eeg_names = BoardShim.get_eeg_names(self.board_id, d)
        self.accel_fs = self.accel_channels = None
        if aux in available:
            self.accel_fs = BoardShim.get_sampling_rate(self.board_id, aux)
            self.accel_channels = BoardShim.get_accel_channels(self.board_id, aux)
        self.ppg_fs = self.ppg_channels = None
        if anc in available:
            self.ppg_fs = BoardShim.get_sampling_rate(self.board_id, anc)
            self.ppg_channels = BoardShim.get_ppg_channels(self.board_id, anc)
        self._started_at = None

    def start(self):
        self.board.prepare_session()
        if self.ppg_preset and self.ppg_channels:
            try:
                self.board.config_board(self.ppg_preset)  # turns on the optical heart sensor
            except Exception as exc:  # older firmware may refuse; EEG still works
                print(f"[muse] could not enable PPG with '{self.ppg_preset}': {exc}")
        self.board.start_stream(450000)
        self._started_at = time.time()

    def stop(self):
        try:
            self.board.stop_stream()
        finally:
            self.board.release_session()

    def _get(self, n, preset, channels):
        if channels is None:
            return None
        data = self.board.get_current_board_data(int(n), preset)
        if data.shape[1] == 0:
            return None
        return data[channels, :]

    def get_eeg(self, n):
        return self._get(n, self._presets.DEFAULT_PRESET, self.eeg_channels)

    def get_ppg(self, n):
        return self._get(n, self._presets.ANCILLARY_PRESET, self.ppg_channels)

    def get_accel(self, n):
        return self._get(n, self._presets.AUXILIARY_PRESET, self.accel_channels)

    def status(self):
        return {"kind": self.kind, "connected": self._started_at is not None}

    def sample_count(self, stream):
        preset = {"eeg": self._presets.DEFAULT_PRESET, "ppg": self._presets.ANCILLARY_PRESET,
                  "accel": self._presets.AUXILIARY_PRESET}[stream]
        return self.board.get_board_data_count(preset)

    def on_phase(self, phase):
        pass


class _Ring:
    def __init__(self, channels, size):
        self.buf = np.zeros((channels, size))
        self.count = 0
        self.lock = threading.Lock()

    def push(self, block):
        n = min(block.shape[1], self.buf.shape[1])
        if n == 0:
            return
        with self.lock:
            self.buf = np.roll(self.buf, -n, axis=1)
            self.buf[:, -n:] = block[:, -n:]
            self.count += block.shape[1]

    def last(self, n):
        with self.lock:
            n = int(min(n, self.count, self.buf.shape[1]))
            return self.buf[:, -n:].copy() if n > 0 else None


class LslSource:
    """Muse data read from Lab Streaming Layer, as published by muselsl.

    `muselsl stream --ppg --acc` creates three LSL streams named "Muse":
      type EEG -> TP9, AF7, AF8, TP10, Right AUX at 256 Hz, microvolts (we drop Right AUX)
      type PPG -> PPG1, PPG2, PPG3 at 64 Hz (optical pulse sensor)
      type ACC -> X, Y, Z at 52 Hz, in g
    The connection itself is handled by a HeadsetManager (muselsl in a child process), driven from
    the web app's Connect / Disconnect buttons, or started at launch with autoconnect=True.
    A muselsl started outside the app (another terminal) is picked up too.
    Streams are looked for continuously in the background, so the app starts before the headset
    is connected and recovers when it reconnects.
    """

    kind = "muselsl"
    EEG_KEEP = ["TP9", "AF7", "AF8", "TP10"]
    STALE_S = 10.0  # no EEG for this long: drop the streams and look for them again

    def __init__(self, name: str = "", autoconnect: bool = False):
        from .headset import HeadsetManager
        self.headset = HeadsetManager(name)
        self.autoconnect = autoconnect
        # Muse 2 defaults until the streams say otherwise
        self.eeg_fs, self.ppg_fs, self.accel_fs = 256, 64, 52
        self.eeg_names = list(self.EEG_KEEP)
        self._inlets, self._rings, self._keep, self._last_rx = {}, {}, {}, {}
        self._opened_at = None
        self.stream_id = None
        self.on_samples = None  # callback(stream, timestamps, block) for raw recording
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def _labels(info):
        labels, ch = [], info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
        return labels

    @staticmethod
    def _find(stype, timeout):
        from pylsl import resolve_byprop
        for info in resolve_byprop("type", stype, timeout=timeout):
            if info.name() == "Muse":
                return info
        return None

    def start(self, block=False, timeout=90.0):
        if self.autoconnect:
            if self._find("EEG", 1.5) is None:
                self.headset.connect()
            else:
                print("[lsl] a Muse stream is already running (started outside the app): using it")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if block:
            deadline = time.time() + timeout
            while not self.status()["connected"]:
                if time.time() > deadline:
                    raise RuntimeError("No data from the headset. Check it is on, close, and not paired to a phone.")
                time.sleep(0.2)

    def _open(self, eeg_info):
        from pylsl import StreamInlet
        found = {"eeg": eeg_info}
        for key, stype in (("ppg", "PPG"), ("accel", "ACC")):
            info = self._find(stype, 2.0)
            if info is not None:
                found[key] = info
        for key, info in found.items():
            inlet = StreamInlet(info, max_buflen=30)
            info = inlet.info(timeout=5.0)  # full metadata (channel labels) only comes with an open inlet
            labels = self._labels(info)
            if key == "eeg":
                keep = [labels.index(n) for n in self.EEG_KEEP if n in labels] or list(range(4))
                self.eeg_names = [labels[i] for i in keep] if len(keep) == 4 else list(self.EEG_KEEP)
            else:
                keep = list(range(info.channel_count()))
            fs = int(round(info.nominal_srate()))
            setattr(self, f"{key}_fs", fs)
            self._keep[key] = keep
            self._rings[key] = _Ring(len(keep), fs * 30)
            self._last_rx[key] = None
            self._inlets[key] = inlet
        self._opened_at = time.time()
        self.stream_id = eeg_info.source_id()  # muselsl uses "Muse<bluetooth address>"
        print("[lsl] streams: " + ", ".join(f"{k} {getattr(self, k + '_fs')} Hz" for k in found))

    def _close(self):
        for inlet in self._inlets.values():
            try:
                inlet.close_stream()
            except Exception:
                pass
        self._inlets, self._rings, self._keep, self._last_rx = {}, {}, {}, {}
        self._opened_at = None
        self.stream_id = None

    def _run(self):
        while not self._stop.is_set():
            if not self._inlets:
                info = self._find("EEG", 1.0)
                if info is not None:
                    try:
                        self._open(info)
                    except Exception as exc:
                        print(f"[lsl] could not open streams: {exc}")
                        self._close()
                continue
            for key, inlet in list(self._inlets.items()):
                try:
                    samples, stamps = inlet.pull_chunk(timeout=0.0, max_samples=2048)
                except Exception:
                    samples = None
                if samples:
                    block = np.asarray(samples, dtype=float).T[self._keep[key], :]
                    self._rings[key].push(block)
                    self._last_rx[key] = time.time()
                    if self.on_samples:
                        self.on_samples(key, stamps, block)
                    if key == "eeg":
                        self.headset.mark_streaming()
            last = self._last_rx.get("eeg") or self._opened_at
            if last is not None and time.time() - last > self.STALE_S:
                print("[lsl] EEG stream silent, looking for it again")
                self._close()
            time.sleep(0.02)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.headset.stop()

    def _get(self, key, n):
        ring = self._rings.get(key)
        last = self._last_rx.get(key)
        if ring is None or last is None or time.time() - last > 2.0:
            return None  # no fresh data: never compute on a frozen buffer
        return ring.last(n)

    def get_eeg(self, n):
        return self._get("eeg", n)

    def get_ppg(self, n):
        return self._get("ppg", n)

    def get_accel(self, n):
        return self._get("accel", n)

    def sample_count(self, stream):
        ring = self._rings.get(stream)
        return ring.count if ring else 0

    def status(self):
        last = self._last_rx.get("eeg")
        connected = last is not None and time.time() - last < 2.0
        hs = self.headset.status()
        hs["external"] = connected and not hs["managed"]
        hs["address"] = (self.stream_id or "").replace("Muse", "", 1) or None
        return {"kind": self.kind, "connected": connected, "headset": hs}

    def on_phase(self, phase):
        pass


class MockSource:
    """Synthetic Muse 2 for building the app without a headset.

    A hidden 'stress' level (0-1) shapes the signals the way real ones respond:
    alpha shrinks and beta grows with stress, heart rate goes from ~62 to ~95 bpm.
    Stress follows the session phase automatically, or can be forced from the UI debug panel.
    """

    kind = "mock"
    PHASE_STRESS = {"idle": 0.3, "calib_rest": 0.25, "calib_stress": 0.85, "calib_recovery": 0.4, "warmup": 0.4, "round1": 0.75,
                    "reset": 0.2, "round2": 0.5, "results": 0.3}

    def __init__(self, seed: int = 0):
        self.eeg_fs, self.ppg_fs, self.accel_fs = 256, 64, 52
        self.eeg_names = ["TP9", "AF7", "AF8", "TP10"]
        self.rng = np.random.default_rng(seed)
        self.eeg = _Ring(4, self.eeg_fs * 30)
        self.ppg = _Ring(3, self.ppg_fs * 30)
        self.acc = _Ring(3, self.accel_fs * 30)
        self.stress = 0.3
        self.target = 0.3
        self.override = None
        self._lock = threading.Lock()
        self._t = 0.0
        self._last = None
        self._beat_phase = 0.0
        self._ar = np.zeros(4)
        self._frac = {"eeg": 0.0, "ppg": 0.0, "acc": 0.0}
        self._clock = time.time()  # simulated device clock, for raw timestamps
        self.on_samples = None     # callback(stream, timestamps, block) for raw recording

    def start(self):
        self._last = time.time()
        self._clock = self._last

    def _emit(self, stream, block, fs):
        if self.on_samples and block.shape[1]:
            n = block.shape[1]
            self.on_samples(stream, list(self._clock - (n - 1 - np.arange(n)) / fs), block)

    def stop(self):
        pass

    def on_phase(self, phase):
        self.target = self.PHASE_STRESS.get(phase, 0.3)

    def set_stress(self, value):
        self.override = None if value is None else float(np.clip(value, 0, 1))

    def _advance(self):
        now = time.time()
        dt = now - self._last
        if dt < 0.02:
            return
        self._last = now
        self.step(dt)

    def _count(self, key, exact):
        """Whole samples to emit now, carrying the fraction so the long-run rate is exact."""
        total = exact + self._frac[key]
        n = int(total)
        self._frac[key] = total - n
        return n

    def step(self, dt):
        """Generate dt seconds of data (called with wall-clock time live, or directly in tests)."""
        target = self.override if self.override is not None else self.target
        self.stress += (target - self.stress) * (1 - np.exp(-dt / 8.0))  # stress drifts slowly, like a body
        s = self.stress
        rng = self.rng

        n = self._count("eeg", dt * self.eeg_fs)
        t = self._t + np.arange(n) / self.eeg_fs
        eeg = np.empty((4, n))
        for c in range(4):
            noise = np.empty(n)
            for i in range(n):  # 1/f-ish background
                self._ar[c] = 0.97 * self._ar[c] + rng.normal(0, 3.0)
                noise[i] = self._ar[c]
            alpha = (11 - 3 * s) * np.sin(2 * np.pi * 10 * t + c)
            beta = (4 + 2 * s) * rng.normal(0, 1, n)
            beta = np.convolve(beta, np.ones(3) / 3, mode="same")
            eeg[c] = noise + alpha + beta + 0.5 * rng.normal(0, 1, n)
        if rng.random() < dt / 3.5:  # blink on the forehead channels
            k = min(n, int(0.3 * self.eeg_fs))
            eeg[1:3, :k] += 120 * np.hanning(k)
        self.eeg.push(eeg)

        m = self._count("ppg", dt * self.ppg_fs)
        bpm = 64 + 12 * s
        phase = self._beat_phase + np.cumsum(np.full(m, bpm / 60 / self.ppg_fs))
        self._beat_phase = phase[-1] if m else self._beat_phase
        pulse = np.sin(2 * np.pi * phase) + 0.35 * np.sin(4 * np.pi * phase)
        ppg = np.vstack([
            1000 + rng.normal(0, 2, m),
            50000 - 300 * pulse + rng.normal(0, 40, m),
            30000 - 150 * pulse + rng.normal(0, 40, m),
        ])
        self.ppg.push(ppg)

        a = self._count("acc", dt * self.accel_fs)
        acc = np.vstack([rng.normal(0, 0.01, a), rng.normal(0, 0.01, a), 1 + rng.normal(0, 0.01, a)])
        self.acc.push(acc)
        self._t += n / self.eeg_fs
        self._clock += dt
        self._emit("eeg", eeg, self.eeg_fs)
        self._emit("ppg", ppg, self.ppg_fs)
        self._emit("accel", acc, self.accel_fs)

    def get_eeg(self, n):
        with self._lock:
            self._advance()
            return self.eeg.last(n)

    def get_ppg(self, n):
        return self.ppg.last(n)

    def get_accel(self, n):
        return self.acc.last(n)

    def status(self):
        return {"kind": self.kind, "connected": True, "true_stress": round(self.stress, 2)}
