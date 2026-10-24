# Clutch: Staying Sharp Under Pressure

A Muse 2 measures stress in real time while you solve timed mental arithmetic. A guided breathing
reset brings stress down, and a second round shows the effect on performance.

## Quick start (Windows)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Headset data comes through [muse-lsl](https://github.com/alexandrebarachant/muse-lsl)
(`muselsl stream --ppg --acc`), which publishes the Muse on Lab Streaming Layer.
On Windows it uses the built-in Bluetooth (Bleak backend), so no BlueMuse or dongle is needed.

1. **Check the headset** (Muse on, lights blinking, Bluetooth on, headset not connected to a phone):
   ```bash
   .venv\Scripts\python tools\check_muse.py --eyes-test
   ```
   In a room with several Muses, pass the name of yours: `--name Muse-XXXX`
   (`.venv\Scripts\muselsl list` shows the names nearby).
2. **Run the app with the headset**, then open http://localhost:8000
   ```bash
   .venv\Scripts\python -m clutch --name Muse-XXXX
   ```
   This starts muselsl for you. To run muselsl yourself in another terminal instead
   (handy to watch it with `muselsl view`), use:
   ```bash
   .venv\Scripts\muselsl stream --ppg --acc --name Muse-XXXX
   ```
   ```bash
   .venv\Scripts\python -m clutch --source lsl
   ```
   `--source brainflow` is kept as a fallback if muselsl will not connect.
3. **Or without a headset:**
   - Simulated signals inside the app, for building the UI:
     ```bash
     .venv\Scripts\python -m clutch --source mock
     ```
     Open http://localhost:8000/?quick&debug for 2-minute sessions and a slider that forces the simulated stress.
   - A fake muselsl stream, to test the LSL path end to end:
     ```bash
     .venv\Scripts\python tools\fake_muselsl.py
     ```
     Then run `--source lsl` as above.

Keys: `Esc` aborts a session, `S` hides the live stress pill, `D` toggles the debug panel.

## What the Muse 2 gives us

| LSL stream (type) | Content | Rate |
|---|---|---|
| `EEG` | TP9, AF7, AF8, TP10 in microvolts (reference at Fpz, centre of forehead) + Right AUX (unused) | 256 Hz |
| `ACC` | accelerometer X, Y, Z in g (`--acc`) | 52 Hz |
| `PPG` | optical pulse, 3 channels (`--ppg`) | 64 Hz |

- **AF7 / AF8 (forehead):** good contact, but blinks show up as big slow spikes.
- **TP9 / TP10 (behind the ears):** need skin contact (hair is the usual problem); jaw clenching shows up as fast muscle noise.
- **Mains noise** is 60 Hz in the US; the app notches it (`--notch 50` in Europe).
- If PPG does not appear, the app still works on EEG alone.

## How the stress score works

Every 0.5 s, on the last 4 s of EEG and 10 s of PPG (`clutch/engine.py`, `clutch/processing.py`):

1. **EEG:** filter 1-45 Hz and notch mains, then compute band powers (Welch).
   The calm feature is `log10(alpha / beta)`, the median over the channels with usable contact.
   Alpha (8-12 Hz) rises when relaxed; beta (13-30 Hz) rises with tension.
2. **Heart:** band-pass 0.7-3.5 Hz, pick the cleanest optical channel, and detect beats.
   The quality score drops when beat intervals are irregular or disagree with the spectral peak.
3. **Calibration (60 s, eyes open):** stores your median and spread for both signals.
4. **Score:** `z = 0.6 * z_eeg + 0.4 * z_hr` relative to baseline, `score = 50 + 12 * z`,
   smoothed over about 5 s and clipped to 0-100. **50 = your baseline**, 65+ is clearly above it.
5. **Rejected windows:** bad-contact channels are dropped; windows with head motion are skipped.

Known limits:
- Mental effort raises beta on its own, not only stress. That is why we always compare the same task before and after, and combine EEG with heart rate.
- Closing the eyes makes alpha jump, so participants keep their eyes open.

## Sea widget

**Calibrate (60 s)** on the home screen, then **Pop out widget**. A small window with a sailboat
appears and stays on top of your other apps:
- Calm mind: flat sea, sun and gulls.
- Rising stress: waves grow, clouds gather.
- High stress: storm, rain and lightning, and the boat bounces on the swell.

The always-on-top window uses Chrome/Edge Document Picture-in-Picture. Other browsers get a normal
small window (`/widget.html`), which can also be opened on its own, for example on a second screen.

## Session flow

Calibration 60 s → warm-up 30 s (not scored) → **round 1** 90 s under pressure → **reset** 90 s → **round 2** 90 s → results.

- **Reset condition:** *breathing* (guided, 6 breaths per minute, live feedback) or *rest* (control, no feedback).
- **Round 1 pressure:** 6 s per problem, ticking clock, error buzzer.

## Evaluation

Each session is saved to `sessions/<time>_<participant>/`:
- `metrics.csv`: every tick
- `events.csv`: every answer and phase change
- `summary.json`

Run both conditions on each volunteer, in random order, then:

```bash
.venv\Scripts\python analysis\summarize_sessions.py
```

The breathing effect is the breathing change minus the rest change (rest captures practice effects).

## Project layout

```
clutch/processing.py   EEG features, contact quality, heart rate, motion (pure functions)
clutch/sources.py      LslSource (muselsl, default), MuseSource (BrainFlow fallback), MockSource (simulated)
clutch/engine.py       real-time loop, baseline, stress score, session logging
clutch/server.py       FastAPI + WebSocket, serves frontend/
frontend/              the web app (no build step); widget.js draws the sea scene
tools/check_muse.py    headset diagnostic + eyes-closed alpha test
tools/fake_muselsl.py  simulated Muse published on LSL like muselsl
tests/test_pipeline.py end-to-end test on simulated data
analysis/              pilot results across sessions
```

Wellness prototype, not a medical device.
