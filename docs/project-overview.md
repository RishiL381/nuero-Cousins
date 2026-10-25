# Clutch: Project Overview

UC Berkeley neurotech hackathon, October 2026.
**Track:** General Track, *"How can neurotechnology expand human cognitive potential?"*

This document records what we built, why, what works, what does not yet, and where the project is going next.

---

## 1. The idea

Stress degrades working memory, attention and decision-making exactly when they matter most: in exams, in a pitch, before a decisive shot.

Clutch measures stress in real time with a Muse 2 EEG headband, helps the user bring it back down with a guided breathing reset, and shows the effect on performance on a timed mental task.

The pitch in one line: **we show how much stress costs you cognitively, and we help you get it back.**

---

## 2. How we got here

### Hardware available at the event

| Type | Devices |
|---|---|
| Consumer EEG | 3 × Muse, Neurosity Crown, Neurosity Notion |
| Research EEG | Emotiv, Emotiv 32-channel, 3 × g.tec 8-channel, Mentalab Explore |
| Other | Mira AR glasses, an unidentified "blue box", 2 more headsets after lunch |
| Openwater stations | Open-Motion (optical blood flow) and Open-LIFU (focused ultrasound), shared, not lendable |

### Options considered

| Idea | Decision | Reason |
|---|---|---|
| Openwater sponsor tracks | Not chosen | They require Openwater's own data and tools (Open-Motion / Open-LIFU) |
| Controlling an object with EEG | Not chosen | Motor imagery is near chance on consumer EEG in a weekend; SSVEP is feasible but risky |
| Nap optimizer (sleep staging) | Dropped | Needs model training on sleep datasets: out of scope for a weekend |
| Mind Garden / Brain Tug-of-War | Kept as inspiration | They became the sea widget |
| **Stress → breathing reset → performance** | **Chosen** | No model training, measurable, answers the track question directly |

### Why stress
What consumer EEG measures reliably: physical actions (blinks, jaw clenches), general states (calm vs tense, focused vs distracted, alert vs drowsy) and visual attention (SSVEP, P300).
What it cannot measure: thoughts, words or specific emotions. Stress versus calm sits among the most reliable signals.

---

## 3. What Clutch does today

### Session flow (about 6 minutes)
1. **Calibration (60 s):** the user sits still with eyes open. The app records a personal baseline.
2. **Warm-up (30 s):** practice arithmetic, not scored.
3. **Round 1 (90 s):** timed two-digit arithmetic under pressure, with 6 s per problem, a ticking clock and an error buzzer.
4. **Reset (90 s):** either guided breathing at 6 breaths per minute with live feedback, or quiet rest (the control condition).
5. **Round 2 (90 s):** same task, same difficulty.
6. **Results:**
   - a headline sentence;
   - stress, correct answers, accuracy and response time, before and after;
   - **what stress costs you:** accuracy and speed grouped by the stress level at the moment of each answer;
   - the stress timeline across the session.

### Sea widget
- After a standalone calibration, an always-on-top window shows a sailboat whose sea follows the stress score.
- **Calm:** flat sea, sun, gulls. **High stress:** storm, rain, lightning, and the boat pitching on the swell.
- It uses Chrome/Edge Document Picture-in-Picture, with a fallback to a normal popup (`/widget.html`).

### Data logging
Every session is saved to `sessions/<time>_<participant>/`:
- `metrics.csv`: every tick;
- `events.csv`: every answer, with the stress at that moment;
- `summary.json`.

`analysis/summarize_sessions.py` compares the breathing condition with the rest condition across volunteers.

---

## 4. How the stress score works

### What the Muse 2 measures
| Stream | Content | Rate |
|---|---|---|
| EEG | TP9, TP10 (behind the ears), AF7, AF8 (forehead), referenced to the centre of the forehead, in µV | 256 Hz |
| PPG | Optical pulse sensor on the forehead, 3 channels | 64 Hz |
| Accelerometer | Head motion, in g | 52 Hz |

The electrodes are dry. **Every sensor, including the reference in the centre of the forehead, must touch bare skin.**

### Pipeline (every 0.5 s)
1. **Clean:** band-pass 1–45 Hz and notch 60 Hz mains, on the last 4 s of each channel.
2. **Frequency analysis:** Welch power spectrum (averaged FFTs) for each channel.
3. **Band power:** alpha (8–12 Hz, relaxed) and beta (13–30 Hz, alert or tense).
4. **Calm feature:** `log10(alpha / beta)` per channel, then the **median** over channels with good contact.
5. **Heart rate:** beat detection on the cleanest PPG channel, cross-checked against the spectral peak.
6. **Score:**
   - both signals are compared with the calibration baseline as z-scores;
   - they are combined as 60% EEG and 40% heart;
   - the score is `50 + 12 × z`, smoothed over about 5 s, from 0 to 100;
   - **50 = your own baseline.**
7. **Quality control:**
   - channels with bad contact are excluded;
   - windows with head motion are skipped;
   - when the headset stops streaming, no score is shown.

### Scientific basis (cited from memory, verify before quoting)
- **Alpha rhythm and relaxation:** Berger (1929); Klimesch (1999), *Brain Research Reviews*.
- **Band power estimation:** Welch (1967), *IEEE Trans. Audio and Electroacoustics*.
- **Muse validity for research:** Krigolson et al. (2017), *Frontiers in Neuroscience*.
- **Arithmetic as a stressor:** Kirschbaum et al. (1993), Trier Social Stress Test; Dedovic et al. (2005), Montreal Imaging Stress Task.
- **Stress and prefrontal function:** Arnsten (2009), *Nature Reviews Neuroscience*.
- **Heart rate variability and stress:** Kim et al. (2018), *Psychiatry Investigation*.
- **Slow breathing:**
  - Lehrer & Gevirtz (2014), *Frontiers in Psychology*;
  - Zaccaro et al. (2018), *Frontiers in Human Neuroscience*.

### Honest limits
- Alpha/beta as a stress index is debated, and effects are modest on a 4-channel dry headset.
- The weights, the scaling and the thresholds are engineering choices, not validated parameters.
- Mental effort raises beta on its own, which is why we always compare the same task before and after, and add heart rate.
- This is a wellness prototype, not a medical device.

---

## 5. Engineering log: problems hit and fixes

| Problem | Cause | Fix |
|---|---|---|
| muselsl crashed on connect (`assert self._result`) | Race condition in Bleak's Windows backend | `clutch/muselsl_runner.py` patches Bleak to wait for the result |
| Every packet failed to decode | bitstring 5.0 removed `Bits(bytes=...)`, which muselsl uses | Pinned `bitstring<5` |
| muselsl disconnected after about 50 s | It quits after only 3 s without a packet; the room's Bluetooth is crowded (up to 214 devices) | The wrapper raises the delay to 30 s |
| LSL channel names were empty | Full metadata only comes with an open inlet | Read labels from `inlet.info()` |
| The app kept showing frozen data after a disconnect | Ring buffers kept old samples | Stale data (> 2 s) is ignored, and a "Headset not streaming" message appears |
| Page blank for seconds on slow Wi-Fi | Google Fonts stylesheet blocked rendering | Font loads in the background |
| Sea animation exploded | Negative time step | Time step clamped to 0–50 ms |

### Connecting the headset (current procedure, two terminals)
```bash
.venv\Scripts\python -m clutch.muselsl_runner stream --ppg --acc --name Muse-3724
```
```bash
.venv\Scripts\python -m clutch --source lsl
```
Then open http://localhost:8000.
- To watch the raw EEG: `.venv\Scripts\muselsl view`.
- For a quick diagnostic: `.venv\Scripts\python tools\check_muse.py --source lsl --eyes-test`.

### Reading signal quality
- The number next to each channel in `muselsl view` is the channel's standard deviation in µV over the window.
- **5–30 µV** is good, **30–50 µV** is borderline, and **above 100 µV** means poor contact.
- When all four channels are bad together with no mains noise, suspect the **reference electrode** in the centre of the forehead.

---

## 6. Status (end of day 1, 24 October)

### What changed since this document was first written
- **Calibration** is now a research-based protocol with two anchors and per-signal Cohen's d. See `docs/methods.md`.
  - **Steps:** vanilla baseline, STAI-6, stressor, STAI-6, recovery.
  - **Stressors:** quick math (default), Stroop, or serial subtraction.
  - **Presets:** Standard (~9 min) and Short (~3 min).
  - **Score:** 0 = your baseline, 100 = your stress-test level.
- **Muse connection** happens from the web app (Connect / Disconnect, live status, automatic reconnection). Start the app with `.venv\Scripts\python -m clutch --source muselsl --name Muse-3724`.
- **Face sensor** (home page, `frontend/face-sensor.js`): MediaPipe runs locally and gives 7 indicators. They join the calibration, and the app shows a second, **headset-free score** (heart + face).
- **Riley's face tracker page** (`face.html`): 14 action units and micro-expressions, with its own logger (`clutch/face.py`). Merged.
- **Dataset recording:** participant code and consent, raw EEG/PPG/ACC and face features. `tools/export_dataset.py` builds the anonymised zip (`docs/dataset.md`). First session published in `data/clutch_dataset/` (P01).

### Status

| Item | Status |
|---|---|
| Muse streams into the app, connected from the web page | ✅ Verified on Muse-3724 (00:55:DA:B6:37:24) |
| Electrode contact | ⚠️ Front (AF7/AF8) good once adjusted; behind the ears (TP9) often fair or bad |
| Real calibration (P01, standard protocol, quick math) | ✅ STAI-6 27 → 63. Heart rate +6 bpm (d = 2.78), blinks 12 → 4 per minute (d = 2.80). Brain not used (median of the 4 channels: no change). |
| Per-sensor EEG analysis of that session | ✅ Right forehead **AF8: alpha down, d = 1.14** (expected direction). TP9/TP10 went the other way, AF7 shows a likely muscle artefact. The median cancelled the signal. |
| Headset-free score (heart + face) on a real person | ✅ Available for P01 (heart + blinks) |
| Dataset export (anonymised) | ✅ Tested and published (1 session) |
| Always-on-top widget, breathing screen, full pressure session | ⚠️ Not visually verified |
| Garmin heart rate | ❌ Not started (watch model needed) |

### Done on the morning of day 2 (25 October, without the headset, tested on the simulator)
- **Per-sensor EEG:**
  - log10 absolute alpha power per channel, each channel compared with itself;
  - the model counts each modality (brain, heart, face) once;
  - the profile shows a head map and a per-sensor table.
  - Chosen after a comparison: relative alpha lost the forehead channels to blinks on simulated data; log absolute alpha kept them. On P01's real data, AF8 keeps d = 0.67. See `docs/methods.md`.
- **Dataset:** per-channel columns (alphalog_*, alpha_*, beta_*, theta_*, q_*).
- **Home page in 3 steps:** Connect → Calibrate → Live, with a ✓ on each step when it is ready.
- **Chart comparing both scores live** (with headset solid, without headset dashed).
- **Presentation mode** for the jury (P key or the "Present" button): two big scores, the sea, the comparison chart.
- **Deterministic tests:** the simulator no longer depends on the PC's speed during tests.

### Done in the afternoon of day 2
- **Why beta was wrong in P01:** beta followed muscle activity (30–44 Hz) on every channel (r = 0.74–0.87). The left forehead sensor picked up muscle tension during the baseline.
- **Automatic muscle rejection:** a window is dropped for one sensor when its 30–44 Hz share is above 0.10. The live view says "relax forehead and jaw", and the profile shows the share of windows rejected per sensor. Effect on P01: AF8 goes from d = 0.70 to 0.84.
- **More robust heart rate:** one missed beat no longer rejects the window, so 2–3× more usable windows on P01.
- **Trained model:** `tools/train_model.py`, gradient-boosted trees on EEG + heart + face. It compares the full set with the headset-free set and validates without leaks.
  - On P01 alone: heart AUC 0.93, headset-free 0.81, full 0.77. The trees overfit 250 seconds of one person.
  - The headset-free model runs live in the app ("Trained model" line) when `models/headset_free.joblib` exists.
  - Retrain after recording the team:
    `.venv\Scripts\python tools\export_dataset.py`, then `.venv\Scripts\python tools\train_model.py --data dataset_export --save models`.
- **Two ways to train:** the label model (rest or stress) and the EEG-teacher model (heart + face imitate the EEG alpha index). With several sessions of one person, they are tested on a new session.
- **Step 4 in the app, "Train your model":**
  - **Build dataset** and **Train model** buttons, which run the two scripts with their output shown live;
  - result tiles;
  - charts: AUC per signal set, learning curve, score over time on unseen data, and progress across trainings (`models/history.json`).
  - The new model runs live as soon as training ends.

### Next steps (day 2)
1. **Check on the real headset** that the per-sensor profile and the head map behave as on the simulator.
2. **Record the team:** about 4 people × 2 calibrations, Standard protocol, camera on, consent ticked, codes P2, P3… Then export and send.
3. **Garmin:** enable heart-rate broadcast and add it as a Bluetooth heart-rate source that replaces the Muse PPG.
4. **Demo prep:** headset score and headset-free score side by side, sea widget, pitch numbers from the recorded sessions.

---

## 7. Next phase: removing the headset

### Goal
Use the Muse only once, as a **teacher**. During a calibration session we record the EEG stress signal alongside signals that need no headset:
- a **webcam** (face);
- a **blood pressure cuff**.

We then learn, **per user**, how the camera and blood pressure signals map onto the EEG alpha/beta stress score. After that, Clutch runs on the webcam alone.

### Signals the webcam can provide
| Signal | How | Notes |
|---|---|---|
| Heart rate and HRV | **Remote photoplethysmography (rPPG):** tiny colour changes of the facial skin at each heartbeat (POS or CHROM algorithms) | Needs steady light and a still face; HRV is harder than heart rate |
| Facial tension | Face landmarks and expression coefficients (e.g. MediaPipe Face Landmarker blendshapes): brow lowering, lip pressing, jaw clench | Works well in real time on a laptop |
| Blink rate | Eye-aspect ratio from landmarks | Blink rate shifts with stress and fatigue |
| Head movement and fidgeting | Landmark motion | Restlessness |
| Breathing rate | Subtle head and shoulder motion | Optional, harder |

### Role of blood pressure
- A standard automatic cuff gives **one reading every 1–2 minutes**, not a continuous stream. The inflation itself is mildly stressful.
- It is best used as **an extra anchor during calibration** (rest versus the arithmetic task), not as a moment-to-moment target.
- None of the lendable devices measures blood pressure, so we need to bring or borrow a cuff. Readings can be typed in by hand, or read over Bluetooth if the cuff supports it.

### Proposed calibration protocol (about 10 minutes, one session per user)
1. Muse, webcam and cuff all running together.
2. Rest (2 min), with a cuff reading.
3. Arithmetic under pressure (3 min), with a cuff reading.
4. Breathing reset (2 min), with a cuff reading.
5. Second round (3 min), with a cuff reading.
6. **Train a small per-user model:** camera features (plus the latest blood pressure) as input, EEG stress score as target. Ridge regression is enough, with no deep learning.
7. **Validate on a held-out segment:** correlation between the camera-predicted and EEG stress, compared with a heart-rate-only baseline.

### Risks to state openly
- **This brings back model training,** which we avoided so far. It stays light (one regression per user), but it needs time and data.
- **We would be learning a proxy of a proxy:** the EEG score itself is not validated. A better long-term target may be performance on the task, or self-reported stress, with EEG as one input.
- **rPPG is fragile:** it suffers from bad lighting, movement and makeup. Heart rate from the camera may end up doing most of the work.
- **Privacy:** process the video locally, store only the extracted numbers and never the images, and ask for consent.

### Milestones
1. **Camera heart rate (rPPG)** working live, cross-checked against the Muse PPG.
2. **Face features** (MediaPipe) streaming into the same engine as a new source.
3. **Combined recording session** (EEG, camera, cuff) saved in the existing session format.
4. **Per-user model and validation report.**
5. **Camera-only mode** in the app, with the sea widget driven by the webcam.

---

## 8. Repository map

```
clutch/processing.py     EEG band powers, contact quality, heart rate, motion
clutch/sources.py        LSL (muselsl), BrainFlow fallback, simulated headset
clutch/engine.py         real-time loop, baseline, stress score, session logging
clutch/server.py         FastAPI + WebSocket, serves the web app
clutch/muselsl_runner.py muselsl launcher with the Bleak fix and longer disconnect delay
frontend/                web app (setup, calibration, tasks, breathing, results, sea widget)
tools/check_muse.py      headset diagnostic and eyes-closed alpha test
tools/fake_muselsl.py    simulated Muse published on LSL, for testing without a headset
tests/test_pipeline.py   end-to-end test on simulated data
analysis/                pilot results across sessions
docs/                    this document
```
