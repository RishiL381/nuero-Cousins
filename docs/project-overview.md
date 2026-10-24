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

## 6. Status

| Item | Status |
|---|---|
| Full pipeline on simulated data (tests pass) | ✅ Verified |
| Muse connects over Bluetooth and streams EEG, PPG and ACC into the app | ✅ Verified |
| Good electrode contact on a real head | ❌ **Not achieved yet:** all four channels at 300–600 µV |
| Stress score on real data | ❌ Blocked by contact |
| Sea widget rendering (calm and storm) | ✅ Verified on simulated data |
| Always-on-top widget window, breathing screen, full session on screen | ⚠️ Not visually verified |
| Automatic reconnection after a disconnect | ⚠️ Implemented, not verified |
| Code on GitHub | ✅ https://github.com/RishiL381/nuero-Cousins |

**Immediate priority:** get green contact on one person (or switch to Muse B or C), run a real calibration, and record a first real session.

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
