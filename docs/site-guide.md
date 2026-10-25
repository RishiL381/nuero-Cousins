# Clutch: How the App Works

The interface follows Rishi's pixel-art sea: pixel font for titles and numbers, sand and sea colours, with a deep-sea dark mode.

Clutch estimates stress live from three sources: a Muse 2 headband (brain), a heart rate source (a Garmin watch, or the Muse forehead sensor) and a webcam (face). Each user is calibrated against their own calm and stress, so every score is personal. The goal is to keep a good score without the headset: the brain teaches the heart and face models during calibration.

Installation, headset checks and command-line options: `README.md`. Method and references: `docs/methods.md`. Dataset format: `docs/dataset.md`.

---

## 1. Starting the app

```bash
.venv\Scripts\python -m clutch --name Muse-XXXX
```

Then open http://localhost:8000 in Chrome. Chrome is needed for the webcam and the always-on-top widget.

| Option | Use |
|---|---|
| `--source mock` | simulated signals, no hardware (`?quick&debug` in the URL for short sessions and a stress slider) |
| `--source lsl` | read a muselsl stream started elsewhere |
| `--notch 50` | mains frequency in Europe (60 Hz in the US) |

The page has four steps, shown top to bottom. A step turns green when it is done.

---

## 2. Step 1: Connect

**Muse headset**
- The app connects to the Muse on its own and retries if the link drops.
- The head diagram shows each of the 4 sensors in green, orange or red, with the reason. Examples: "not touching skin", "muscle tension: relax forehead and jaw".
- Forehead sensors (AF7, AF8) need the bar flat on the skin. The ear sensors (TP9, TP10) need skin contact, not hair.

**Face (webcam)**
- MediaPipe runs in the browser. No image is sent or saved, only 7 numbers per second: brows lowered, lips pressed, eyes squinted, frown, smile, blink rate and head motion.
- Start the camera before calibrating, so the face is calibrated too.

**Heart rate · watch (optional, recommended)**
- Works with a Garmin watch broadcasting heart rate over Bluetooth (tested with a Forerunner 255) or any Bluetooth heart rate strap.
- On the watch, start **Broadcast Heart Rate**, then click **Connect watch**.
- While it streams, the watch replaces the Muse forehead sensor. That sensor is often wrong: it read about 125 bpm while the watch read about 60 bpm.
- The header shows ⌚ when the heart rate comes from the watch.
- **One computer at a time.** The app connects only on demand and gives up after 3 failed tries. Click **Disconnect** to hand the watch over to another computer.
- Garmin broadcasts heart rate only, not beat-to-beat intervals, so heart rate variability (RMSSD) is only available with a strap.

---

## 3. Step 2: Calibrate and record

**Fields**
- **Participant code:** use the same code for every session of the same person, and no real names.
- **Record for the dataset** and **consent:** both must be ticked. Without consent, the session is never exported.
- **Protocol:** Standard for real data, Short for a demo.
- **Stress test:** quick math (default), colour words (Stroop) or serial subtraction.

**Two ways to run it**

| Button | Duration | Content | Use |
|---|---|---|---|
| **Run study protocol** | ~25 min | calibration, then relaxation, easy math, hard math, hard colour words, with a 1–5 stress rating after each block | building the dataset |
| **Calibration only** | ~9 min | 3 min baseline, STAI-6 questionnaire, 2 min stress test, questionnaire, 2 min recovery | quick personal setup |

- Alternate study orders A and B between sessions.
- **Easy math is a control:** same posture and keyboard as hard math, but little stress. A good model scores it between relaxation and the hard tasks.

**What happens at the end**
- The app computes the calibration report:
  - per signal, the change from baseline to stress test and whether it is used (Cohen's d ≥ 0.5);
  - the brain map: theta, alpha and beta per sensor, in bands set on the person's alpha frequency (shown), with the direction each moved and ✓ when it feeds the score, plus the share of windows removed for muscle tension;
  - the **calibration check**: anchors set on the first half, tested on the second half. "Redo recommended" means the calibration did not separate calm and stress clearly.
- Riley's **personal model** (heart + face imitating the EEG) is trained on this calibration, then retrained with the study blocks.

---

## 4. Step 3: Live

**The gauge (main score, 0 = your calm, 100 = your stress test)** uses the best source available, in this order:

| # | Source | When |
|---|---|---|
| 1 | Trained brain model | headset streaming, a model trained in step 4, and a calibration in this session |
| 2 | Calibration rule with the EEG (brain + heart + face) | headset streaming, no trained model yet |
| 3 | Personal heart + face model | no headset, trained at the end of this calibration |
| 4 | Trained heart + face model | no headset, a model trained in step 4 |
| 5 | Calibration rule on heart + face | otherwise |

- The line under the gauge says which source is used ("Score from: …").
- The lines below the gauge show the other scores side by side: without headset, trained model with and without headset.
- The trained models start 45 s into a calibration. They need that rest to measure the person's baseline and alpha frequency.

**Other elements**
- **Chips:** which signal groups (brain, heart, face) feed the score right now.
- **Chart:** the score with the headset and without it over the last 2 minutes.
- **Sea (Rishi's pixel boat):** waves, rain and darkness follow stress, and day or night follows focus. When the score stays at 70 or more for 20 s, the sea offers a 5-breath break drawn on the water. This is never offered during a calibration or a session.
- **Desktop boat:** a small always-on-top boat that drifts along the bottom of the screen while you work.
  - It goes faster and pitches more when stress rises: about 24 px/s at rest and about 48 px/s at a score of 80.
  - Drag it anywhere: it stays there for 20 s, then drifts on at that height.
  - Close it with the × (shown on hover) or Esc.
  - It is started from the **Desktop boat** button, or with `.venv\Scripts\python -m clutch.desk` (uses `pywebview`).
- **Pop out window:** the same sea in a browser window that stays on top.
- **Present** (or the `P` key): full screen for a jury, with both scores, the sea and the chart.

**Pressure test session (optional, collapsed at the bottom):**
1. warm-up;
2. round 1 of timed math;
3. 6-breaths-per-minute breathing reset;
4. round 2;
5. results comparing both rounds.

---

## 5. Step 4: Train your model

1. **Build dataset:**
   - gathers every session recorded with consent;
   - replaces codes with P01, P02… The mapping stays on this computer;
   - writes `dataset_export/` and a zip to send.
2. **Train model:** runs `tools/train_model.py` and shows its output live. It takes about 1–2 minutes per session. As soon as it ends, the new models run in the app, without a restart.

**What is trained** (gradient-boosted trees, scikit-learn)

| Model | Signals | Target |
|---|---|---|
| With headset | the best of: brain · brain + muscle · everything | rest (0) or stress (1) |
| Label model | heart + face | rest or stress |
| Brain-taught model | heart + face | imitate the brain model's estimate, second by second |

**Features**
- Brain features use bands set on the person's **individual alpha frequency**, measured on the first 45 s of rest (Klimesch, 1999).
- Muscle tension (30–44 Hz) is a separate feature.
- Every feature is taken relative to that person's rest.

**Validation, chosen automatically**
- With 2 or more people: leave-one-person-out.
- With 1 person and 2 or more sessions: leave-one-session-out.
- With 1 session: time blocks only (weak).
- The score is the **AUC** on data the model never saw: 0.5 = chance, 1.0 = perfect.

**Results panel**
- **Tiles:** AUC with headset and without headset, "Brain → body" (how well heart + face reproduce the brain model, 1 = perfectly) and the amount of training data.
- **Which signals separate calm from stress:** AUC per signal set, with the chance line.
- **Learning curve:** AUC when training on 25 / 50 / 75 / 100% of each phase. If it still rises at 100%, more data will help.
- **Score on unseen data:** each session's scores over time. Calm phases are green, stress phases red.
- **Progress across trainings:** one point per training, to follow the models as the dataset grows.

---

## 6. What is recorded

One folder per session in `sessions/` (never pushed; the anonymised export is):

| File | Content |
|---|---|
| `raw_eeg.csv`, `raw_ppg.csv`, `raw_accel.csv` | Muse signals at 256 / 64 / 52 Hz |
| `raw_hr.csv` | watch: heart rate and beat intervals when sent |
| `face.csv` | the 7 face numbers, about once per second |
| `metrics.csv` | every 0.5 s: phase, every score and its source, band powers per sensor, heart rate and its source, contact quality |
| `events.csv` | phase changes, answers, questionnaires, ratings, calibration report |

No image or video is ever recorded.

---

## 7. Keyboard shortcuts

| Key | Action |
|---|---|
| `P` | presentation mode |
| `Esc` | abort a session / close |
| `S` | show or hide the live stress pill |
| `D` | debug panel |

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| A sensor stays red | wet the sensor slightly, move hair aside, press the bar on the forehead |
| "Muscle tension" on a sensor | relax the forehead, unclench the teeth |
| "No Muse found" | Muse on (lights blinking), close, not connected to a phone; click Connect |
| Watch not found | start Broadcast Heart Rate on the watch; Disconnect it on the other computer; click Retry |
| Gauge empty after training | run a calibration: trained models need its first 45 s of rest |
| Two tabs open on the app | fine: only the tab running a protocol controls the phase |

---

## 9. Current state (25 October, evening)

**Dataset:** 2 people, 3 sessions (P01 × 2 with the Muse forehead pulse; P02 × 1 with a Garmin watch), about 30 minutes labelled.

**Trained on one person, tested on the other:**
- brain: AUC 0.87;
- muscle: 0.64;
- one heart + face model for everyone: below chance (0.38). It is therefore not installed (installation requires AUC ≥ 0.6).

**Per person** (the person's own calibration, judged on their study blocks):
- EEG rule: 0.86 (P01) and 0.78 (P02);
- heart + face student taught by it: 0.71 and 0.60.

**What this means for the pitch:**
- The headset calibrates each person reliably.
- Without the headset, a personal model trained on that calibration does better than a model shared by everyone. It is still weaker than the brain.

**Open points:**
- more people and more sessions with the watch;
- the easy-math control still scores high with the brain model, so part of what it detects is "doing a task".
