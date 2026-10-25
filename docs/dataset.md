# Clutch Stress Dataset

Recordings of people at rest and under a lab stressor, collected at the same time from:
- a **Muse 2 EEG headband** (brain, heart rate and head motion);
- a **webcam** (facial indicators computed by MediaPipe, never images).

The labels come from the calibration protocol, not from manual annotation. The protocol and its references are described in `methods.md`.

## Purpose
Train and evaluate models that estimate stress, especially **from heart rate and face alone**, so that the EEG headset is not needed in the end. The heart rate would then come from a watch.

## Protocol (per session)
1. **Baseline:** low-demand task, eyes open (3 min standard, 1 min short) → **label 0**.
2. **STAI-6 questionnaire.**
3. **Stressor:** quick mental arithmetic, Stroop colour words, or serial subtraction, under time pressure with social comparison (2 min standard, 1 min 15 short) → **label 1**.
4. **STAI-6 questionnaire.**
5. **Recovery:** quiet rest (2 min, standard protocol only). Unlabelled.

Some sessions continue with the pressure test (warm-up, rounds, breathing reset). Those phases are unlabelled, but the `phase` column lets you label them yourself.

## Files
```
README.md          this file
methods.md         protocol, rationale and references
sessions.csv       one row per session: participant, protocol, seconds of rest/stress, STAI before/after
features.csv       one row every 0.5 s, all sessions together (ready for a first model)
sessions/<id>/     raw data per session
    raw_eeg.csv    EEG, 256 Hz: lsl_ts, t, TP9, AF7, AF8, TP10 (microvolts)
    raw_ppg.csv    optical heart sensor, 64 Hz: lsl_ts, t, ppg1, ppg2, ppg3
    raw_accel.csv  accelerometer, 52 Hz: lsl_ts, t, x, y, z (g)
    face.csv       face indicators, ~1 Hz (columns below)
    events.csv     phase changes, task answers, questionnaires, calibration report (JSON in `data`)
    labels.csv     t_start, t_end, phase, label
    meta.json      anonymised session information
```

**Time base:** `t` is seconds on the session clock, shared by every file. Use it to align raw signals with `labels.csv`.

## features.csv columns

**Identifiers and label**

| Column | Meaning |
|---|---|
| participant, session | anonymised IDs (P01, P01_S01) |
| t, phase | session clock and protocol phase |
| label | **0 = rest, 1 = stress**, empty = other phase |

**EEG (Muse)**

| Column | Meaning |
|---|---|
| alpha, beta, theta | relative band power: median of channels with acceptable contact, 4 s window |
| ratio_log | log10(alpha / beta) |
| usable_channels | number of channels with acceptable contact |
| contact | contact quality per channel, in the order TP9 AF7 AF8 TP10: g = good, f = fair, b = bad |

**EEG per channel** (`<ch>` = TP9, AF7, AF8, TP10; 4 s window)

| Column | Meaning |
|---|---|
| alphalog_\<ch\> | **log10 absolute alpha power (µV²): the brain feature the app uses.** Robust to blinks. |
| alpha_\<ch\>, beta_\<ch\>, theta_\<ch\> | relative band power of that channel (share of 1–45 Hz) |
| alphashare_\<ch\> | alpha share of 4–30 Hz power |
| q_\<ch\> | contact quality of that channel: good, fair or bad. Ignore the channel's values when it is bad. |

Compare each channel with itself: channels can react in opposite directions, so averaging them can hide the effect. Sessions recorded before 25 October lack these columns; compute them from `raw_eeg.csv` if needed.

**Heart and motion**

| Column | Meaning |
|---|---|
| hr_bpm, hr_quality | heart rate from the Muse PPG over 10 s, with quality from 0 to 1 (use ≥ 0.5) |
| motion | head motion from the accelerometer |

**Face (webcam, 1 s means)**

| Column | Meaning |
|---|---|
| face_brow | brows lowered, 0–1 |
| face_press | lips pressed, 0–1 |
| face_squint | eyes squinted, 0–1 |
| face_frown | mouth corners down, 0–1 |
| face_smile | smile, 0–1 |
| blink_rate | blinks per minute |
| head_motion | inter-eye distances per second |

**Reference outputs**

| Column | Meaning |
|---|---|
| stress, stress_body | the app's own scores (with EEG / heart + face). These come from the app, not ground truth: **do not train on them**. |

## Recommended use
- **Target:** `label`, using only rows where it is not empty.
- **Validation:** keep participants apart (**leave-one-participant-out**). Windows from the same session are strongly correlated, so a random split gives inflated scores.
- **Per-person normalisation:** subtract each session's baseline (label 0) mean. Physiology varies a lot between people.
- **Headset-free model:** use only `hr_bpm` (quality ≥ 0.5) and the face columns. Watch heart rate (Garmin) replaces `hr_bpm` later.
- **Manipulation check:** in `sessions.csv`, a session where STAI-6 did not rise (`stai_after - stai_before < 5`) may have a weak stress label.

## Caveats
- Small number of participants and short sessions: treat any result as a pilot.
- The labels say what the person was doing (rest or stressor), not how stressed they felt. STAI-6 gives the self-report.
- EEG quality depends on contact: filter on `usable_channels` and `contact`.
- Missing values are empty cells. Face columns are empty when the camera was off or no face was found.

## Consent and privacy
- Every exported session was recorded with the participant's explicit consent, ticked in the app.
- Participant codes are replaced by P01, P02…; the mapping stays on the recording computer.
- **No images or video are recorded**, only numeric face indicators.
- Do not redistribute beyond the purpose agreed with the participants.
