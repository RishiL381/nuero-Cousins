# Stress model, face tracker and study protocol (Oct 24-25)

What we built on day 1 evening, what we found, and how to use it on day 2.

## 1. Webcam face tracker (in the repo)

`frontend/face.html` (open http://localhost:8000/face.html, or **Face tracker** in the top bar).
- MediaPipe Face Landmarker runs in the browser: the video never leaves it, only numbers go to the server.
- 14 action units from the 52 blendshapes (brow lowerer AU4, lip press AU24, ...), micro-expressions
  (60-500 ms rises above the person's own baseline), blinks, head pose and fidgeting.
- **Record** writes `face_features.csv` (every 0.5 s, with the EEG stress targets) and `face_frames.csv`
  (every frame) into the running session folder. Details in the README.
- Needs WebGL: if Chrome says it is off, turn on Settings -> System -> "Use graphics acceleration when available".

## 2. Public datasets we looked at

We wanted a public dataset recorded **with a Muse** (same 4 sensors) with **heart signal** and **stress labels**.

| Dataset | Muse? | Heart | Labels | Access | Verdict |
|---|---|---|---|---|---|
| **EPIStress** (Zenodo 10.5281/zenodo.16407549) | Muse S | Empatica wrist pulse | easy/hard arithmetic, Stroop, N-back, Sudoku + relaxation video; 1-5 stress rating per task | open, CC BY 4.0 | **used**: 18 people with usable lab data |
| Student exam stress (Mendeley fyj9by2t22) | Muse 2 | none | one questionnaire score per person | open | EEG sanity checks only |
| Kaggle "mental state" / "feeling emotions" (Bird) | Muse | none | relaxed/neutral/concentrating; positive/neutral/negative | open | features only, 2-4 people |
| NeuroSense | Muse 2 | none | valence/arousal ratings | request + DUA | best Muse emotion set, not today |
| EmoWork | Muse S + Polar + E4 | yes | stress/emotion in call-centre role-plays | request + DUA, 185 GB | not today |
| DEAP, DREAMER, AMIGOS | no (research caps / Emotiv) | yes | arousal/valence | applications | no Muse, no fast access |

**EPIStress caveats:** the participants have epilepsy and many take medication (some drugs raise beta or slow
alpha), the heart signal is a clean wrist sensor (ours is the Muse forehead sensor), and their calm state is a
relaxation *video*. We use it to build and test the pipeline and as extra training data, not as the final answer.

## 3. Stress model pipeline (code in `ml/`, setup in `ml/README.md`)

- **Features**, one row per 10 s window (5 s step): EEG relative band power per sensor (delta..gamma),
  log(alpha/beta), frontal alpha asymmetry, heart rate and HRV from the pulse wave, head motion.
  Same filter and bands as `clutch/processing.py`.
- **Testing on people the model never saw.** Folds hold out whole people. Each test person is calibrated like
  the live app (first 1-2 min of their calm and stress phases), then scored on everything else.
- **Models compared:** the app's own two-anchor score, logistic regression, LightGBM, nearest centroid
  (each person's calm average vs stress average), centroid + LDA.

### Results on EPIStress (relaxation video vs hard tasks)

AUC: chance that a stressed window scores above a calm one (0.5 = coin flip).

| Model | Signals | AUC |
|---|---|---|
| **Per-person centroid** | **heart only** | **0.70** |
| Per-person centroid | EEG + heart | 0.64 |
| LightGBM | EEG + heart | 0.62 |
| Logistic regression | EEG + heart | 0.60 |
| App two-anchor (current live score) | alpha/beta + heart | 0.60 |
| Centroid + LDA | EEG + heart | 0.59 |
| LightGBM | all incl. motion | 0.78, **but motion alone gets 0.79** |

### What we learned
1. **Per-person calibration matters most.** The simple centroid (each person's own calm vs stress average) beat
   every model trained on everyone. One shared "stress direction" (LDA) did worse than each person's own.
2. **Split by person, never by random windows.** Same model: random 75/25 split of windows = AUC 0.82,
   75/25 split of people = 0.48 (coin flip). The 0.82 is the model recognising people it already saw.
3. **Motion is a trap.** LightGBM's best score came from "typing vs sitting still". Keep motion out of the score,
   and ask participants to keep still (answer out loud if possible).
4. **Heart rate is the strongest honest signal.** EEG alone: about two thirds of demos work; EEG + heart: most.
5. **It works for most people, not all:** about half of EPIStress people at AUC >= 0.7, and 3-4 of 17 point the
   wrong way. That is why we added the flip check.

### Odds of a good demo (EPIStress, 1 min + 1 min calibration, same stressor as calibration)

| Live signals | One demo, no check | With flip check passed | 3 in a row (with check) |
|---|---|---|---|
| EEG + heart | 82% | 90% | 73% |
| EEG only | 65% | 70% | 34% |

Calibrating on one stressor and demoing on a different one is much weaker (EEG + heart: 36%), so demo the
same task you calibrated on. With 17 people the ranges are wide (90% could be anywhere from about 60% to 98%).

## 4. Flip check (in the repo, `clutch/engine.py`)

After calibration, `split_half_check()` rebuilds the person's calm and stress anchors from the **first half** of
the rest and stress phases (same rules as the calibration: a signal needs Cohen's d >= 0.5), then scores the
**second half**, which those anchors never saw. It passes when stress ranks above rest (AUC >= 0.75) with a
visible gap (>= 0.2 on the 0-1 scale). The report screen shows it first: **Passed** (green) or **Failed, redo
recommended** (red, Redo highlighted). Tests: `tests/test_flip_check.py`.

## 5. Study protocol (in the repo, "Run study protocol" in the Dataset recording card)

EPIStress-style blocks so our recordings match the public data:

1. Calibration (with the flip check)
2. **Relax**: calm sea, eyes open -> phase `study_relax` (calm)
3. **Easy math**: easiest level, 10 s per answer, no buzzer or ticking -> `study_math_easy` (in between)
4. **Hard math** -> `study_math_hard` (stressed)
5. **Hard colour words** (Stroop) -> `study_stroop_hard` (stressed)
6. 1-5 stress rating after every block (`study_rating` events), summary table at the end

Order A: relax, easy math, hard math, colours. Order B: relax, colours, hard math, easy math (alternate between
participants). Blocks are 3 min (or 2 min). Everything is one recorded session; export it with
`tools/export_dataset.py` as usual.

## 6. Day 2 plan

**Recording each person (about 30 min with setup):**
- Eyes open in every block (closing them makes alpha jump on its own).
- Check the contact dots before each block; headband snug, forehead bar on bare skin, hair pushed away.
- Keep still while answering.
- Calibrate with the same stressor you will demo.

**Then one command** (setup: `ml/README.md`):
```bash
python tools/export_dataset.py --no-zip
python ml/run.py dataset_export --cal-seconds 60
```
It prints, per person, the average score of every block (relax should be near 0, hard blocks near 1, easy math
in between), then compares every model with leave-one-person-out. The EPIStress people are added to training
automatically, so the table also shows whether the epilepsy data helps on our own people.

**Schedule:** 9-10 set up and run one short end-to-end test; 10-1 record 4 people; 1-2 run the command and
look at the results; 2-3 demo and slides, no code changes after 2. Pre-calibrate the demo person before the
judges arrive and only go on stage once their flip check passes.

## 7. Open items
- **P01's raw files are missing from the repo:** `.gitignore` used to have `sessions/`, which also matched
  `data/clutch_dataset/sessions/`. It is now `/sessions/`: add that folder so raw EEG and pulse are shared.
- **Garmin live heart rate** (optional): if the watch has "Broadcast Heart Rate" over Bluetooth, a small reader
  could feed it to the app. Otherwise we use the Muse pulse sensor plus EEG.
- Face features are not in the stress model yet: a good "next step" for the pitch.
