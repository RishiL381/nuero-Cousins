# Clutch: Calibration Method and References

This document describes how Clutch calibrates its stress score for each user, why each step is there, and which research it rests on. Every reference below was checked against a publisher or index record; see the "Notes on the references" section for the points that remain to be confirmed.

---

## 1. Why calibrate per person

EEG markers of mental stress are inconsistent across studies. Reviews attribute this to differences in protocols, stressors, recording sites and analysis choices, with no standard method in the field (Katmah et al., 2021; Giannakakis et al., 2019).

Rather than assuming that one marker works for everyone, Clutch:
1. records each user in a known **calm** state and a known **stressed** state;
2. measures whether each signal actually responds for that user;
3. builds the score only from the signals that do.

---

## 2. Protocol

Two presets are available: **Standard** (recommended) and **Short** (for live demos).

| Step | Standard | Short | What happens | Basis |
|---|---|---|---|---|
| 1. Baseline | 3 min | 1 min | Low-demand "vanilla" task: watch a circle change colour and press Space when it turns blue. Eyes stay open. | Jennings et al. (1992); Barry et al. (2007) |
| 2. Self-report | ~30 s | ~30 s | Six-item State-Trait Anxiety Inventory (STAI-6) | Marteau & Bekker (1992) |
| 3. Stressor | 2 min | 1 min 15 | One of three tasks (below), always under an adaptive answer time limit and with a bar comparing the user to a fictitious, always-better "average participant". | Dedovic et al. (2005); Dickerson & Kemeny (2004); Stroop (1935); Kirschbaum et al. (1993) |
| 4. Self-report | ~30 s | ~30 s | STAI-6 again, as a manipulation check | Marteau & Bekker (1992); Allen et al. (2014) |
| 5. Recovery | 2 min | — | Quiet rest, eyes open on a fixation dot | Allen et al. (2014); Birkett (2011) |
| 6. Profile | — | — | Per-signal effect sizes, decision, and debrief | Cohen (1988) |

### Rationale for each step

**Baseline as a "vanilla" task, not passive rest.**
- Jennings et al. (1992) compared 20-minute resting baselines with a minimally demanding colour-detection task. The vanilla baseline was as stable as, or more stable than, rest.
- Their abstract also notes that long resting periods do not guarantee stability.
- The task keeps the user awake and attentive, and it keeps the eyes open. This matters for EEG: eyes-closed and eyes-open resting states differ in both power and topography, so they are not interchangeable baselines (Barry et al., 2007).
- Hits on the blue circle are recorded as an attention check.

**Choice of stressor.** Three tasks are offered. All share the same pressure elements (see below).

| Task | What the user does | Basis | Intensity |
|---|---|---|---|
| **Quick math** (default) | A new problem each time, with a single-digit answer pressed on one key. Five difficulty levels, going from `7 + 2` to `6 × 8 − 5 − 34`. Up a level after 3 correct in a row, down a level after an error. | Montreal Imaging Stress Task (MIST): computerised arithmetic under time pressure with negative feedback (Dedovic et al., 2005) | moderate |
| **Colour words** | Press the key for the ink colour of a colour word. 75% of items are incongruent (e.g. "RED" printed in blue). | Stroop interference (Stroop, 1935); a paced computer Stroop task raises heart rate and self-reported anxiety (Renaud & Blondin, 1997) | mild to moderate |
| **Serial subtraction** | Count down from 1022 by 13, typed. An error or timeout restarts from 1022. | Arithmetic part of the Trier Social Stress Test (TSST; Kirschbaum et al., 1993), which raised ACTH, cortisol and heart rate across six studies | strongest |

- **Why quick math is the default:** in testing, the serial subtraction's restart rule made users repeat the same sequence for minutes, which felt punishing.
- The MIST-style task keeps time pressure, adaptive difficulty and social comparison, but presents a new item each time.
- **Trade-off:** a milder stressor produces a smaller physiological response, so effect sizes may be smaller. The self-report check (STAI-6) shows whether the chosen task actually stressed the user.
**Time pressure and social comparison.**
- The Montreal Imaging Stress Task (MIST; Dedovic et al., 2005) adapted the TSST to a computer. It combines mental arithmetic under time limits with social-evaluative threat.
- Clutch borrows two of its elements, for all three tasks:
  - an answer time limit that tightens as the user succeeds;
  - a comparison bar showing a fictitious average participant who always does better.
- A meta-analysis of 208 laboratory stressors found that tasks that are **uncontrollable** and carry **social-evaluative threat** produce the largest cortisol and ACTH responses and the longest recovery (Dickerson & Kemeny, 2004). This is why the stressor includes both elements rather than arithmetic alone.
- The comparison bar is a mild deception, so the profile screen debriefs the user about it.

**Self-report (manipulation check).**
- Physiological signals alone cannot tell us whether the stressor worked, and stress research recommends multiple readouts (Allen et al., 2014).
- The STAI-6 (Marteau & Bekker, 1992) is a validated six-item short form of the STAI state scale, correlating above .90 with the full form.
- Clutch scores it as follows:
  - each item is scored 1–4, with the calm, relaxed and content items reversed;
  - the sum is multiplied by 20/6 to match the usual 20–80 range.
- An increase of at least 5 points is flagged as "the stressor worked". **This threshold is our heuristic, not a published cut-off.**

**Recovery.**
- TSST protocols include recovery periods after the stressor (Birkett, 2011).
- Clutch reports how far each signal returned toward baseline: 0% = still at the stress level, 100% = back at baseline.

---

## 3. Signals and analysis

### EEG
- **Device:** Muse 2. Its validity for research-grade event-related potentials has been shown by Krigolson et al. (2017).
- **Channels:** TP9, AF7, AF8, TP10.
- **Preprocessing:** band-pass 1–45 Hz and a 60 Hz notch, on 4-second windows updated every 0.5 s.
- **Band power:** Welch's averaged-periodogram method (Welch, 1967), integrated over theta (4–8 Hz), alpha (8–12 Hz) and beta (13–30 Hz).
- **Feature: alpha power per channel.** For each of the four channels, `log10` of the absolute alpha power (µV²). Each channel is compared **with itself** between baseline and stressor, and only in windows where that channel has acceptable contact.
- **Why alpha, and not alpha/beta:**
  - in a meta-analysis of spectral EEG studies of psychosocial stress, alpha power decreased consistently (Hedges' g = 0.6, p = 0.001), while beta showed no significant pooled effect (g = −0.31, p = 0.29) and frontal alpha asymmetry no effect (Vanhollebeke et al., 2022);
  - alpha oscillations track cognitive state and performance (Klimesch, 1999); frontal alpha decreases under mental arithmetic stress (Al-Shargie et al., 2016).
- **Why per channel:** in our first real session (P01), the median over the four channels showed no change because channels moved in opposite directions. The right forehead channel (AF8) showed a large alpha decrease (d = 1.14), the channels behind the ears an increase. Contact also differed between phases, so a median mixed different channels across conditions.
- **Why absolute log power rather than relative power:** blinks put most of their energy below 4 Hz but spill into theta. On simulated data with blinks on the forehead channels, relative alpha lost those channels (d ≈ 0), while log absolute alpha kept them (d ≈ 1.8–2.3). Absolute power is only compared within one channel and one session, so electrode impedance differences between people do not matter.
- **Bands:**
  - The live calibration rule uses fixed bands.
  - The trained models use bands set on the person's **individual alpha frequency** (IAF), as Klimesch (1999) recommends: theta IAF−6 to IAF−4, alpha IAF−4 to IAF+2, low beta IAF+2 to 20 Hz.
  - **How the IAF is measured:** on the first 45 s of rest, take the highest local peak between 7 and 13 Hz, above the 1/f background by at least 0.1 log units. Peaks at the edge of the range are ignored. The IAF is the median over channels, and 10 Hz when no channel shows a peak.
  - **P01:** 11.0 Hz in session 1, 12.0 Hz in session 2. The fixed 8–12 Hz band missed the upper half of this alpha.
- **Muscle rejection:** forehead and jaw muscle activity (EMG) adds broadband power from about 20 Hz upwards. It is largest at peripheral sites such as the Muse forehead and ear channels, and is detectable above 12 Hz even with weak contractions (Goncharova et al., 2003). A channel window whose relative 30–44 Hz power exceeds **0.10** is treated as muscle-contaminated and left out, for that channel only, as with bad contact or head motion. The interface then asks the user to relax forehead and jaw, and the calibration report gives the share of windows rejected per sensor.
  - **Why:** in P01, beta correlated with 30–44 Hz power on every channel (r = 0.74–0.87), so the beta changes reflected muscle tension rather than the brain. The left forehead channel (AF7) had 30–44 Hz power around 0.13 during the baseline, against about 0.01 on the clean ear channels.
  - **Effect on P01:** with the rule, AF8's alpha response grows from d = 0.70 to 0.84, and AF7's spurious opposite response shrinks from d = −0.59 to −0.28.
  - **Limitation:** the 0.10 threshold was set on one session. EMG spectra vary between people (Goncharova et al., 2003), so it should be checked on more participants.
### Heart rate
- **Watch (preferred):** a Garmin watch broadcasting heart rate over Bluetooth, or any Bluetooth heart rate strap, read with the standard Heart Rate Service. When it streams, it replaces the Muse forehead PPG. When the device sends beat-to-beat intervals, the app also computes **RMSSD** over 30 s, a standard short-term heart rate variability measure (Task Force, 1996), which is expected to drop under stress. It is used as the signal `hrv`, in the heart group.
- **Source:** the Muse optical (PPG) sensor, 10-second windows, beat detection cross-checked against the spectral peak.
- **Quality:** the share of beat-to-beat intervals within 15% of their median. A single missed beat no longer fails the whole window. On P01, this raised the share of usable windows from 16% to 40% at rest and from 5% to 18% under stress, with consistent values (median 54 bpm at rest and 59 bpm under stress, matching the spectral estimate). The forehead PPG remains noisy, and a watch should do better.
- **Why:** heart rate rises reliably in the TSST (Kirschbaum et al., 1993), and cardiac measures are standard stress readouts (Allen et al., 2014; Giannakakis et al., 2019).
- **Methodological guidance:** heart rate variability (HRV) is a stronger marker but needs cleaner recordings; we follow the recommendations of Laborde et al. (2017) for any future HRV work.

### Decision rule
For each signal (each EEG channel, heart rate, each face indicator) we compute **Cohen's d** between the stressor and baseline samples, signed so that a positive value means the expected direction: alpha down, heart rate up. For face indicators with no established direction (smile, blink rate), the direction is learned from the calibration.

| d | Label (Cohen, 1988) | Used? |
|---|---|---|
| ≥ 0.8 | large | yes |
| ≥ 0.5 | medium | yes |
| ≥ 0.2 | small | no |
| between −0.2 and 0.2 | negligible | no |
| ≤ −0.2 | opposite direction | no |

- **Signals that are used** are placed on their own scale between the baseline and stressor medians: 0 = baseline, 100 = stressor level.
- **Each modality counts once.** Inside a modality (brain, heart, face), used signals are averaged weighted by d. The modality then weighs as much as its strongest signal. So four EEG channels or seven face indicators cannot outvote the heart.
- **Two scores:** the full score uses brain, heart and face; the **headset-free score** uses heart and face only.
- **If no signal reaches d ≥ 0.5,** the score falls back to distance from baseline, where 3 baseline standard deviations = 100. This is shown as **low confidence**.

**Caveat:** samples come from overlapping windows, so they are not independent. d is used here as a descriptive measure of response size, not as a significance test.

### Intervention used in sessions
The breathing reset paces breathing at 6 breaths per minute. The mechanisms of slow-paced breathing and HRV biofeedback are reviewed by Lehrer & Gevirtz (2014).

### Learned model (towards no headset)
`tools/train_model.py` trains gradient-boosted decision trees (scikit-learn `HistGradientBoosting`, which handles missing values natively) in two ways, and compares them on data the model never saw.
- **Features** (`clutch/learned.py`, shared with the live app), every second:
  - **brain:** log theta, alpha and low-beta power per channel, in IAF bands. Muscle-free windows only.
  - **muscle:** log 30–44 Hz power per channel. Kept separate so each model shows which one it relies on.
  - **heart:** heart rate, plus RMSSD when the watch sends beat intervals.
  - **face:** the seven face indicators.
- **Personal reference:** each feature is taken relative to the median of the first 45 s of the session's first rest block. Those seconds are then excluded from training and testing. The features are averaged over the last 10 s.
- **A. Label model:** predicts what the person was doing. Rest = 0 (`calib_rest`, `study_relax`), stressor = 1 (`calib_stress`, `study_math_hard`, `study_stroop_hard`). It uses labelled seconds only. Feature sets: full, headset-free (heart + face), and each modality alone.
- **B. Brain-taught model:** heart + face learn to imitate the brain.
  - An EEG model (A, on EEG features only) gives its stress estimate for every second, labelled or not.
  - The estimate is cross-fitted: each second is scored by an EEG model that did not train on it.
  - A heart + face regressor learns to reproduce that estimate. Its output is scaled so that the median at rest = 0 and under stress = 1.
  - A first version used a fixed rule (alpha drop on the channels that responded). It found no responding channel in session 2, so it was replaced.
- **Screens left out:** screens where the person reads or answers questions (report, questionnaires, ratings) are not used. Alpha drops with any visual attention, not only with stress.
- **Common judge:** every model is judged by its AUC between the labelled rest and stress seconds of unseen data. The script also prints the mean score per phase. Easy math (`study_math_easy`, unlabelled) should fall between rest and hard tasks: it has the same posture and keyboard as hard math, so it tests whether a model learned stress or just "typing".
- **Validation**, chosen automatically:
  - with two or more participants: leave-one-participant-out;
  - with one participant and two or more sessions: leave-one-session-out, for a personal model;
  - with one session: 5 contiguous time blocks per phase, with 15 s removed around each test block. This is weak.
  - Random splits are not used: neighbouring seconds share overlapping windows and would inflate the score.
- **Live use:** `--save models` (or "Train model" in the app) saves:
  - `with_headset.joblib`: the best of the brain, brain + muscle and full models. It **drives the live gauge while the Muse streams**, and the app shows which score the gauge uses;
  - `headset_free.joblib`: the better of the label and brain-taught heart + face models.
  - Both start scoring 45 s into a calibration, next to the rule-based scores, with ~5 s output smoothing.

**Results, P01, 2 sessions (25 October), IAF bands, leave-one-session-out, AUC on the unseen session:**

| Model | Signals | Session 1 | Session 2 | Pooled |
|---|---|---|---|---|
| A label | **brain** | **0.95** | **0.88** | **0.87** |
| A label | brain + muscle | 0.93 | 0.88 | 0.85 |
| A label | everything | 0.99 | 0.72 | 0.77 |
| A label | muscle | 0.67 | 0.40 | 0.40 |
| A label | heart + face | 1.00 | 0.67 | 0.72 |
| A label | face | 1.00 | 0.65 | 0.72 |
| A label | heart (Muse PPG) | 0.73 | 0.62 | 0.65 |
| B brain-taught | heart + face | 0.85 | 0.64 | 0.65 (imitation r = 0.03) |

- **The brain transfers best to a new session.** Heart + face scores 1.00 on session 1 when trained on session 2, but only 0.67 the other way. Their pattern is not stable between sessions.
- **Muscle does not explain it.** 30–44 Hz power alone is at chance, and adding it does not help the brain features. An earlier reading of this data said the stress signal was mostly muscle: that was wrong.
- **What the brain model uses** (permutation importance): low beta (IAF+2 to 20 Hz) at TP10 and AF7, then alpha at AF8.
  - AF8 alpha drops under stress in both sessions and does not drift with time. This is the marker expected from the literature (Vanhollebeke et al., 2022).
  - Low beta in muscle-free windows can still hold weak muscle activity below the 0.10 threshold. That cannot be ruled out with 4 dry channels.
- **Heart rate in these two sessions came from the Muse forehead PPG.** It was often wrong: about 125 bpm while a Garmin read about 60 bpm at the same time. Heart results should be read with that in mind. New sessions use the watch.
- **Score per phase on session 2** (brain model, unseen): rest 0.33, stress test 0.95, recovery 0.35, relaxation 0.23, easy math 0.92, hard math 0.96, hard colour words 0.61.
  - Relaxation is correctly calm.
  - Easy math scores almost as high as hard math, so part of what the model detects is "doing a task" rather than stress.
- **Heart + face cannot yet imitate the brain** (r = 0.03 on the unseen session). Removing the headset is not supported by two sessions. More sessions are needed to tell whether this is a data problem or a real limit.
- **Session 2 data issue:**
  - another browser tab reset the phase to idle 22 s into the 3-minute baseline. The 566–723 s range was relabelled `calib_rest` (originals kept as `*.orig`).
  - The bug is fixed: only the tab running a protocol restores its phase after a reconnect.
  - The live calibration of that session used only those 22 s, which explains its wrong live score (93 during relaxation).
- **Self-report, session 2:** STAI-6 rose from 37 to 63. Block ratings were relax 2, easy math 3, hard math 5, hard colour words 5, so the stress induction worked.

---|---|---|
| A label | full | 0.77 |
| A label | heart + face | 0.81 |
| A label | EEG | 0.75 |
| A label | face | 0.78 |
| A label | heart | 0.93 |
| B teacher | heart + face | 0.66 (correlation with the EEG index on held-out data: 0.46) |
| EEG index itself (teacher) | AF8 | 0.77 (channel chosen on the same data, optimistic) |

- **A** relies mostly on head motion and smile. These may reflect the task (looking at the keyboard) rather than stress.
- **B** relies mostly on heart rate and smile, the more physiological signals. It can only be as good as its teacher, which here is one channel with AUC 0.77.
- **Heart rate alone separates best.**
- **Validity:** with one session, none of this is conclusive. It needs several sessions (personal model) or several people.

---

## 4. Limitations
- This is a single-session, single-user calibration on a 4-channel dry-electrode consumer headset. Contact quality strongly affects EEG validity.
- The calibration stressor (2 min) is much shorter than the full TSST (5 min of arithmetic plus a speech) and has no live audience, so the response is expected to be milder, especially with the quick math and colour word tasks.
- Baseline durations (1–3 min) are shorter than the 10–20 min studied by Jennings et al. (1992). This is a practical trade-off.
- The STAI is copyrighted (Mind Garden). Its use here is for a research prototype; other uses need a license.
- Clutch is a wellness and performance prototype, not a medical or diagnostic device.

---

## 5. References

- Allen, A. P., Kennedy, P. J., Cryan, J. F., Dinan, T. G., & Clarke, G. (2014). Biological and psychological markers of stress in humans: Focus on the Trier Social Stress Test. *Neuroscience & Biobehavioral Reviews, 38*, 94–124. https://research.ucc.ie/en/publications/biological-and-psychological-markers-of-stress-in-humans-focus-on/
- Al-Shargie, F., et al. (2016). Mental stress assessment using simultaneous measurement of EEG and fNIRS. *Biomedical Optics Express, 7*(10), 3882–3898. https://pmc.ncbi.nlm.nih.gov/articles/5102531
- Barry, R. J., Clarke, A. R., Johnstone, S. J., et al. (2007). EEG differences between eyes-closed and eyes-open resting conditions. *Clinical Neurophysiology, 118*(12), 2765–2773. https://ro.uow.edu.au/hbspapers/1640
- Birkett, M. A. (2011). The Trier Social Stress Test protocol for inducing psychological stress. *Journal of Visualized Experiments, 56*, e3238. https://doi.org/10.3791/3238
- Renaud, P., & Blondin, J.-P. (1997). The stress of Stroop performance: Physiological and emotional responses to color-word interference, task pacing, and pacing speed. *International Journal of Psychophysiology, 27*(2), 87–97. https://doi.org/10.1016/S0167-8760(97)00049-4
- Stroop, J. R. (1935). Studies of interference in serial verbal reactions. *Journal of Experimental Psychology, 18*, 643–662. https://doi.org/10.1037/h0054651
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Dedovic, K., et al. (2005). The Montreal Imaging Stress Task: Using functional imaging to investigate the effects of perceiving and processing psychosocial stress in the human brain. *Journal of Psychiatry & Neuroscience, 30*(5), 319–325. https://www.jpn.ca/content/30/5/319
- Dickerson, S. S., & Kemeny, M. E. (2004). Acute stressors and cortisol responses: A theoretical integration and synthesis of laboratory research. *Psychological Bulletin, 130*(3), 355–391. https://doi.org/10.1037/0033-2909.130.3.355
- Giannakakis, G., Grigoriadis, D., Giannakaki, K., Simantiraki, O., Roniotis, A., & Tsiknakis, M. (2019). Review on psychological stress detection using biosignals. *IEEE Transactions on Affective Computing* (published 2022, 13(1), 440–460). https://doi.org/10.1109/TAFFC.2019.2927337
- Goncharova, I. I., McFarland, D. J., Vaughan, T. M., & Wolpaw, J. R. (2003). EMG contamination of EEG: Spectral and topographical characteristics. *Clinical Neurophysiology, 114*(9), 1580–1593. https://doi.org/10.1016/S1388-2457(03)00093-2
- Jennings, J. R., Kamarck, T., Stewart, C., Eddy, M., & Johnson, P. (1992). Alternate cardiovascular baseline assessment techniques: Vanilla or resting baseline. *Psychophysiology, 29*(6), 742–750. https://doi.org/10.1111/j.1469-8986.1992.tb02052.x
- Katmah, R., Al-Shargie, F., Tariq, U., Babiloni, F., Al-Mughairbi, F., & Al-Nashash, H. (2021). A review on mental stress assessment methods using EEG signals. *Sensors, 21*(15). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8347831/
- Kirschbaum, C., Pirke, K. M., & Hellhammer, D. H. (1993). The 'Trier Social Stress Test': A tool for investigating psychobiological stress responses in a laboratory setting. *Neuropsychobiology, 28*(1–2), 76–81.
- Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: A review and analysis. *Brain Research Reviews, 29*(2–3), 169–195. https://pubmed.ncbi.nlm.nih.gov/10209231/
- Krigolson, O. E., Williams, C. C., Norton, A., Hassall, C. D., & Colino, F. L. (2017). Choosing MUSE: Validation of a low-cost, portable EEG system for ERP research. *Frontiers in Neuroscience, 11*, 109. https://doi.org/10.3389/fnins.2017.00109
- Laborde, S., Mosley, E., & Thayer, J. F. (2017). Heart rate variability and cardiac vagal tone in psychophysiological research: Recommendations for experiment planning, data analysis, and data reporting. *Frontiers in Psychology, 8*, 213. https://doi.org/10.3389/fpsyg.2017.00213
- Lehrer, P. M., & Gevirtz, R. (2014). Heart rate variability biofeedback: How and why does it work? *Frontiers in Psychology, 5*, 756. https://doi.org/10.3389/fpsyg.2014.00756
- Marteau, T. M., & Bekker, H. (1992). The development of a six-item short-form of the state scale of the Spielberger State-Trait Anxiety Inventory (STAI). *British Journal of Clinical Psychology, 31*(3), 301–306. https://doi.org/10.1111/j.2044-8260.1992.tb00997.x
- Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. *Circulation, 93*(5), 1043–1065. https://doi.org/10.1161/01.CIR.93.5.1043
- Vanhollebeke, G., De Smet, S., De Raedt, R., Baeken, C., van Mierlo, P., & Vanderhasselt, M.-A. (2022). The neural correlates of psychosocial stress: A systematic review and meta-analysis of spectral analysis EEG studies. *Neurobiology of Stress, 18*, 100452. https://doi.org/10.1016/j.ynstr.2022.100452
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics, 15*(2), 70–73. https://doi.org/10.1109/TAU.1967.1161901

### Notes on the references
- **Kirschbaum et al. (1993):** citation details confirmed. The starting number 1022 and step of 13 are **not** in the original paper. They are a widely used variant, and some studies use other numbers.
- **Dedovic et al. (2005):** the article is confirmed. The five difficulty levels and adaptive time-outs follow public descriptions of MIST implementations (e.g. the Millisecond Inquisit script, itself a best-guess implementation), not a rule quoted from the paper. Our parameters are our own: limit ×0.92 after a success and ×1.12 after a failure, 2–10 s for math and 0.9–3 s for colour words; ×0.9 and ×1.1 within 2.5–10 s for serial subtraction.
- **Al-Shargie et al. (2016):** stress was induced with mental arithmetic under time pressure and negative feedback. It is not confirmed that this was the MIST itself.
- **Barry et al. (2007), Al-Shargie et al. (2016), Dedovic et al. (2005):** full author lists were not re-checked, hence "et al.". For Barry et al., the DOI also differs between indexes. Check the publisher records before formal citation.
- **Cohen (1988):** standard textbook, cited for the conventional small, medium and large thresholds. Not re-checked online.
- **Marteau & Bekker (1992):** an erratum was published in 2020 (*Br J Clin Psychol* 59(2):276).
