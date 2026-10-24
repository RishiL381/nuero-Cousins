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
| 3. Stressor | 3 min | 1.5 min | Serial subtraction from 1022 in steps of 13. An error or a timeout restarts from 1022. Answer time limit tightens after each success. A bar compares the user with a fictitious, always-better "average participant". | Kirschbaum et al. (1993); Dedovic et al. (2005); Dickerson & Kemeny (2004) |
| 4. Self-report | ~30 s | ~30 s | STAI-6 again, as a manipulation check | Marteau & Bekker (1992); Allen et al. (2014) |
| 5. Recovery | 2 min | — | Quiet rest, eyes open on a fixation dot | Allen et al. (2014); Birkett (2011) |
| 6. Profile | — | — | Per-signal effect sizes, decision, and debrief | Cohen (1988) |

### Rationale for each step

**Baseline as a "vanilla" task, not passive rest.**
- Jennings et al. (1992) compared 20-minute resting baselines with a minimally demanding colour-detection task. The vanilla baseline was as stable as, or more stable than, rest.
- Their abstract also notes that long resting periods do not guarantee stability.
- The task keeps the user awake and attentive, and it keeps the eyes open. This matters for EEG: eyes-closed and eyes-open resting states differ in both power and topography, so they are not interchangeable baselines (Barry et al., 2007).
- Hits on the blue circle are recorded as an attention check.

**Serial subtraction as the stressor.**
- Serial subtraction is the arithmetic component of the Trier Social Stress Test (TSST; Kirschbaum et al., 1993). The TSST protocol tells participants that each error means starting again from the initial number.
- Across the six studies reported with the original test, the TSST raised ACTH, cortisol, GH, prolactin and heart rate.
- Clutch uses the commonly used variant that counts down from 1022 by 13.

**Time pressure and social comparison.**
- The Montreal Imaging Stress Task (MIST; Dedovic et al., 2005) adapted the TSST to a computer. It combines mental arithmetic under time limits with social-evaluative threat.
- Clutch borrows two of its elements:
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
- **Band power:** Welch's averaged-periodogram method (Welch, 1967), integrated over alpha (8–12 Hz) and beta (13–30 Hz).
- **Feature:** `log10(alpha / beta)`, the median over channels with acceptable contact.
- **Why alpha and beta:**
  - alpha oscillations track cognitive state and performance (Klimesch, 1999);
  - frontal alpha decreases under mental arithmetic stress (Al-Shargie et al., 2016);
  - alpha and beta power are among the features most used in EEG stress studies (Katmah et al., 2021).
- **Limitation:** Klimesch (1999) recommends adjusting bands to each person's individual alpha frequency. Clutch uses fixed bands for now.

### Heart rate
- **Source:** the Muse optical (PPG) sensor, 10-second windows, beat detection cross-checked against the spectral peak.
- **Why:** heart rate rises reliably in the TSST (Kirschbaum et al., 1993), and cardiac measures are standard stress readouts (Allen et al., 2014; Giannakakis et al., 2019).
- **Methodological guidance:** heart rate variability (HRV) is a stronger marker but needs cleaner recordings; we follow the recommendations of Laborde et al. (2017) for any future HRV work.

### Decision rule
For each signal we compute **Cohen's d** between the stressor and baseline samples, signed so that a positive value means the expected direction (EEG ratio down, heart rate up).

| d | Label (Cohen, 1988) | Used? |
|---|---|---|
| ≥ 0.8 | large | yes |
| ≥ 0.5 | medium | yes |
| ≥ 0.2 | small | no |
| between −0.2 and 0.2 | negligible | no |
| ≤ −0.2 | opposite direction | no |

- **Signals that are used** are weighted by d. Each one is placed on its own scale between the baseline and stressor medians: 0 = baseline, 100 = stressor level.
- **If no signal reaches d ≥ 0.5,** the score falls back to distance from baseline, where 3 baseline standard deviations = 100. This is shown as **low confidence**.

**Caveat:** samples come from overlapping windows, so they are not independent. d is used here as a descriptive measure of response size, not as a significance test.

### Intervention used in sessions
The breathing reset paces breathing at 6 breaths per minute. The mechanisms of slow-paced breathing and HRV biofeedback are reviewed by Lehrer & Gevirtz (2014).

---

## 4. Limitations
- This is a single-session, single-user calibration on a 4-channel dry-electrode consumer headset. Contact quality strongly affects EEG validity.
- The calibration stressor is shorter than the full TSST (5 min of arithmetic plus a speech) and has no live audience, so the response is expected to be milder.
- Baseline durations (1–3 min) are shorter than the 10–20 min studied by Jennings et al. (1992). This is a practical trade-off.
- The STAI is copyrighted (Mind Garden). Its use here is for a research prototype; other uses need a license.
- Clutch is a wellness and performance prototype, not a medical or diagnostic device.

---

## 5. References

- Allen, A. P., Kennedy, P. J., Cryan, J. F., Dinan, T. G., & Clarke, G. (2014). Biological and psychological markers of stress in humans: Focus on the Trier Social Stress Test. *Neuroscience & Biobehavioral Reviews, 38*, 94–124. https://research.ucc.ie/en/publications/biological-and-psychological-markers-of-stress-in-humans-focus-on/
- Al-Shargie, F., et al. (2016). Mental stress assessment using simultaneous measurement of EEG and fNIRS. *Biomedical Optics Express, 7*(10), 3882–3898. https://pmc.ncbi.nlm.nih.gov/articles/5102531
- Barry, R. J., Clarke, A. R., Johnstone, S. J., et al. (2007). EEG differences between eyes-closed and eyes-open resting conditions. *Clinical Neurophysiology, 118*(12), 2765–2773. https://ro.uow.edu.au/hbspapers/1640
- Birkett, M. A. (2011). The Trier Social Stress Test protocol for inducing psychological stress. *Journal of Visualized Experiments, 56*, e3238. https://doi.org/10.3791/3238
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Dedovic, K., et al. (2005). The Montreal Imaging Stress Task: Using functional imaging to investigate the effects of perceiving and processing psychosocial stress in the human brain. *Journal of Psychiatry & Neuroscience, 30*(5), 319–325. https://www.jpn.ca/content/30/5/319
- Dickerson, S. S., & Kemeny, M. E. (2004). Acute stressors and cortisol responses: A theoretical integration and synthesis of laboratory research. *Psychological Bulletin, 130*(3), 355–391. https://doi.org/10.1037/0033-2909.130.3.355
- Giannakakis, G., Grigoriadis, D., Giannakaki, K., Simantiraki, O., Roniotis, A., & Tsiknakis, M. (2019). Review on psychological stress detection using biosignals. *IEEE Transactions on Affective Computing* (published 2022, 13(1), 440–460). https://doi.org/10.1109/TAFFC.2019.2927337
- Jennings, J. R., Kamarck, T., Stewart, C., Eddy, M., & Johnson, P. (1992). Alternate cardiovascular baseline assessment techniques: Vanilla or resting baseline. *Psychophysiology, 29*(6), 742–750. https://doi.org/10.1111/j.1469-8986.1992.tb02052.x
- Katmah, R., Al-Shargie, F., Tariq, U., Babiloni, F., Al-Mughairbi, F., & Al-Nashash, H. (2021). A review on mental stress assessment methods using EEG signals. *Sensors, 21*(15). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8347831/
- Kirschbaum, C., Pirke, K. M., & Hellhammer, D. H. (1993). The 'Trier Social Stress Test': A tool for investigating psychobiological stress responses in a laboratory setting. *Neuropsychobiology, 28*(1–2), 76–81.
- Klimesch, W. (1999). EEG alpha and theta oscillations reflect cognitive and memory performance: A review and analysis. *Brain Research Reviews, 29*(2–3), 169–195. https://pubmed.ncbi.nlm.nih.gov/10209231/
- Krigolson, O. E., Williams, C. C., Norton, A., Hassall, C. D., & Colino, F. L. (2017). Choosing MUSE: Validation of a low-cost, portable EEG system for ERP research. *Frontiers in Neuroscience, 11*, 109. https://doi.org/10.3389/fnins.2017.00109
- Laborde, S., Mosley, E., & Thayer, J. F. (2017). Heart rate variability and cardiac vagal tone in psychophysiological research: Recommendations for experiment planning, data analysis, and data reporting. *Frontiers in Psychology, 8*, 213. https://doi.org/10.3389/fpsyg.2017.00213
- Lehrer, P. M., & Gevirtz, R. (2014). Heart rate variability biofeedback: How and why does it work? *Frontiers in Psychology, 5*, 756. https://doi.org/10.3389/fpsyg.2014.00756
- Marteau, T. M., & Bekker, H. (1992). The development of a six-item short-form of the state scale of the Spielberger State-Trait Anxiety Inventory (STAI). *British Journal of Clinical Psychology, 31*(3), 301–306. https://doi.org/10.1111/j.2044-8260.1992.tb00997.x
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics, 15*(2), 70–73. https://doi.org/10.1109/TAU.1967.1161901

### Notes on the references
- **Kirschbaum et al. (1993):** citation details confirmed. The starting number 1022 and step of 13 are **not** in the original paper. They are a widely used variant, and some studies use other numbers.
- **Dedovic et al. (2005):** the article is confirmed. The exact adaptive time-limit rule used by Clutch (×0.9 after a success, ×1.1 after a failure, 2.5–10 s) is our own simplification of the MIST.
- **Al-Shargie et al. (2016):** stress was induced with mental arithmetic under time pressure and negative feedback. It is not confirmed that this was the MIST itself.
- **Barry et al. (2007), Al-Shargie et al. (2016), Dedovic et al. (2005):** full author lists were not re-checked, hence "et al.". For Barry et al., the DOI also differs between indexes. Check the publisher records before formal citation.
- **Cohen (1988):** standard textbook, cited for the conventional small, medium and large thresholds. Not re-checked online.
- **Marteau & Bekker (1992):** an erratum was published in 2020 (*Br J Clin Psychol* 59(2):276).
