# Muse EEG datasets: stress and emotion

Public datasets recorded with a Muse headband (TP9, AF7, AF8, TP10, 256 Hz), the same sensors as our Muse 2.
Downloaded files live in `data/` (gitignored). Checked October 2026.

## Downloaded

| Folder | Dataset | People | Labels | Signal | Fit for Clutch |
|---|---|---|---|---|---|
| `data/epistress/` | **EPIStress** (Zenodo 10.5281/zenodo.16407549, CC BY 4.0) | 20 epilepsy patients (Muse S) | Lab tasks at easy/hard difficulty: mental arithmetic, Stroop, N-back, Sudoku, plus a relaxation video; self-reported stress and workload (5-point), NASA-TLX, arousal/valence sliders, PANAS | Raw EEG, plus Empatica E4 heart rate and skin conductance | **Best match.** Arithmetic under pressure vs relaxation, raw signal, per-task stress ratings. Patients have epilepsy, so it may not transfer to healthy users |
| `data/mendeley-exam-stress/` | **Student exam stress** (Mendeley 10.17632/fyj9by2t22.1, CC BY 4.0) | 26 students (Muse 2, Mind Monitor) | One questionnaire stress score per student (Low 7, Moderate 8, High 9, Very high 1, No 1) | Raw EEG + Muse band powers + accelerometer, ~2 min per student during an exam | Good for checking whether alpha/beta tracks trait stress across people. One label per person, not per moment |
| `data/kaggle-mental-state/` | **EEG brainwave: mental state** (Bird et al., Kaggle birdy654) | 4 (Muse) | relaxed / neutral / concentrating, 60 s each; label column 0/1/2 (mapping to names not confirmed in the file) | Pre-computed features only (988 columns), no raw EEG | Calm vs concentrating, close to our alpha/beta idea. Tiny and features only |
| `data/kaggle-emotions/` | **EEG brainwave: feeling emotions** (Bird et al., Kaggle birdy654) | 2 (Muse) | positive / neutral / negative (valence from film clips) | Pre-computed features only (2,548 columns) | Mood, not stress. No subject IDs, so no subject-wise split |

EPIStress is 53 GB unpacked, so only a subset is here: lab-session raw Muse EEG (`Lab/Raw/Muse/Muse.csv.gz`, kept gzipped),
`Lab/Task_Labels.csv`, `Lab/Panas_Labels.csv`, `Trigger_Questionnaire.csv`, `Notes.pdf`, E4 heart rate / skin conductance / beat intervals,
and the in-situ questionnaires. The `.pickle` files were left out on purpose: loading a pickle from an outside source can run arbitrary code.

## Need an access request

| Dataset | People | Labels | How to get it |
|---|---|---|---|
| **NeuroSense** (Zenodo 14002375) | 30 students, 40 sessions each (Muse 2) | Valence / arousal / dominance self-ratings on music videos | Request form on the dataset website + Data Use Agreement (research only). Best Muse emotion set |
| **EmoWork** (Zenodo 10.5281/zenodo.17513978, CC BY-NC-ND) | Call-centre workers (Muse S + Polar H10 + Empatica E4) | Self-reported emotion and stress during role-plays with angry customers | Zenodo access request + Google form + DUA. 185 GB |
| **Perceived mental stress** (Arsalan, Majid et al., UET Taxila) | Not stated | Perceived stress (questionnaire), pre / during / post activity | The public OneDrive link is dead (asks for sign-in); email aamir.arsalan@uettaxila.edu.pk |
| **PROPER** (same lab) | 40 (Muse) | Big Five personality, before / during / after a 5-min public speech | Signed release agreement via the lab's form. Public speaking is a classic stressor, but labels are personality |

## Seen but not downloaded

| Dataset | Why not |
|---|---|
| Muse2 motor imagery (Kaggle muhammadatefelkaffas) | Labels are left / right / relaxed imagery, not stress or emotion |
| Muse2-eeg (GitHub yueeyue, MIT) | 8 people, activities eat / game / read / rest / toy / tv; "game vs rest" is at best a weak arousal proxy |
| Consumer neuroscience raw (Kaggle julianurariani) | Muse Monitor CSVs, but the labels are not described |
| MindBigData (Muse) | Labels are digits being viewed |
| DREAMER, SAM-40, DEAP | Emotion / stress with valence-arousal labels, but recorded on Emotiv or research caps, not a Muse |

## Cautions
- Split by person, never by random windows: neighbouring windows from one recording leak between train and test.
- Most labels are the task or stimulus, not how the person felt. EPIStress and NeuroSense are the exceptions, with self-ratings.
- Few people per dataset: expect large person-to-person differences, which is why Clutch calibrates per user.
