# Stress model pipeline

Turns Clutch recordings into stress scores and compares models on people they never saw.
Results, lessons and the day-2 plan: `docs/stress-model-and-study.md`.

## Setup (Windows, from the project folder)
```bash
.venv\Scripts\python -m pip install -r ml\requirements.txt
```
(macOS/Linux: `.venv/bin/python` instead of `.venv\Scripts\python`.)

## Day 2: record, export, run
1. In the app, **Dataset recording** card: participant code, tick consent, then **Run study protocol**
   (alternate orders A and B between people). Check the flip check on the calibration report: redo if it failed.
2. Export and run:
```bash
.venv\Scripts\python tools\export_dataset.py --no-zip
.venv\Scripts\python ml\run.py dataset_export --cal-seconds 60
```
`run.py` prints, per person, the average score of every block (0 = their calm, 1 = their stress; relax should be
low, hard math and hard colours high, easy math in between), for three scores: EEG + heart centroid, heart-only
centroid, and the app's live score. With 2+ people it then compares every model with leave-one-person-out,
adding the 18 EPIStress people to training (`--no-epistress` to turn off). Results go to `ml/results_<export>.md`.

Phase labels: `calib_rest`, `study_relax` = calm; `calib_stress`, `study_math_hard`, `study_stroop_hard` = stressed;
`study_math_easy` is left unlabelled (in between). Extra names: `--calm name1,name2 --stress name3`.

## Files
| File | What it does |
|---|---|
| `features.py` | one feature row per 10 s window (5 s step): EEG relative band power per sensor, alpha/beta, alpha asymmetry, heart rate, HRV, head motion |
| `clutch_data.py export/` | Clutch export (raw_eeg/raw_ppg/raw_accel/labels.csv) -> `<export>_windows.csv` |
| `run.py export/` | **one command:** features + per-phase score table + model comparison |
| `train.py [--data windows.csv] [--labels rating\|condition] [--cal-seconds 120] [--extra windows.csv]` | compares models on held-out people, calibrated like the live app |
| `epistress.py` | EPIStress tasks -> `data/epistress_windows.csv` |
| `download_epistress.py` | fetches the needed part of EPIStress (~0.6 GB, 15-30 min) into `data/epistress/` |
| `record_muse.py out.npz --seconds 180` | records raw EEG + PPG + accelerometer from muselsl, without the app |
| `predict_muse.py --calm a --stress b session...` | scores recordings between the person's own calm (0) and stress (1) |
| `safe_pickle.py` | loads the EPIStress pickles while refusing anything that could run code |

## Data
- `data/epistress_windows.csv` (6.6 MB, **not in the repo**): 12,403 ten-second feature windows from the EPIStress lab
  tasks. Get it by copying it from Riley's laptop (fastest), or rebuild it: `download_epistress.py` (15-30 min), then
  `epistress.py` (1 min). Without it, `run.py` still works; it just skips the "+ EPIStress" models.
- EPIStress: Zenodo, doi:10.5281/zenodo.16407549, **CC BY 4.0** (credit the dataset). Participants are adults with
  epilepsy (Muse S headband, Empatica E4 wristband).

## Results on EPIStress (relaxation video vs hard tasks, people never seen, 2 min + 2 min calibration)
| Model | Signals | AUC |
|---|---|---|
| per-person centroid | heart only | 0.70 |
| per-person centroid | EEG + heart | 0.64 |
| app two-anchor (live score) | alpha/beta + heart | 0.60 |
| LightGBM | all incl. motion | 0.78, but motion alone gets 0.79 (typing vs sitting still) |

Keep still while answering, eyes open in every block, and calibrate with the same stressor you demo.

## Without the app (raw files only)
```bash
.venv\Scripts\python ml\record_muse.py calm.npz --seconds 180
.venv\Scripts\python ml\record_muse.py math_hard.npz --seconds 180
.venv\Scripts\python ml\predict_muse.py --calm calm.npz --stress math_hard.npz other.npz
```
Needs `muselsl stream --ppg --acc` running in another terminal.
