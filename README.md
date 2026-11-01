# EEG × Deep Learning — Live Exam (3 hours)

Two parts, 90 minutes each, with a 5-minute break between them. Read the
problem-statement PDF for the full brief; this file is the map.

```
part1_diagnose/     Part 1 — a buggy motor-imagery pipeline you must fix and interpret
part2_transformer/  Part 2 — give self-attention a fair shot on tiny EEG data
```

## Before the clock starts

Set up the environment once (conda covers both parts):

```bash
conda env create -f environment.yml
conda activate eeg-dl-exam
```

Or use pip per part (each folder has its own `requirements.txt`, they overlap):

```bash
pip install -r part1_diagnose/requirements.txt
```

Then kick off the data download at the start of each part and read code while
it runs:

```bash
cd part1_diagnose       && python train.py
# ...later...
cd ../part2_transformer && python main.py
```

Both parts use the PhysioNet EEGBCI dataset (subjects 1–10, 64 channels,
160 Hz), downloaded and cached automatically by MNE (~80 MB total). CPU is
enough; each run is 1–2 minutes.

## The arc

- **Part 1 — Diagnose & Trust.** The pipeline runs cleanly but the accuracy it
  prints is wrong. Find the planted defects, fix them, interpret the ERD, and
  work out what signal the model is really using. You end with an EEGNet
  baseline whose number you can *trust*.
- **Part 2 — Build & Defend.** Carry that trusted baseline forward. Inject one
  inductive bias into an attention-based decoder and show, honestly, whether it
  helps — and whether you understand why.

Part 1 tests rigor; Part 2 tests ideas. They're weighted equally. Accuracy is
not scored — your reasoning is.

## What you hand in

One GitHub repo (commit as you work, one commit per bug fix, no squashing) with
both parts' code, an `ANSWERS.md`, and the generated artifacts
(`results.jsonl`, `erd_mu.png`, the Part 2 figure). Then a live code
walkthrough with Q&A. Details in the PDF.


## Completed assessment: reproduction

The official brief is `../EEG_DL_Exam_3hr.pdf`; the unmodified starter is local
commit `b76b586` and the original ZIP remains untouched outside this repository.
`ANSWERS.md` contains measured results and the scientific interpretation;
`WALKTHROUGH_NOTES.md` prepares the defense. No remote is configured.

Tested on CPU with Python 3.10.18; exact direct versions are recorded in
`requirements-lock.txt` and `environment_actual.json`. The supplied Conda YAML
remains available. For an isolated pip environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$env:MNE_DATASETS_EEGBCI_PATH = "$PWD\mne_data"
New-Item -ItemType Directory -Force $env:MNE_DATASETS_EEGBCI_PATH
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

This workspace uses `../.venv/Scripts/python.exe` (a local venv with the existing
PyTorch installation exposed through system site packages); a fresh installation
using the commands above does not need that existing environment.

Run from each part directory, using the appropriate Python executable:

```powershell
python train.py --channels all --seed 0
python train.py --channels frontal --seed 0
python erd.py --band mu
python erd.py --band beta
# From part2_transformer:
python main.py --epochs 40 --seeds 0 1 2
```

Part 1 runs 30 epochs, training on subjects 1-6, selecting on 7-8 and testing on
9-10. Part 2 uses the same split, 40 epochs for every model and every seed, with
batch size 16 to keep the vanilla transformer's quadratic training memory modest.
The vanilla architecture is unchanged. Its EEGNet configuration uses the Part 1
kernel length 80 and dropout 0.5; the Part 2 input remains 7-30 Hz and 0-3 seconds,
so its accuracy is not directly the Part 1 1-40 Hz, 0-4 second measurement.
All models optimize raw preprocessed EEG; no handcrafted classifier features,
data augmentation, ICA, adaptive channel selection, or test-fitted transformations
are used. Predeclared quality checks report large amplitudes and stop on invalid
values, flat channels, inconsistent channel order or duplicate epochs.

`results.jsonl` is append-only: rerunning adds new experiments. Use a fresh output
file via Part 2 `--results path.jsonl` for an additional comparison. The original
flawed measurement is kept separately in `part1_diagnose/original_results.jsonl`.
Transcripts are in `logs/`, quality records in each part's `data_audit.json`, and
confusion matrices, per-class metrics, per-subject accuracies and loss histories
are in the result records. Test metrics do not select checkpoints or seeds.

`run_assessment.py` is the local orchestration record: it expects the original ZIP
extracted at `../original_starter/eeg_dl_exam_3hr` and the cache at `../mne_data`.
It runs the preserved original, the corrected experiments, both ERD bands and the
three-model comparison. Required figures and small result records are versioned;
raw EDF recordings, virtual environments and checkpoints are excluded.


## Separate eight-subject refits

The required development results remain in their original files. The added
final refit protocol is documented in `TRAINING_STRATEGY.md`; exact durations
and seeds are frozen in `refit_plan.json`. From this project root, run only
after the development comparison has completed:

```powershell
python refit.py --prepare-plan
# Review/commit the fixed plan before training, as in the local history.
python refit.py
python generate_report.py
python verify_submission.py
```

The script refuses to overwrite existing refit result files. It optimizes on
all subjects 1-8 for a validation-derived fixed duration, then scores subjects
9-10 after every refit model has finished. Separate outputs live in `refits/`.
The existing test cohort has already been scored for the required comparisons;
these follow-up results do not constitute a newly untouched external test.
