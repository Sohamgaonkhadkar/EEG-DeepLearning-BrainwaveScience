"""Build the concise assessment report strictly from saved experimental records."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent

def read_jsonl(path):
 return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def pct(x): return f'{100*x:.2f}%'

def main():
 original=read_jsonl(ROOT/'part1_diagnose/original_results.jsonl')[-1]
 p1=read_jsonl(ROOT/'part1_diagnose/results.jsonl')
 all_result=next(r for r in reversed(p1) if r['channels']=='all')
 frontal=next(r for r in reversed(p1) if r['channels']=='frontal')
 p2=read_jsonl(ROOT/'part2_transformer/results.jsonl')
 models=['EEGNet','VanillaEEGTransformer','YourModel']
 selected={m:{s:next(r for r in reversed(p2) if r['model']==m and r['seed']==s and r['epochs']==40) for s in [0,1,2]} for m in models}
 stats={m:np.array([selected[m][s]['test_acc'] for s in [0,1,2]]) for m in models}
 figures=json.loads((ROOT/'figure_interpretations.json').read_text(encoding='utf-8'))
 audits=json.loads((ROOT/'part1_diagnose/data_audit.json').read_text())
 flagged=sum(a['epochs_peak_over_500uv'] for a in audits)
 quality_table='\n'.join(f"| {a['subject']} | {a['shape'][0]} | {a['class_counts'][0]}/{a['class_counts'][1]} | {a['epochs_peak_over_500uv']} |" for a in audits)
 rows=[]
 for m in models:
  a=stats[m]
  rows.append('| '+m+' | '+' | '.join(pct(x) for x in a)+f' | {pct(a.mean())} | {100*a.std():.2f} pp | {pct(a.min())} | {pct(a.max())} |')
 pvalue=binomtest(26,45,.5,alternative='greater').pvalue
 text=f'''# EEG × Deep Learning — Assessment Answers

## Part 1: Diagnose & Trust

### Task 1: Six Defects

The six independently corrected behaviors below are evidenced in the untouched starter (commit `b76b586`). Train/test and train/validation subject mixing are counted separately because they invalidate different estimates; the PDF provides no answer key, so this is not a claim to know the examiner's intended six-way grouping. No event-label or Hilbert-power error was invented to reach six.

| Defect / original location | Problem and effect | Correction and verification |
|---|---|---|
| 1. `data.py:RUNS` | Runs 5/9/13 are executed movements mislabeled as imagery; this changes the task, with no guaranteed accuracy direction. | Use 6/10/14; verified against MNE's run documentation and saved data audits. Commit `11f4b6f`. |
| 2. `train.py:main`, first `train_test_split` | Random trials from a person enter both optimization and test sets; subject/session signatures can make the estimate optimistic for unseen people. | Hold out subjects 9/10 completely; actual group assertions and tests verify disjointness. Commit `8c1dca8`. |
| 3. `train.py:main`, second `train_test_split` | Validation shares people with optimization, so checkpoint validation estimates within-person rather than unseen-person performance; this can make validation optimistic, while the final test effect is unpredictable. | Use subjects 7/8 exclusively for validation; train only 1–6. Verified by group tests and saved subject lists. Commit `c4ede07`. |
| 4. `train.py:main`, channel mean/std | Scaling fits all samples, including held-out people; test distribution information leaks into preprocessing, with no guaranteed sign of accuracy change. | Fit on 1–6 and reuse unchanged on 7–10; a held-out perturbation test proves training scaling is invariant. Commit `24c4e0d`. |
| 5. `train.py:main`, `if te_acc > best_acc` | The maximum of repeated test evaluations chooses weights and produces optimistic selection bias. | Choose earliest maximum validation accuracy, restore that state, infer test once; verify selected epoch against history. Commit `ded795d` (obsolete print removed in `cc88d2e`). |
| 6. `erd.py:BASELINE` | 0–0.5 s is after cue onset, not pre-cue rest; it changes the ERD reference and can attenuate or reverse apparent ERD. It has no direct effect on classifier accuracy. | Reference -1.5 to -0.5 s; verify synthetic constant baseline and a known 50% power reduction, plus real mu/beta plots. Commit `3fe8efd`. |

### Accuracy Before and After

| Configuration | Accuracy | Test samples | Checkpoint |
|---|---:|---:|---|
| Untouched original, all channels, seed 0 | {pct(original['test_acc'])} | {original['n_test']} | Maximum test accuracy over 30 epochs |
| Corrected, all channels, seed 0 | {pct(all_result['test_acc'])} | {all_result['n_test']} | Validation-selected epoch {all_result['best_epoch']+1}/30 |
| Corrected, frontal, seed 0 | {pct(frontal['test_acc'])} | {frontal['n_test']} | Validation-selected epoch {frontal['best_epoch']+1}/30 |

Original: executed-movement runs, stratified trial splits (64%/16%/20%), all-data channel normalization and test-based selection. Corrected: imagery runs, subjects 1–6 / 7–8 / 9–10 and optimization-only normalization. Both use AdamW, learning rate 0.001, weight decay 0.001, batch 64, 30 epochs, EEGNet kernel 80 and dropout 0.5. These are different tasks and generalization targets; their accuracy difference is not an isolated causal estimate of a particular fix. Full original warnings/output are in `logs/original_part1.log`; its result is retained separately. Environment compatibility issues (an interactive MNE cache prompt and MNE 1.12 rejecting a bare NumPy ufunc in `apply_function`) were fixed without changing the scientific computations; failed attempts are retained in the logs and are not counted among the six scientific defects.

### Task 2: Mu-Band ERD

{figures['erd']}

The formula is 100 × (mean trial power − pre-cue reference power) / reference power. Continuous mu 8–13 Hz or beta 13–30 Hz filtering precedes Hilbert envelope squaring and epoching; `erd_mu_summary.json` and `erd_beta_summary.json` give measured 0–4 s averages. Gray shading marks the -1.5 to -0.5 s baseline. These are equal-subject means, not uncertainty intervals or inferential tests.

### Task 3: Frontal Versus All Channels

The predefined frontal subset (Fp1, Fpz, Fp2, AF3, AF4, F3, Fz, F4) gives {pct(frontal['test_acc'])}, versus {pct(all_result['test_acc'])} with all 64 electrodes under the same protocol; balanced binary chance is 50%. Frontal prediction is compatible with cue-linked vertical eye movements, but also with neural signals, source mixing and noise, so accuracy alone does not identify an ocular mechanism. The top/bottom cue is confounded with class, and this experiment cannot separate that cue response from imagery. Counterbalance cue position independently of class, record EOG/eye tracking and test transfer across cue mappings to establish the source.

### Task 4: Short Answers

1. **Statistical significance:** 58% of 45 is approximately 26/45; the one-sided exact binomial test against p=0.5 gives p={pvalue:.4f}, so this is not significant at 0.05 (29/45 is the first significant count). The test assumes independent trials and a fixed classifier; serial dependence or adaptive selection calls for an appropriate block/subject permutation analysis.
2. **Depthwise convolution:** It learns D spatial electrode combinations independently for each of F1 temporal features, emphasizing informative scalp patterns and suppressing competing sources. These are learned spatial filters, not necessarily anatomical source localizations.
3. **Confound prevention:** Randomize and counterbalance visual target position independently of imagery class, with fixation/EOG monitoring; class must no longer be predictable from gaze direction alone.

## Part 2: Build & Defend

### Hypothesis

Learned 32-sample temporal patches impose local aggregation and reduce attention cost, which may stabilize learning relative to individual-sample attention on a small subject pool; superiority to EEGNet is not assumed. Recorded in `EXECUTION_PLAN.md`, commit `c9c5bc2`, before implementation.

### Architectural Modification

I deliberately validate the starter's existing Conv1d patch-tokenizer bet rather than claim its invention. I add right padding to cover the last partial patch, shape checks and embedding/backpropagation tests; no extra spatial, spectral or augmentation mechanism is added. Input is B×64×481, patch width/stride 32 (200 ms), padding 31 samples, 16 patch tokens plus CLS, embedding 64, four heads, three encoder layers, feedforward width 128, dropout 0.3 and output B×2 (embedding B×64).

Vanilla uses 481 timestep tokens plus CLS: **482×482** scores/head/layer (232,324 entries). YourModel uses **17×17** (289 entries), a **803.89×** score-count reduction; there are four heads and three layers in each. This is quadratic-attention cost, not a measured wall-clock speedup: token projection and other operations still cost time. Parameter counts: EEGNet {selected['EEGNet'][0]['n_params']:,}; vanilla {selected['VanillaEEGTransformer'][0]['n_params']:,}; YourModel {selected['YourModel'][0]['n_params']:,}. Patching reduces tokens but increases tokenizer parameters, so it does not automatically reduce overfitting.

### Experimental Setup

Imagery runs 6/10/14, 64 channels, 160 Hz, continuous 7–30 Hz filtering and inclusive 0–3 s epochs (481 samples), fists=0 / feet=1. Optimization subjects 1–6 ({selected['EEGNet'][0]['n_train']} trials), validation 7–8 ({selected['EEGNet'][0]['n_validation']}), held-out test 9–10 ({selected['EEGNet'][0]['n_test']}); validation comes only from the allowed 1–8 pool. All three models use seeds 0/1/2, 40 epochs, cross entropy, AdamW (lr=0.001, decay=0.001), batch 16, the same channel normalization fitted on 1–6, and earliest maximum validation accuracy for checkpoint selection. CPU execution and exact library versions are in `environment_actual.json`. EEGNet carries the Part 1 kernel/dropout values; the Part 2 filter/window differ from Part 1, so the resulting accuracy is a separate measurement.

### Results

| Model | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD | Worst | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Interpretation

YourModel's mean accuracy is {pct(stats['YourModel'].mean())}, compared with EEGNet {pct(stats['EEGNet'].mean())} and vanilla {pct(stats['VanillaEEGTransformer'].mean())}. The paired seed differences relative to EEGNet are {', '.join(f'{100*x:+.2f}' for x in stats['YourModel']-stats['EEGNet'])} percentage points, and relative to vanilla are {', '.join(f'{100*x:+.2f}' for x in stats['YourModel']-stats['VanillaEEGTransformer'])}; these explicitly show whether the ordering repeats across the three seeds. The measured standard deviations describe seed sensitivity on this fixed two-person test population, not uncertainty across new subjects or proof that patching improves optimization. EEGNet's temporal/spatial convolution and pooling constrain the hypothesis space; vanilla must learn useful structure from few people, while a large patch projection can itself overfit. More held-out subject folds, matched parameter budgets, patch-size ablation selected on validation data, loss-curve analysis and independent cue controls would strengthen any conclusion.

### Figure

{figures['embedding']}

### Data Quality and Leakage Audit

The untouched original run deliberately retains its documented leaks for the required diagnostic reproduction; the controls below apply to the corrected experiments.

Part 1 retains 42 of 45 annotated task events for each subject: two epochs meet a `BAD boundary` and one is `TOO_SHORT` because the full -2 to 5 s epoch extends beyond valid recording data. Thus there are 420 balanced trials, including 84 test trials. A fixed peak-over-500-µV diagnostic flags {flagged}/420 epochs (including all 42 from subject 3); this is a residual-artifact limitation, not a reason to change thresholds using test outcomes. The high-amplitude epochs are retained to preserve the specified controlled comparison and cue-confound investigation; their scalp origin cannot be inferred from a peak alone.

| Subject | Retained trials | Fists/feet | Epoch peak >500 µV |
|---|---:|---:|---:|
{quality_table}

- Every subject's actual sampling rate, 64 unique standardized electrode names and identical channel order are checked; labels are mapped explicitly from T1/T2. Subject-level shapes/class counts and units (Part 1 µV, Part 2 V before scaling) are saved in `data_audit.json`.
- Invalid/non-finite samples, flat channels and exact duplicated epochs raise errors. Audits record absolute-amplitude and epoch-peak percentiles and flags above 500 µV; these are transparent diagnostics, not a test-adapted rejection rule. No automatic amplitude/ICA rejection or fitted artifact cleaning is applied. Under the fixed 7–30 Hz Part 2 preprocessing none of its 450 epochs exceeds the 500 µV diagnostic threshold; this does not establish that ocular or muscle artifacts are absent.
- Subject identities and sample hashes prevent duplicate windows crossing evaluation boundaries. Filtering is deterministic, label-free and performed before epoching, independently by subject with annotated run boundaries respected; no window or subject crosses a split. Zero-phase filtering is appropriate for this offline task and is not a claim of causal online operation.
- Shared normalization is fitted on optimization subjects only. Tests verify held-out perturbations cannot alter it, disjoint subject partitions and validation-selected checkpoints. Models receive preprocessed EEG directly; ERD/band summaries are separate physiological analyses, never classifier inputs.
- Architecture, frontal electrodes, seeds and hyperparameters were frozen before corrected test inference. No test metric is read by a training/checkpoint-selection function; test labels only score locked predictions and color the post-hoc t-SNE. No seed is selected, no augmentation is applied, and loss histories/confusion matrices/per-class/per-subject metrics are preserved.

### Limitations

Only two test people constrain population-level inference; three seeds do not create independent test cohorts. Residual eye/muscle artifacts, cue confounding, scalp mixing and reference choice remain possible; no accuracy establishes motor-cortex specificity. Scaling uses the starter's 1e-6 denominator stabilizer (1 µV in Part 2's volt units), and the EEGNet and transformer parameter budgets differ. Validation uses two of the eight allowed training people, leaving six for gradient fitting; results describe that declared protocol rather than a final refit on all eight. No causal attention-map or statistically significant superiority claim is made.

Dataset/run and cue context: [PhysioNet EEGBCI](https://physionet.org/content/eegmmidb/1.0.0/) and MNE's installed `eegbci.load_data` documentation. The local official PDF is authoritative for assessment deliverables.
'''
 from report_refits import refit_report
 text += refit_report(ROOT)
 (ROOT/'ANSWERS.md').write_text(text,encoding='utf-8')
 print('Generated ANSWERS.md from real records')

if __name__=='__main__': main()
