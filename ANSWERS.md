# Cross-Subject EEG Motor Imagery: Methods and Results

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
| Untouched original, all channels, seed 0 | 66.67% | 84 | Maximum test accuracy over 30 epochs |
| Corrected, all channels, seed 0 | 52.38% | 84 | Validation-selected epoch 9/30 |
| Corrected, frontal, seed 0 | 54.76% | 84 | Validation-selected epoch 6/30 |

Original: executed-movement runs, stratified trial splits (64%/16%/20%), all-data channel normalization and test-based selection. Corrected: imagery runs, subjects 1–6 / 7–8 / 9–10 and optimization-only normalization. Both use AdamW, learning rate 0.001, weight decay 0.001, batch 64, 30 epochs, EEGNet kernel 80 and dropout 0.5. These are different tasks and generalization targets; their accuracy difference is not an isolated causal estimate of a particular fix. Full original warnings/output are in `logs/original_part1.log`; its result is retained separately. Environment compatibility issues (an interactive MNE cache prompt and MNE 1.12 rejecting a bare NumPy ufunc in `apply_function`) were fixed without changing the scientific computations; failed attempts are retained in the logs and are not counted among the six scientific defects.

### Task 2: Mu-Band ERD

After a shared early positive transient, mu power falls below baseline more strongly for fists at C3 (0–4 s mean -9.8%) and C4 (-4.7%) than for feet (+1.3% and -1.2%). At Cz, feet show only slightly more mean desynchronization (-2.2%) than fists (-0.6%), with substantial fluctuations. The stronger lateral fist response and weak medial foot contrast are broadly compatible with hand/foot somatotopy but do not provide a clean spatial dissociation. Weak Cz could reflect source mixing, reference choice, intersubject variation, noise or cue-related activity; this plot alone cannot identify the cause.

The formula is 100 × (mean trial power − pre-cue reference power) / reference power. Continuous mu 8–13 Hz or beta 13–30 Hz filtering precedes Hilbert envelope squaring and epoching; `erd_mu_summary.json` and `erd_beta_summary.json` give measured 0–4 s averages. Gray shading marks the -1.5 to -0.5 s baseline. These are equal-subject means, not uncertainty intervals or inferential tests.

### Task 3: Frontal Versus All Channels

The predefined frontal subset (Fp1, Fpz, Fp2, AF3, AF4, F3, Fz, F4) gives 54.76%, versus 52.38% with all 64 electrodes under the same protocol; balanced binary chance is 50%. Frontal prediction is compatible with cue-linked vertical eye movements, but also with neural signals, source mixing and noise, so accuracy alone does not identify an ocular mechanism. The top/bottom cue is confounded with class, and this experiment cannot separate that cue response from imagery. Counterbalance cue position independently of class, record EOG/eye tracking and test transfer across cue mappings to establish the source.

### Task 4: Short Answers

1. **Statistical significance:** 58% of 45 is approximately 26/45; the one-sided exact binomial test against p=0.5 gives p=0.1856, so this is not significant at 0.05 (29/45 is the first significant count). The test assumes independent trials and a fixed classifier; serial dependence or adaptive selection calls for an appropriate block/subject permutation analysis.
2. **Depthwise convolution:** It learns D spatial electrode combinations independently for each of F1 temporal features, emphasizing informative scalp patterns and suppressing competing sources. These are learned spatial filters, not necessarily anatomical source localizations.
3. **Confound prevention:** Randomize and counterbalance visual target position independently of imagery class, with fixation/EOG monitoring; class must no longer be predictable from gaze direction alone.

## Part 2: Build & Defend

### Hypothesis

Learned 32-sample temporal patches impose local aggregation and reduce attention cost, which may stabilize learning relative to individual-sample attention on a small subject pool; superiority to EEGNet is not assumed. Recorded in `EXECUTION_PLAN.md`, commit `c9c5bc2`, before implementation.

### Architectural Modification

This study evaluates the supplied Conv1d patch tokenizer and adds right padding to cover the last partial patch, shape checks and embedding/backpropagation tests; no extra spatial, spectral or augmentation mechanism is added. Input is B×64×481, patch width/stride 32 (200 ms), padding 31 samples, 16 patch tokens plus CLS, embedding 64, four heads, three encoder layers, feedforward width 128, dropout 0.3 and output B×2 (embedding B×64).

Vanilla uses 481 timestep tokens plus CLS: **482×482** scores/head/layer (232,324 entries). Temporal-Patch EEG Transformer uses **17×17** (289 entries), a **803.89×** score-count reduction; there are four heads and three layers in each. This is quadratic-attention cost, not a measured wall-clock speedup: token projection and other operations still cost time. Parameter counts: EEGNet 2,738; vanilla 135,746; Temporal-Patch EEG Transformer 232,962. Patching reduces tokens but increases tokenizer parameters, so it does not automatically reduce overfitting.

EEGNet uses eight temporal kernels of width 80, depthwise full-electrode convolution with multiplier two, then depthwise/separable temporal convolution with 16 features, ELU, batch normalization, pooling factors 4 and 8, dropout 0.5 and a two-class linear head. Vanilla projects each 64-channel timestep into dimension 64, adds CLS and learned positions, applies three post-norm transformer layers (four heads, feedforward width 128, GELU, dropout 0.3), and classifies the normalized CLS embedding. Temporal-Patch EEG Transformer preserves that transformer stack but replaces timestep projection with the temporal Conv1d tokenizer. All encoder layers have distinct parameter storage despite identical initial cloned layer values; this inherited initialization is unchanged across the two transformer configurations.

### Experimental Setup

Imagery runs 6/10/14, 64 channels, 160 Hz, continuous 7–30 Hz filtering and inclusive 0–3 s epochs (481 samples), fists=0 / feet=1. Optimization subjects 1–6 (270 trials), validation 7–8 (90), held-out test 9–10 (90); validation comes only from the allowed 1–8 pool. The planned comparison uses seeds 0/1/2 and 40 epochs for each model (completed counts and missing runs are explicit below), cross entropy, AdamW (lr=0.001, decay=0.001), batch 16, the same channel normalization fitted on 1–6, and earliest maximum validation accuracy for checkpoint selection. CPU execution and exact library versions are in `environment_actual.json`. EEGNet carries the Part 1 kernel/dropout values; the Part 2 filter/window differ from Part 1, so the resulting accuracy is a separate measurement.

### Results

Uniform random guessing is 50%. A majority-class baseline fitted on optimization labels alone chooses feet (training counts [133, 137]) and scores 48.89% on the full test set; test labels do not choose its class.

| Model | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD | Worst | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet | 44.44% | 51.11% | 48.89% | 48.15% | 2.77 pp | 44.44% | 51.11% |
| VanillaEEGTransformer | 43.33% | 46.67% | 54.44% | 48.15% | 4.66 pp | 43.33% | 54.44% |
| Temporal-Patch EEG Transformer | 46.67% | 45.56% | 50.00% | 47.41% | 1.89 pp | 45.56% | 50.00% |

### Interpretation

Temporal-Patch EEG Transformer's mean accuracy is 47.41%, compared with EEGNet 48.15% and vanilla 48.15%. Completed seed counts are EEGNet 3, vanilla 3, Temporal-Patch EEG Transformer 3; missing runs are excluded from summaries. Descriptive comparisons do not select a model. The all-seed scores above provide the comparison; these explicitly show whether the ordering repeats across the three seeds. The measured standard deviations describe seed sensitivity on this fixed two-person test population, not uncertainty across new subjects or proof that patching improves optimization. EEGNet's temporal/spatial convolution and pooling constrain the hypothesis space; vanilla must learn useful structure from few people, while a large patch projection can itself overfit. More held-out subject folds, matched parameter budgets, patch-size ablation selected on validation data, loss-curve analysis and independent cue controls would strengthen any conclusion.

### Figure

The figure uses the validation-selected original Temporal-Patch EEG Transformer seed 0, preselected independently of test accuracy. Fixed-random-seed (0) t-SNE projects its 64-dimensional CLS embeddings from all 90 test trials; colors identify fists/feet and markers identify subjects 9/10. Axes are arbitrary t-SNE coordinates. It is a descriptive visualization of an already evaluated cohort, not a model-selection instrument: local neighborhoods and global distances can be distorted, and visible separation would not establish accuracy, statistical significance or physiological specificity. Visual inspection shows substantial overlap of the two class colors and both subject markers, without a clean class-separated cluster structure.

### Data Quality and Leakage Audit

The untouched original run deliberately retains its documented leaks for the required diagnostic reproduction; the controls below apply to the corrected experiments.

Part 1 retains 42 of 45 annotated task events for each subject: two epochs meet a `BAD boundary` and one is `TOO_SHORT` because the full -2 to 5 s epoch extends beyond valid recording data. Thus there are 420 balanced trials, including 84 test trials. A fixed peak-over-500-µV diagnostic flags 91/420 epochs (including all 42 from subject 3); this is a residual-artifact limitation, not a reason to change thresholds using test outcomes. The high-amplitude epochs are retained to preserve the specified controlled comparison and cue-confound investigation; their scalp origin cannot be inferred from a peak alone.

| Subject | Retained trials | Fists/feet | Epoch peak >500 µV |
|---|---:|---:|---:|
| 1 | 42 | 21/21 | 3 |
| 2 | 42 | 21/21 | 0 |
| 3 | 42 | 21/21 | 42 |
| 4 | 42 | 21/21 | 0 |
| 5 | 42 | 21/21 | 0 |
| 6 | 42 | 21/21 | 0 |
| 7 | 42 | 21/21 | 0 |
| 8 | 42 | 21/21 | 0 |
| 9 | 42 | 21/21 | 40 |
| 10 | 42 | 21/21 | 6 |

- Every subject's actual sampling rate, 64 unique standardized electrode names and identical channel order are checked; labels are mapped explicitly from T1/T2. Subject-level shapes/class counts and units (Part 1 µV, Part 2 V before scaling) are saved in `data_audit.json`.
- Invalid/non-finite samples, flat channels and exact duplicated epochs raise errors. Audits record absolute-amplitude and epoch-peak percentiles and flags above 500 µV; these are transparent diagnostics, not a test-adapted rejection rule. No automatic amplitude/ICA rejection or fitted artifact cleaning is applied. Under the fixed 7–30 Hz Part 2 preprocessing none of its 450 epochs exceeds the 500 µV diagnostic threshold; this does not establish that ocular or muscle artifacts are absent.
- Subject identities and sample hashes prevent duplicate windows crossing evaluation boundaries. Filtering is deterministic, label-free and performed before epoching, independently by subject with annotated run boundaries respected; no window or subject crosses a split. Zero-phase filtering is appropriate for this offline task and is not a claim of causal online operation.
- Shared normalization is fitted on optimization subjects only. Tests verify held-out perturbations cannot alter it, disjoint subject partitions and validation-selected checkpoints. Models receive preprocessed EEG directly; ERD/band summaries are separate physiological analyses, never classifier inputs.
- Architecture, frontal electrodes, seeds and hyperparameters were frozen before corrected test inference. No test metric is read by a training/checkpoint-selection function; test labels only score locked predictions and color the post-hoc t-SNE. No seed is selected, no augmentation is applied, and loss histories/confusion matrices/per-class/per-subject metrics are preserved.

### Limitations

Only two test people constrain population-level inference; three seeds do not create independent test cohorts. Residual eye/muscle artifacts, cue confounding, scalp mixing and reference choice remain possible; no accuracy establishes motor-cortex specificity. Scaling uses the starter's 1e-6 denominator stabilizer (1 µV in Part 2's volt units), and the EEGNet and transformer parameter budgets differ. Validation uses two of the eight allowed training people, leaving six for gradient fitting; results describe that declared protocol rather than a final refit on all eight. No causal attention-map or statistically significant superiority claim is made.

Dataset/run and cue context: [PhysioNet EEGBCI](https://physionet.org/content/eegmmidb/1.0.0/) and MNE's installed `eegbci.load_data` documentation. The local official PDF is authoritative for assessment deliverables.

## Optional eight-subject refits

Not executed; mandatory Stage A results are reported above. Any later refit is exploratory because subjects 9–10 have already been evaluated. Fixed durations must be frozen from validation before refit scoring.

## Execution safeguards and historical limitations

The original five completed runs are preserved without retraining. They have no persistent selected weights or evaluation-mode training accuracy; retrospective validation-subject predictions cannot be recovered reliably. Newly completed runs save a distinct checkpoint per model/seed, verify identical validation predictions after reloading, and record evaluation-mode training accuracy and loss. Additional logging preserves RNG state; a regression check verifies identical optimization histories and selected weights. Training loss in all original histories is the online optimization loss, distinct from evaluation-mode loss. Part 2 uses baseline=None, matching all completed baselines; pre-cue referencing applies only to ERD. The interrupted vanilla seed-2 attempt is incomplete and excluded; its replacement starts from the original seed.

## Selected-checkpoint metrics

Epochs below are one-based. Training accuracy is measured in evaluation mode at the selected epoch; unavailable historical values are explicitly marked. Full class recalls and confusion matrices are saved in results.jsonl.

| Model | Seed | Epoch | Train accuracy | Validation accuracy | Validation loss | S7 | S8 | S9 | S10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EEGNet | 0 | 19 | Unavailable | 72.22% | 0.6259 | Unavailable | Unavailable | 48.89% | 40.00% |
| EEGNet | 1 | 36 | Unavailable | 67.78% | 0.6101 | Unavailable | Unavailable | 48.89% | 53.33% |
| EEGNet | 2 | 1 | Unavailable | 64.44% | 0.6899 | Unavailable | Unavailable | 51.11% | 46.67% |
| VanillaEEGTransformer | 0 | 26 | Unavailable | 60.00% | 0.7097 | Unavailable | Unavailable | 42.22% | 44.44% |
| VanillaEEGTransformer | 1 | 31 | Unavailable | 62.22% | 1.6209 | Unavailable | Unavailable | 46.67% | 46.67% |
| VanillaEEGTransformer | 2 | 1 | 50.74% | 50.00% | 0.6935 | 51.11% | 48.89% | 62.22% | 46.67% |
| Temporal-Patch EEG Transformer | 0 | 4 | 75.93% | 63.33% | 0.6510 | 64.44% | 62.22% | 42.22% | 51.11% |
| Temporal-Patch EEG Transformer | 1 | 5 | 72.59% | 63.33% | 0.7052 | 62.22% | 64.44% | 40.00% | 51.11% |
| Temporal-Patch EEG Transformer | 2 | 31 | 100.00% | 67.78% | 1.3139 | 66.67% | 68.89% | 51.11% | 48.89% |

## Trial accounting

Part 1 has 45 task annotations per subject but retains 42: its full -2 to 5 second epoch drops two boundary-crossing events and one too-short event. The classifier then uses 0–4 seconds. Part 2 directly epochs 0–3 seconds and retains all 45 events per subject, without amplitude rejection. This accounts for 420 versus 450 trials; it is not a label-count inconsistency.

| Subject | Part 1 retained (fists/feet) | Part 1 dropped | Part 2 retained (fists/feet) | Part 2 dropped |
|---|---:|---:|---:|---:|
| 1 | 42 (21/21) | 3 | 45 (21/24) | 0 |
| 2 | 42 (21/21) | 3 | 45 (24/21) | 0 |
| 3 | 42 (21/21) | 3 | 45 (21/24) | 0 |
| 4 | 42 (21/21) | 3 | 45 (22/23) | 0 |
| 5 | 42 (21/21) | 3 | 45 (23/22) | 0 |
| 6 | 42 (21/21) | 3 | 45 (22/23) | 0 |
| 7 | 42 (21/21) | 3 | 45 (22/23) | 0 |
| 8 | 42 (21/21) | 3 | 45 (23/22) | 0 |
| 9 | 42 (21/21) | 3 | 45 (23/22) | 0 |
| 10 | 42 (21/21) | 3 | 45 (23/22) | 0 |

## Generalization evidence

Vanilla seeds 0/1 end with online training losses 0.090/0.100 versus validation losses 1.533/1.764: a strong overfitting signal. Validation-to-test differences suggest subject-dependent generalization, but only two people in each cohort cannot identify a population-level cause. A suboptimal learning rate is unverified; no baseline hyperparameters were changed. Historical EEGNet validation predictions were not saved, so we cannot determine retrospectively whether S7 or S8 dominated its validation score. Newly saved Temporal-Patch EEG Transformer validation predictions permit that descriptive check without retraining. No test-based preprocessing or architecture selection was performed. Optional compact models, robust scaling and refits were skipped under the submission deadline.

Across the three Temporal-Patch EEG Transformer runs, S7/S8 validation accuracy differs by only 2.22 percentage points within each seed, and the easier subject changes. Neither validation subject consistently dominates. Saved predictions show mild class preferences, not single-class collapse. All three Temporal-Patch EEG Transformer runs develop substantial late train/validation loss gaps. The patch model does not exceed EEGNet in mean test accuracy; the proposed accuracy benefit is not established. Reduced attention-score count is demonstrated structurally, while its causal contribution to generalization is not isolated by this unequal-parameter comparison.

### Defense questions

Scaled dot-product attention is softmax(QKᵀ/√dₖ)V. Dividing by √dₖ controls the variance of dot products as key dimension grows, helping avoid saturated softmax. With 80 training subjects, greater subject diversity could improve learned invariance and make vanilla attention more competitive; this is a hypothesis, not a result. Patching would still reduce the attention matrix size, but any generalization advantage would need a new subject-held-out comparison with matched capacity and untouched evaluation subjects. Attention weights alone do not establish causal physiological relevance.

### Saved validation-subject predictions

Balanced accuracy is the mean of fists and feet recall. Prediction counts use fists/feet order; these are descriptive development diagnostics only.

| Model | Seed | Subject | Balanced accuracy | Fists recall | Feet recall | Predicted fists/feet |
|---|---:|---:|---:|---:|---:|---:|
| Temporal-Patch EEG Transformer | 0 | 7 | 64.43% | 63.64% | 65.22% | 22/23 |
| Temporal-Patch EEG Transformer | 0 | 8 | 62.35% | 56.52% | 68.18% | 20/25 |
| Temporal-Patch EEG Transformer | 1 | 7 | 62.45% | 72.73% | 52.17% | 27/18 |
| Temporal-Patch EEG Transformer | 1 | 8 | 64.23% | 73.91% | 54.55% | 27/18 |
| Temporal-Patch EEG Transformer | 2 | 7 | 66.50% | 59.09% | 73.91% | 19/26 |
| Temporal-Patch EEG Transformer | 2 | 8 | 68.97% | 65.22% | 72.73% | 21/24 |
| VanillaEEGTransformer | 2 | 7 | 50.00% | 0.00% | 100.00% | 0/45 |
| VanillaEEGTransformer | 2 | 8 | 50.00% | 0.00% | 100.00% | 0/45 |

The brief development-only amplitude check found a median channel standard-deviation / IQR-derived scale ratio of 1.32 (maximum 1.87). Subjects 1 and 3 contribute disproportionately to training variance (about 30% and 35% averaged across channels). This shows amplitude heterogeneity, not a confirmed artifact or scaling defect. No robust-scaling experiment, rejection or normalization change was performed.
