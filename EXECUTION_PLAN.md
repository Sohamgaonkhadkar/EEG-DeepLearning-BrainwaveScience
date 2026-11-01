# Execution plan

1. Preserve untouched starter (commit b76b586), read all five PDF pages and source files.
2. Install missing MNE/PDF dependencies into workspace venv using available PyTorch environment.
3. Run unmodified Part 1 (30 epochs, all channels, seed 0); preserve transcript and JSON result.
4. Correct evidence-backed scientific defects in separate commits: run selection, test subject split, validation subject split, training-only normalization, validation checkpoint selection, pre-cue ERD baseline. The PDF says six but supplies no answer key; the two subject leakage paths are reported separately and transparently.
5. Run corrected all/frontal experiments under identical settings; generate mu ERD and inspect it.
6. Part 2: fixed 40-epoch final checkpoint for all three models, subjects 1-8 training / 9-10 test, seeds 0/1/2. No model selection on test subjects.
7. Implement and verify temporal patching, retain unchanged vanilla baseline, generate t-SNE using a preselected seed 0 model.
8. Check data invariants, model forward/backward, leakage protection, numerical ERD, plots and real results; write concise answers and walkthrough; commit all required artifacts locally.

Scientific risks: few held-out people, cue/eye confounding, zero-phase filtering and ERD baseline smearing; accuracy changes do not isolate individual fixes. No synthetic measurement will be reported as EEG performance.

## Hypothesis recorded before implementation
Learned 32-sample temporal patches impose local aggregation and reduce pairwise attention cost; on eight training subjects they may improve stability relative to per-sample attention, without guaranteeing superiority to EEGNet. The starter already contains this example: the contribution will be explicit validation, complete final-patch coverage and an unbiased reproducible comparison, rather than a claim of inventing the patch tokenizer.
