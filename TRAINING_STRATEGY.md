# Development and final refit strategy

## Preserved development stage

Subjects 1-6 optimize weights; 7-8 validate checkpoints. The original required
Part 1 all/frontal and Part 2 three-model/three-seed runs remain unchanged.
Normalization is fitted only on 1-6. Earliest maximum validation accuracy picks
the checkpoint. Part 1 uses 252/84/84 trials, and Part 2 uses 270/90/90 because
its shorter 0-3 s epoch retains all 45 trials per subject instead of 42.

## Separate final refit stage

Freeze architecture, learning rate, optimizer, regularization and batch sizes.
For each Part 2 model, take the median of the three earliest validation-best
epoch counts. Train all three seeds for that same model-specific duration.
The common rule prevents test-driven duration choices and reduces sensitivity
to one unusually early or late validation peak. Part 1 has only seed 0, so its
all/frontal refits use their respective selected epoch counts directly.

Commit the exact `refit_plan.json` before training. Reinitialize weights using
each declared seed, refit normalization on all subjects 1-8, and optimize for
exactly the planned duration. This provides 336 Part 1 trials and 360 Part 2
trials. Preserve kernel/dropout and other architecture settings; do not choose
which models or seeds to report using their test outcomes.

All eleven refits (two Part 1 configurations and nine Part 2 model/seed runs)
finish before test data are loaded for this stage. The final predetermined
epoch is evaluated once on the unchanged subjects 9-10: 84 Part 1 / 90 Part 2
trials. Store every result in `refits/<part>/results.jsonl`, with training loss
history, exact duration, confusion matrix, per-class and per-subject metrics.

## Interpretation and independence

The official test cohort has already been scored for the required development
comparisons before this additional request. That cannot be undone. The refit
plan uses validation histories only, and a test-score perturbation test checks
that refit durations cannot depend on recorded development test accuracy.
Report all development and refit results separately; never pick whichever
stage scored highest. The refit experiment is a prespecified follow-up on a
previously evaluated cohort, not a fresh external confirmation of superiority.
A new subject cohort is needed for such confirmation.

## Execution status

All nine development runs are complete. No eight-subject refits were executed. The refit stage above is a proposal.
