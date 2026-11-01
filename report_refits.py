"""Separate reporting for the user-requested eight-subject refit experiment."""
import json
import numpy as np


def refit_report(root):
    plan=json.loads((root/'refit_plan.json').read_text())
    load=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    p1=load(root/'refits/part1_diagnose/results.jsonl')
    p2=load(root/'refits/part2_transformer/results.jsonl')
    assert len(p1)==2 and len(p2)==9
    p1rows='\n'.join(f"| {r['model']} | {r['epochs']} | {r['n_train']} | {100*r['test_acc']:.2f}% | {r['n_test']} |" for r in p1)
    rows=[]
    for name in ['EEGNet','VanillaEEGTransformer','YourModel']:
        records=sorted([r for r in p2 if r['model']==name],key=lambda r:r['seed'])
        values=np.array([r['test_acc']*100 for r in records])
        assert [r['seed'] for r in records]==[0,1,2]
        rows.append(f"| {name} | {plan['part2'][name]['epochs']} | "+' | '.join(f'{v:.2f}%' for v in values)+f" | {values.mean():.2f}% | {values.std():.2f} pp | {values.min():.2f}% | {values.max():.2f}% |")
    return f'''

## Additional Final Refits on Subjects 1–8

These are separately reported follow-up measurements requested after development scoring began; none replaces the required six-subject optimization / two-subject validation results above. The architectures, learning rate, regularization and batch sizes stay fixed. For each Part 2 model, the common refit duration is the median of its three earliest validation-best epoch counts; Part 1 uses its single existing validation-selected count. `refit_plan.json` was committed before refitting, and a test-score perturbation test verifies that its durations do not depend on recorded development test accuracy.

All models were reinitialized with their declared seeds and normalization was refitted on subjects 1–8 only. The final predetermined epoch was used, with all eleven model states frozen before loading subjects 9–10 for this scoring stage; there is no test-based checkpoint or stage selection. This restores 336 Part 1 or 360 Part 2 training trials while retaining every official test trial.

### Part 1 Refit Results

| Channel group | Fixed epochs | Refit trials | Test accuracy | Test trials |
|---|---:|---:|---:|---:|
{p1rows}

### Part 2 Refit Results

| Model | Fixed epochs | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD | Worst | Best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The median-duration rule and the underlying validation epochs are recorded in the frozen plan; all predictions, confusion matrices, per-class/per-subject metrics and training histories are in `refits/<part>/results.jsonl`. More training people can help, but a higher score is not guaranteed and whichever stage scores higher is not selected for reporting. Because subjects 9–10 had already been scored in the required comparisons, these are not fresh untouched external-test results; the refit decisions use validation alone, and an independent new cohort would be needed for a new confirmatory claim. `TRAINING_STRATEGY.md` explains the full chronology and independence limits.
'''
