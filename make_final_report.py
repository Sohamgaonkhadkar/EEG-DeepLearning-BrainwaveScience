"""Create the requested A-J local completion report after successful verification."""
from pathlib import Path
import json
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parent
load=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def summary(rows):
    lines=[]
    for name in ['EEGNet','VanillaEEGTransformer','YourModel']:
        a=np.array([r['test_acc']*100 for r in rows if r['model']==name])
        assert len(a)==3
        lines.append(f'| {name} | '+ ' | '.join(f'{v:.2f}%' for v in a)+f' | {a.mean():.2f}% | {a.std():.2f} pp | {a.min():.2f}% | {a.max():.2f}% |')
    return '\n'.join(lines)


def main():
    verification=json.loads((ROOT/'verification.json').read_text())
    assert verification['passed']
    original=load(ROOT/'part1_diagnose/original_results.jsonl')[-1]
    p1=load(ROOT/'part1_diagnose/results.jsonl')
    p1refit=load(ROOT/'refits/part1_diagnose/results.jsonl')
    p2=load(ROOT/'part2_transformer/results.jsonl')
    p2refit=load(ROOT/'refits/part2_transformer/results.jsonl')
    part1rows='\n'.join(f"| Development {r['channels']} | {r['n_train']} | {r['n_test']} | {100*r['test_acc']:.2f}% |" for r in p1)
    part1rows+='\n'+'\n'.join(f"| Eight-subject refit {r['model']} | {r['n_train']} | {r['n_test']} | {100*r['test_acc']:.2f}% |" for r in p1refit)
    history=subprocess.check_output(['git','log','--oneline'],cwd=ROOT,text=True)
    (ROOT/'COMMIT_HISTORY.txt').write_text(history,encoding='utf-8')
    plan_commit=subprocess.check_output(['git','log','-1','--format=%h','--','refit_plan.json'],cwd=ROOT,text=True).strip()
    text=f'''# Final local completion report

## A. Completion Status

Part 1 and Part 2 are complete locally, including the required original/corrected experiments, all nine 40-epoch development runs, inspected figures, answers and quality audit. The additional eight-subject refit experiment is complete and reported separately; it did not replace or restart completed development runs.

## B. Part 1 Findings

| Location | Confirmed incorrect behavior → correction | Separate commit |
|---|---|---|
| data.py / RUNS | Executed 5/9/13 → imagery 6/10/14 | 11f4b6f |
| train.py / first split | Trial-level train/test mixing → held-out people 9/10 | 8c1dca8 |
| train.py / second split | Same people in train/validation → validation people 7/8 | c4ede07 |
| train.py / channel scaling | All-data statistics → optimization-only statistics | 24c4e0d |
| train.py / checkpoint choice | Maximum test accuracy → earliest maximum validation accuracy | ded795d |
| erd.py / BASELINE | Post-cue 0–0.5 s → pre-cue -1.5 to -0.5 s | 3fe8efd |

The two subject-leakage paths are explicitly counted separately; the PDF supplies no answer key. MNE runtime compatibility fixes are additional engineering work, not invented intentional defects.

## C. Part 1 Experimental Results

The untouched original reports **{100*original['test_acc']:.2f}%** on {original['n_test']} test trials. It uses executed movements and a trial-level split with test-selected weights, so its score and the corrected imagery/unseen-person score estimate different targets.

| Corrected configuration | Train trials | Test trials | Test accuracy |
|---|---:|---:|---:|
{part1rows}

## D. Part 2 Model

Deliberately validate and complete the supplied learned temporal-patch tokenizer: Conv1d kernel/stride 32, right padding to include the final sample, then unchanged-size transformer blocks. With 481 samples there are 16 patches plus CLS (17 tokens), versus vanilla's 482 tokens including CLS; each attention score matrix is 17×17 rather than 482×482, an approximately 804× score-count reduction. This does not imply the same wall-clock speedup or fewer parameters; the patch model has 232,962 parameters versus vanilla's 135,746. The supplied idea is acknowledged rather than claimed as an original invention.

## E. Model Comparison

Development uses optimization subjects 1–6, validation 7–8, test 9–10; every model runs seeds 0/1/2 for 40 epochs and restores its validation-selected checkpoint.

| Model | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD | Worst | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
{summary(p2)}

The separate refit initializes new weights and fits normalization on people 1–8. Each model uses one fixed duration derived as the median validation-selected epoch count across the three development seeds; the exact plan was committed as `{plan_commit}` before training. All eleven refit models finish before refit test scoring begins.

| Refit model | Seed 0 | Seed 1 | Seed 2 | Mean | Population SD | Worst | Best |
|---|---:|---:|---:|---:|---:|---:|---:|
{summary(p2refit)}

See [ANSWERS.md](ANSWERS.md) for interpretation, actual ERD/t-SNE observations and methodological limits. No stage, seed, or checkpoint is selected using test scores.

## F. Generated Artifacts

- [ANSWERS.md](ANSWERS.md), [WALKTHROUGH_NOTES.md](WALKTHROUGH_NOTES.md), [TRAINING_STRATEGY.md](TRAINING_STRATEGY.md), [refit_plan.json](refit_plan.json).
- [Original result](part1_diagnose/original_results.jsonl), [Part 1 results](part1_diagnose/results.jsonl), [mu ERD](part1_diagnose/erd_mu.png), [beta ERD](part1_diagnose/erd_beta.png) and numerical ERD summaries.
- [Part 2 results](part2_transformer/results.jsonl), [t-SNE figure](part2_transformer/figure.png) and labeled embedding coordinates.
- [Part 1 refits](refits/part1_diagnose/results.jsonl), [Part 2 refits](refits/part2_transformer/results.jsonl), per-subject data audits, [transcripts](logs/) and [verification.json](verification.json).

## G. Verification

Successfully executed: untouched `python train.py --channels all --seed 0`; corrected all/frontal training; `python erd.py --band mu` and `--band beta`; `python main.py --epochs 40 --seeds 0 1 2`; validation-derived eight-subject refits; Python compilation; eight integrity tests; final provenance/result audit. `verification.json` confirms all 60 EDF SHA256 hashes, exact original starter preservation, disjoint actual subject/sample sets, recomputed accuracies/confusion matrices, validation selection, fixed refit durations and valid required PNGs. Baseline model source files remain unchanged.

Resolved attempts are retained: MNE cache-path prompt in noninteractive original execution; MNE 1.12 incompatibility with a bare NumPy ufunc; sandbox default Temp permissions in a new test. The Conda launcher could not execute, so the working isolated local Python 3.10.18 environment uses the available CPU PyTorch installation plus locally installed MNE; exact versions are recorded. No missing experiment is hidden or represented by a synthetic result.

## H. Git History

Initial untouched source: `b76b586`. Hypothesis/plan before implementation: `c9c5bc2`. Six correction commits are listed above; Part 1 outputs: `1f98274`; separate refit strategy: `82c1583`; frozen numeric refit plan: `{plan_commit}`. [COMMIT_HISTORY.txt](COMMIT_HISTORY.txt) records history through verification; subsequent final artifact commits remain in `git log`. No history was squashed or rewritten.

## I. Outstanding Problems and Limits

No required execution remains incomplete. Residual artifacts and top/bottom cue confounding remain scientific limitations; frontal accuracy alone does not identify their source. Two test people and three random seeds cannot establish population-level superiority. Subjects 9–10 had already been scored in required development comparisons before the additional refit request, so the follow-up is not a fresh untouched external test; its decisions nevertheless use validation only and all results are reported. The examiner's intended grouping of the six defects is not independently available.

## J. GitHub Status

**The project is complete locally. No files have been pushed to GitHub. Awaiting the new repository URL.**
'''
    (ROOT/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    print('Created FINAL_REPORT.md after successful verification')

if __name__=='__main__': main()
