# Manual submission

Project: `C:\Users\Soham\Documents\Basil Health assignment-soham-gaonkhadkar\eeg_dl_exam_3hr`

No remote is configured and nothing has been pushed. Once finalization finishes,
`verification.json` should say `passed: true` and `comparison_complete: true`.
The finalizer only builds reports, verifies artifacts and commits locally; it
never trains or pushes. If necessary, rerun it after training completes:

```powershell
Set-Location 'C:\Users\Soham\Documents\Basil Health assignment-soham-gaonkhadkar\eeg_dl_exam_3hr'
..\.venv\Scripts\python.exe finalize_submission.py
git status --short
Get-Content verification.json
```

Create an empty GitHub repository, then replace the URL below and run:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin master
```

If `origin` already exists, use `git remote set-url origin URL` instead of
`git remote add`. Do not force-push or squash: the six separate bug-fix commits
must remain. `ANSWERS.md` contains the final results and explanation.

The four newly trained selected checkpoints are local files under
`part2_transformer\checkpoints`; `checkpoint_manifest.json` records their hashes.
They are intentionally excluded from Git, as are raw EEG recordings and the
environment. Keep this folder locally and attach checkpoints separately if the
assessor requests weights. Historical completed runs have no saved weights.
