# Repository publication

Target: https://github.com/Sohamgaonkhadkar/EEG-DeepLearning-BrainwaveScience

From the repository root, inspect remote access before pushing the preserved history:

```powershell
git ls-remote https://github.com/Sohamgaonkhadkar/EEG-DeepLearning-BrainwaveScience.git
git remote add origin https://github.com/Sohamgaonkhadkar/EEG-DeepLearning-BrainwaveScience.git
git push -u origin master
```

If origin already exists, inspect it before changing it. Do not force-push. Local checkpoints are excluded from Git; checkpoint_manifest.json records their hashes.
