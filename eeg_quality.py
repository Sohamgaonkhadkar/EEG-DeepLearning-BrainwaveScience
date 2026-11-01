"""Non-adaptive data audits: report amplitudes, reject invalid data, never tune on test."""
import hashlib
import json
import numpy as np
from pathlib import Path


def validate_subject(X, y, channels, subject, sfreq, runs, units, times):
    if X.ndim != 3 or X.shape[1] != 64 or len(channels) != 64:
        raise ValueError(f"S{subject}: expected 64-channel trials")
    if len(set(channels)) != 64 or not {"C3", "Cz", "C4"}.issubset(channels):
        raise ValueError(f"S{subject}: invalid channel names")
    if sfreq != 160 or len(X) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"S{subject}: invalid sampling rate or labels")
    if not np.isfinite(X).all():
        raise ValueError(f"S{subject}: non-finite samples; inspect source before training")
    # Diagnostics only: no thresholds are learned and no test-driven cleaning.
    uv = X * (1e6 if units == "V" else 1)
    peak = np.max(np.abs(uv), axis=(1, 2))
    flat = np.std(uv, axis=2) < 1e-9
    if flat.any():
        raise ValueError(f"S{subject}: flat channel/epoch; inspect source")
    hashes = [hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in X]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"S{subject}: duplicated epochs")
    return dict(subject=int(subject), sfreq=float(sfreq), runs=list(runs),
                units=units, shape=list(X.shape), channels=channels,
                class_counts=np.bincount(y, minlength=2).tolist(),
                start_end_seconds=[float(times[0]), float(times[-1])],
                absolute_uv_percentiles=np.percentile(np.abs(uv), [50,95,99,100]).tolist(),
                epoch_peak_uv_percentiles=np.percentile(peak,[50,95,100]).tolist(),
                epochs_peak_over_500uv=int((peak > 500).sum()),
                finite=True, duplicate_epochs=0, epoch_hashes=hashes)


def save_audit(records, path):
    channels = records[0]["channels"]
    seen = set()
    for r in records:
        if r["channels"] != channels:
            raise ValueError("Electrode order differs across subjects")
        if seen.intersection(r["epoch_hashes"]):
            raise ValueError("Duplicate epoch crosses subject boundary")
        seen.update(r["epoch_hashes"])
    Path(path).write_text(json.dumps(records, indent=2)+"\n")
