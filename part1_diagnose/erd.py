"""Event-related desynchronisation (ERD) for hands vs feet imagery.

ERD% = (P(t) - P_baseline) / P_baseline * 100, where P(t) is band power
(squared Hilbert envelope) averaged over trials. Negative = desynchronisation.

Usage: python erd.py --subjects 1 2 3
"""
import argparse
import json
import numpy as np
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import load_raw, get_events, EVENT_ID, TMIN, TMAX, SUBJECTS

BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}
BASELINE = (-1.5, -0.5)   # rest period before the imagery starts (s)
PICKS = ["C3", "Cz", "C4"]


def band_power_epochs(raw, events, band):
    """Band-pass the continuous signal (avoids edge effects), then epoch the power."""
    r = raw.copy().filter(*band, fir_design="firwin", verbose="error")
    r.apply_hilbert(envelope=True, verbose="error")
    epochs = mne.Epochs(r, events, EVENT_ID, TMIN, TMAX, baseline=None,
                        preload=True, verbose="error")
    return epochs.apply_function(np.square)   # envelope -> power


def erd_percent(epochs, cond):
    p = epochs[cond].get_data(copy=True).mean(axis=0)          # (ch, time)
    t = epochs.times
    b = (t >= BASELINE[0]) & (t < BASELINE[1])
    if not b.any():
        raise ValueError("Baseline does not overlap the epoch")
    ref = p[:, b].mean(axis=1, keepdims=True)
    if np.any(ref <= 0) or not np.isfinite(ref).all():
        raise ValueError("Baseline power must be finite and positive")
    return (p - ref) / ref * 100.0, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    ap.add_argument("--band", choices=list(BANDS), default="mu")
    args = ap.parse_args()

    curves = {c: [] for c in EVENT_ID}
    for s in args.subjects:
        raw = load_raw(s)
        events = get_events(raw)
        ep = band_power_epochs(raw, events, BANDS[args.band])
        for c in EVENT_ID:
            erd, t = erd_percent(ep, c)
            curves[c].append(erd[[ep.ch_names.index(ch) for ch in PICKS]])

    np.savez_compressed(f"erd_{args.band}_curves.npz", times=t,
                        **{c: np.array(curves[c]) for c in EVENT_ID})
    summary = {}
    task_mask = (t >= 0) & (t <= 4)
    for condition, subject_curves in curves.items():
        summary[condition] = {ch: float(np.mean(np.array(subject_curves)[:, i, task_mask]))
                              for i, ch in enumerate(PICKS)}
    with open(f"erd_{args.band}_summary.json", "w") as f:
        json.dump(dict(band=args.band, subjects=args.subjects, baseline=BASELINE,
                       imagery_window=[0,4], mean_erd_percent=summary), f, indent=2)
    fig, axes = plt.subplots(1, len(PICKS), figsize=(12, 3.5), sharey=True)
    for i, ch in enumerate(PICKS):
        for c in EVENT_ID:
            axes[i].plot(t, np.mean(curves[c], axis=0)[i], label=c)
        axes[i].axvspan(*BASELINE, color="gray", alpha=0.15)
        axes[i].axvline(0, color="k", lw=0.8); axes[i].axhline(0, color="k", lw=0.5)
        axes[i].set_title(f"{ch} {args.band}"); axes[i].set_xlabel("time from cue (s)")
    axes[0].set_ylabel("ERD/ERS (%)"); axes[0].legend()
    fig.tight_layout(); fig.savefig(f"erd_{args.band}.png", dpi=120)
    print(f"saved erd_{args.band}.png")


if __name__ == "__main__":
    main()
