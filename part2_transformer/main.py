"""
EEG Transformer Sprint — entry point.
Run:  python main.py

This script:
  1. loads subjects 1-6 for training, 7-8 validation, 9-10 test,
  2. trains the two baselines (EEGNet, vanilla transformer),
  3. trains YOUR model across a few seeds and reports best + worst,
  4. saves one figure (figure.png).

>>> YOUR JOB <<<
Implement ONE inductive bias in `YourModel` below (or replace it entirely).
State your hypothesis in the README. Do not quietly edit the vanilla
transformer in models.py — the contrast is the point.
"""
import argparse
import json
import time
import math
from pathlib import Path
import numpy as np
import torch

from data import load_validation_data
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed, evaluate, make_loaders


# =========================================================================== #
#  YOUR MODEL  — the only part you must write.
#
#  The starter version below is the vanilla transformer with learned TEMPORAL
#  PATCH tokens instead of per-timestep tokens: one concrete, legal example of
#  "inject one inductive bias" (fewer tokens -> a smaller O(T^2) surface to
#  overfit on 8 subjects). Replace its guts with your own idea. Some directions:
#    - structured positional encoding from electrode geometry (get_channel_positions)
#      and/or frequency band,
#    - an attention bias toward scalp neighbourhoods or band-limited windows,
#    - channel (spatial) tokens instead of / in addition to time tokens,
#    - regularisation aimed specifically at the 8-subject data wall.
# =========================================================================== #
import torch.nn as nn


class YourModel(nn.Module):
    def __init__(self, n_chan, n_time, n_classes=2,
                 patch=32, d_model=64, n_heads=4, depth=3, p=0.3):
        super().__init__()
        # Learned temporal patches: a Conv1d with stride == kernel tokenises
        # the signal into ceil(n_time/patch) tokens, each summarising `patch` samples
        # across all electrodes. This is the ONE bias in the starter example.
        if patch < 1 or n_time < 1 or n_chan < 1:
            raise ValueError("Input dimensions and patch must be positive")
        self.n_time, self.patch = n_time, patch
        self.tokenizer = nn.Conv1d(n_chan, d_model, kernel_size=patch,
                                   stride=patch)
        n_tokens = math.ceil(n_time / patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            dropout=p, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):   # x: (B, C, T)
        if x.ndim != 3 or x.shape[2] != self.n_time:
            raise ValueError(f"Expected (batch, channels, {self.n_time})")
        # Preserve the last partial patch instead of silently dropping samples.
        x = torch.nn.functional.pad(x, (0, (-self.n_time) % self.patch))
        tok = self.tokenizer(x).transpose(1, 2)  # (B, n_tokens, d)
        cls = self.cls.expand(x.size(0), -1, -1)
        z = torch.cat([cls, tok], dim=1) + self.pos
        z = self.encoder(z)
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


# =========================================================================== #


def save_figure(model, Xte, yte, device, path="figure.png"):
    """t-SNE of the CLS embeddings on the test subjects, coloured by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        emb = np.concatenate([model(torch.from_numpy(Xte[i:i+16]).to(device),
                                            return_embed=True)[1].cpu().numpy()
                              for i in range(0, len(Xte), 16)])
    n = len(emb)
    emb2 = TSNE(n_components=2, perplexity=min(30, max(5, n // 4)),
                init="pca", random_state=0).fit_transform(emb)
    from pathlib import Path
    audit = json.loads(Path("data_audit.json").read_text())
    subject_ids = np.array([9]*audit[8]["shape"][0]+[10]*audit[9]["shape"][0])
    with open("embedding_coordinates.json", "w") as f:
        json.dump(dict(method="t-SNE", random_state=0, labels=yte.tolist(),
                       subject_ids=subject_ids.tolist(), coordinates=emb2.tolist(),
                       n_samples=n, checkpoint="validation-selected first requested seed"), f)
    plt.figure(figsize=(6, 4.5))
    for c, name, color in [(0, "fists", "tab:blue"), (1, "feet", "tab:orange")]:
        for subject, marker in [(9,"o"),(10,"^")]:
            m = (yte == c) & (subject_ids == subject)
            plt.scatter(emb2[m, 0], emb2[m, 1], s=24, alpha=0.75,
                        c=color, marker=marker, label=f"{name}, S{subject:02d}")
    plt.legend(); plt.title("Temporal-Patch EEG Transformer\nTest-subject embeddings (seed 0)")
    plt.xlabel('t-SNE dimension 1'); plt.ylabel('t-SNE dimension 2')
    plt.tight_layout(); plt.savefig(path, dpi=130)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--deadline-guard", action="store_true",
                    help="Apply the local execution deadline to a queued vanilla restart")
    ap.add_argument("--models", nargs="+", default=["EEGNet", "VanillaEEGTransformer", "YourModel"],
                    choices=["EEGNet", "VanillaEEGTransformer", "YourModel"])
    args = ap.parse_args()
    # Local deadline guard applies only to the queued expensive restart.
    deadline_file = Path('../execution_deadline.json')
    if args.deadline_guard and args.models == ['VanillaEEGTransformer'] and deadline_file.exists():
        from datetime import datetime, timezone
        deadline = datetime.fromisoformat(json.loads(deadline_file.read_text())['deadline_utc'])
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining < 28 * 60:
            print('Vanilla seed restart skipped: reserve submission time; insufficient runtime budget.', flush=True)
            return
    torch.set_num_threads(args.threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("\nloading data (first run downloads ~50 MB)...")
    Xtr, ytr, Xva, yva, Xte, yte = load_validation_data()
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"train {Xtr.shape}  test {Xte.shape}  "
          f"(class balance train {ytr.mean():.2f} / test {yte.mean():.2f})")

    if args.epochs < 1:
        ap.error("epochs must be positive")
    records = []
    figure_model = None
    builders = {"EEGNet": lambda: EEGNet(C, T, kern=80, p=0.5),
                "VanillaEEGTransformer": lambda: VanillaEEGTransformer(C, T),
                "YourModel": lambda: YourModel(C, T)}
    completed = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()] if Path(args.results).exists() else []
    for name, build in builders.items():
        if name not in args.models:
            continue
        for seed in args.seeds:
            if any(r['model'] == name and r['seed'] == seed for r in completed):
                print(f'Skipping preserved completed run {name} seed {seed}', flush=True)
                continue
            set_seed(seed)
            model = build()
            checkpoint_path = Path('checkpoints') / f'{name}_seed{seed}.pt'
            if checkpoint_path.exists():
                raise FileExistsError(f'Refusing to overwrite existing checkpoint {checkpoint_path}')
            metadata = dict(model=name, seed=seed, epochs=args.epochs,
                configuration=repr(model), n_params=count_params(model),
                train_subjects=list(range(1,7)), validation_subjects=[7,8], test_subjects=[9,10],
                preprocessing=dict(runs=[6,10,14], band_hz=[7,30], interval_seconds=[0,3],
                    baseline=None, units='volts', normalization='train-only per-channel mean/std + 1e-6'),
                lr=.001, weight_decay=.001, batch_size=16)
            started = time.perf_counter()
            val_acc, history = train_model(model, Xtr, ytr, Xva, yva,
                                 epochs=args.epochs, device=device, verbose=True,
                                 checkpoint_path=checkpoint_path, checkpoint_metadata=metadata)
            _, validation_loader = make_loaders(Xtr, ytr, Xva, yva)
            _, _, validation_preds, _ = evaluate(model, validation_loader, device, True)
            _, test_loader = make_loaders(Xtr, ytr, Xte, yte)
            acc, test_loss, preds, labels = evaluate(model, test_loader, device, True)
            record = dict(model=name, seed=seed, epochs=args.epochs,
                          n_params=count_params(model), test_acc=acc,
                          n_train=len(ytr), n_test=len(yte),
                          train_subjects=list(range(1, 7)), validation_subjects=[7, 8],
                          training_pool=list(range(1, 9)), test_subjects=[9, 10],
                          n_validation=len(yva), best_validation_acc=val_acc,
                          selected_epoch=model.selected_epoch, history=history,
                          test_loss=test_loss,
                          lr=1e-3, weight_decay=1e-3, batch_size=16,
                          checkpoint="maximum_validation_accuracy_earliest_tie", device=device,
                          seconds=time.perf_counter()-started,
                          input_shape=[C, T], test_predictions=preds,
                          test_labels=yte.tolist())
            record.update(checkpoint_file=str(checkpoint_path), checkpoint_reload_verified=True,
                          validation_predictions=validation_preds, validation_labels=yva.tolist(),
                          configuration=metadata)
            from sklearn.metrics import confusion_matrix, classification_report
            record["confusion_matrix"] = confusion_matrix(yte, preds, labels=[0,1]).tolist()
            record["per_class"] = classification_report(yte, preds, labels=[0,1], target_names=["fists","feet"], output_dict=True, zero_division=0)
            audit = json.loads(Path("data_audit.json").read_text())
            n9 = audit[8]["shape"][0]
            record["per_subject_accuracy"] = {
                "9": float(np.mean(np.array(preds[:n9]) == yte[:n9])),
                "10": float(np.mean(np.array(preds[n9:]) == yte[n9:]))}
            print(json.dumps(record), flush=True)
            records.append(record)
            with open(args.results, "a") as f:
                f.write(json.dumps(record)+"\n")
                f.flush()
            Path('run_records').mkdir(exist_ok=True)
            Path(f'run_records/{name}_seed{seed}.json').write_text(json.dumps(record, indent=2))
            # Preselect the first requested seed, irrespective of its accuracy.
            if name == "YourModel" and seed == 0:
                figure_model = model
                save_figure(figure_model, Xte, yte, device)
        accs = np.array([r["test_acc"] for r in records if r["model"] == name])
        if not len(accs):
            continue
        print(f"{name}: best {accs.max():.4f}, worst {accs.min():.4f}, "
              f"mean {accs.mean():.4f}, population std {accs.std():.4f}", flush=True)


if __name__ == "__main__":
    main()
