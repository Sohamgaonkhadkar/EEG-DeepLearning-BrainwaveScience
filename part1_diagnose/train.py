"""Train and evaluate a hands-vs-feet decoder on unseen subjects.

Usage:
  python train.py --channels all --seed 0
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from data import build_dataset, SUBJECTS
from eeg_quality import standardize_training, subject_split
from model import EEGNet, count_params

# Electrode groups for the channel-subset run (Task 3).
# Predeclared frontopolar and frontal subset, checked against loaded names.
CHANNEL_GROUPS = {
    "all": None,
    "frontal": ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F3", "Fz", "F4"],
}


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def evaluate(model, X, y, bs=256):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(X[i:i + bs]); yb = torch.from_numpy(y[i:i + bs])
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=list(CHANNEL_GROUPS), default="all")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.epochs < 1 or args.bs < 1:
        ap.error("epochs and bs must be positive")
    set_seed(args.seed)

    X, y, groups, ch_names = build_dataset(args.subjects)
    picks = CHANNEL_GROUPS[args.channels]
    if picks is not None:
        if not picks:
            raise ValueError(f"Fill in CHANNEL_GROUPS['{args.channels}'] first.")
        X = X[:, [ch_names.index(c) for c in picks]]
    print(f"data: {X.shape}, subjects={len(np.unique(groups))}, "
          f"class balance={np.bincount(y)}")

    tr, va, te = subject_split(groups)
    X[tr], X[va], X[te] = standardize_training(X[tr], X[va], X[te])

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_ch, n_t)
    print(f"EEGNet: {count_params(model)} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    best_acc, best_state, best_ep = -1.0, None, -1
    history = []
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        total_loss = 0.0
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b])
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            total_loss += loss.item() * len(b)
        va_acc = evaluate(model, X[va], y[va])
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(torch.from_numpy(X[va])), torch.from_numpy(y[va])).item()
        history.append(dict(epoch=ep, train_loss=total_loss/len(tr),
                            validation_loss=val_loss, validation_acc=va_acc))
        if va_acc > best_acc:   # select without inspecting the held-out test
            best_acc, best_ep = va_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val {va_acc:.3f}")

    model.load_state_dict(best_state)
    result = dict(channels=args.channels, seed=args.seed, best_epoch=best_ep,
                  n_test=len(te),
                  protocol="subject_disjoint_validation", epochs=args.epochs,
                  lr=args.lr, batch_size=args.bs, n_params=count_params(model),
                  train_subjects=np.unique(groups[tr]).tolist(),
                  validation_subjects=np.unique(groups[va]).tolist(),
                  test_subjects=np.unique(groups[te]).tolist(),
                  n_train=len(tr), n_validation=len(va),
                  best_validation_acc=best_acc,
                  selected_channels=picks or ch_names,
                  history=history, test_predictions=[], test_labels=y[te].tolist())
    model.eval()
    with torch.no_grad():
        for i in range(0, len(te), 64):
            result["test_predictions"].extend(model(torch.from_numpy(X[te[i:i+64]])).argmax(1).tolist())
    result["test_acc"] = float(np.mean(np.array(result["test_predictions"]) == y[te]))
    from sklearn.metrics import confusion_matrix, classification_report
    result["confusion_matrix"] = confusion_matrix(result["test_labels"], result["test_predictions"], labels=[0,1]).tolist()
    result["per_class"] = classification_report(result["test_labels"], result["test_predictions"], labels=[0,1], target_names=["fists","feet"], output_dict=True, zero_division=0)
    result["per_subject_accuracy"] = {
        str(subject): float(np.mean(np.array(result["test_predictions"])[groups[te]==subject] == y[te][groups[te]==subject]))
        for subject in np.unique(groups[te])}
    print(json.dumps(result))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
