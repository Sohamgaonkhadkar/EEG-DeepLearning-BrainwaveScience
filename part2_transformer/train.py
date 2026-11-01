"""Validation-selected training and a separately invoked held-out evaluation."""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def make_loaders(Xtr, ytr, Xva, yva, batch=16):
    tr = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    va = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva))
    return (DataLoader(tr, batch_size=batch, shuffle=True),
            DataLoader(va, batch_size=16))


@torch.no_grad()
def evaluate(model, loader, device, return_details=False):
    model.eval()
    labels, predictions = [], []
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += nn.functional.cross_entropy(logits, yb, reduction="sum").item()
        predictions.extend(logits.argmax(1).cpu().tolist())
        labels.extend(yb.cpu().tolist())
    acc = float(np.mean(np.array(predictions) == labels))
    if return_details:
        return acc, loss_sum / len(labels), predictions, labels
    return acc


def train_model(model, Xtr, ytr, Xva, yva, *, epochs=40, lr=1e-3,
                weight_decay=1e-3, device="cpu", verbose=False, batch=16):
    """Select weights on validation subjects; test arrays are never accepted here.

    Return best validation accuracy and training history. Earliest epoch wins ties.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    model.to(device)
    tr, va = make_loaders(Xtr, ytr, Xva, yva, batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc, best_state, best_epoch = -1.0, None, -1
    history = []
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)
        acc, va_loss, _, _ = evaluate(model, va, device, True)
        history.append(dict(epoch=ep, train_loss=total_loss/len(ytr),
                            validation_loss=va_loss, validation_acc=acc))
        if acc > best_acc:
            best_acc, best_epoch = acc, ep
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {ep:2d} train_loss {total_loss/len(ytr):.4f} "
                  f"val_loss {va_loss:.4f} val_acc {acc:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.selected_epoch = best_epoch
    return best_acc, history


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
