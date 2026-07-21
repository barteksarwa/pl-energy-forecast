"""Training loop for the sequence models. Pinball loss, early stopping, fixed seeds."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.deep.data import DaySamples

QUANTILES = torch.tensor([0.1, 0.5, 0.9])


def device() -> torch.device:
    import os

    if os.environ.get("FORCE_CPU"):  # smoke tests while an MPS job runs
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pinball(pred: torch.Tensor, target: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    diff = target.unsqueeze(-1) - pred
    return torch.maximum(q * diff, (q - 1.0) * diff).mean()


def _loader(s: DaySamples, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(s.enc, s.fut, s.anchor, s.y)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train_variant(
    net: torch.nn.Module,
    train: DaySamples,
    val: DaySamples,
    checkpoint: str,
    seed: int,
    max_epochs: int = 80,
    patience: int = 10,
    lr: float = 1e-3,
    batch: int = 32,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device()
    net = net.to(dev)
    q = QUANTILES.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    tr = _loader(train, batch, shuffle=True)  # shuffles whole days, safe
    print(f"[{pd.Timestamp.now()}] start seed={seed} device={dev} "
          f"n_train={len(train.days)} n_val={len(val.days)} params="
          f"{sum(p.numel() for p in net.parameters())}", flush=True)

    best_val, best_epoch, t0 = float("inf"), -1, time.time()
    for epoch in range(max_epochs):
        net.train()
        tr_loss = 0.0
        for enc, fut, anchor, y in tr:
            enc, fut, anchor, y = enc.to(dev), fut.to(dev), anchor.to(dev), y.to(dev)
            opt.zero_grad()
            loss = pinball(net(enc, fut, anchor, y_teacher=y), y, q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tr_loss += float(loss) * len(y)
        tr_loss /= len(train.days)

        net.eval()
        with torch.no_grad():
            v = pinball(
                net(val.enc.to(dev), val.fut.to(dev), val.anchor.to(dev)),
                val.y.to(dev), q,
            ).item()
        print(f"[{pd.Timestamp.now()}] epoch {epoch:03d} train {tr_loss:.4f} "
              f"val {v:.4f}", flush=True)
        if v < best_val:
            best_val, best_epoch = v, epoch
            torch.save(net.state_dict(), checkpoint)
        elif epoch - best_epoch >= patience:
            print(f"early stop at {epoch} (best {best_epoch})", flush=True)
            break

    net.load_state_dict(torch.load(checkpoint, map_location=dev))
    return {"best_val_pinball_norm": best_val, "best_epoch": best_epoch,
            "train_s": round(time.time() - t0, 1), "seed": seed}


@torch.no_grad()
def predict_mw(net: torch.nn.Module, s: DaySamples) -> np.ndarray:
    """(n, 24, 3) predictions in MW, denormalized, quantile-ordered."""
    dev = device()
    net = net.to(dev).eval()
    out = net(s.enc.to(dev), s.fut.to(dev), s.anchor.to(dev)).cpu()
    out = out * s.std.view(-1, 1, 1) + s.mean.view(-1, 1, 1)
    out, _ = torch.sort(out, dim=-1)  # enforce p10<=p50<=p90
    return out.numpy()
