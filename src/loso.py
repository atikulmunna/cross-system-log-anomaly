"""Leave-one-system-out training + zero-shot evaluation.

For each labeled target system: train the causal transformer on sequences from
ALL other systems (labeled + unlabeled pretraining pool), then score the target
purely by next-embedding surprise -- no target labels are ever seen at train
time. Reports PR-AUC (primary) and ROC-AUC, macro-averaged across folds.

Usage:
  python src/loso.py --out out --epochs 3
  python src/loso.py --out out --target BGL --epochs 3   # single fold
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from data import (
    SeqDataset,
    all_systems,
    labeled_systems,
    load_sequences,
    load_vectors,
    make_collate,
)
from model import LogTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cosine_next_loss(pred, target, pad):
    # pred[:, t] predicts target[:, t+1]; valid where the next pos is not padding.
    valid = (~pad[:, 1:]).float()
    sim = F.cosine_similarity(pred[:, :-1], target[:, 1:], dim=-1)  # [B, L-1]
    loss = (1 - sim) * valid
    return loss.sum() / valid.sum().clamp(min=1)


def train(model, seqs, epochs, batch_size, lr, max_len):
    loader = DataLoader(
        SeqDataset(seqs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate(max_len),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total, n = 0.0, 0
        for ids, pad in loader:
            ids, pad = ids.to(DEVICE), pad.to(DEVICE)
            pred, target = model(ids, pad)
            loss = cosine_next_loss(pred, target, pad)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * ids.size(0)
            n += ids.size(0)
        print(f"    epoch {ep + 1}/{epochs}  loss={total / max(n, 1):.4f}", flush=True)
