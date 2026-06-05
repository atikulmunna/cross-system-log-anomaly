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
