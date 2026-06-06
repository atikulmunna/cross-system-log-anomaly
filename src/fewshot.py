"""Few-shot signal combination: resolve the selection problem with k labels.

Label-free selection failed (shape != relevance). Here we give a tiny logistic
combiner k labeled examples PER CLASS per system, fit it on the 3 z-scored
signals (surprise, rarity, severity), and evaluate on the held-out remainder.
Averaged over many random support draws. Tests whether a handful of labels
recovers the per-system oracle the unsupervised selector couldn't.

Usage:
  python src/fewshot.py --out out
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sentence_transformers import SentenceTransformer

from data import labeled_systems, load_sequences
from severity import CONCEPTS

KS = [5, 10, 25]
SEEDS = 20


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)
