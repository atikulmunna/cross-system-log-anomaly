"""Label-free template-rarity anomaly baseline.

A complementary signal to sequential surprise: rare templates are anomalous.
rarity(id) = -log(p(id)) estimated from the TARGET's own template frequencies
(unsupervised, no labels). Sequence score = max rarity over its templates.
This catches POINT anomalies (e.g. a rare alert template) that prediction-based
surprise misses when the anomaly is frequent/repetitive.

Usage:
  python src/rarity_baseline.py --out out
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from data import labeled_systems, load_sequences


def rarity_scores(seqs):
    counts = {}
    for s in seqs:
        for gid in s:
            counts[gid] = counts.get(gid, 0) + 1
    total = sum(counts.values())
    rarity = {gid: -np.log(c / total) for gid, c in counts.items()}
    return np.array([max(rarity[g] for g in s) for s in seqs])
