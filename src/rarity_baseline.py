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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    print(f"{'system':>12} {'base%':>7} {'PR-AUC':>8} {'ROC-AUC':>8}")
    for sys in labeled_systems(args.out):
        seqs, labels = load_sequences(args.out, sys)
        keep = labels >= 0
        labels = labels[keep]
        seqs = [s for s, k in zip(seqs, keep) if k]
        scores = rarity_scores(seqs)
        pr = average_precision_score(labels, scores)
        roc = roc_auc_score(labels, scores)
        print(f"{sys:>12} {labels.mean()*100:6.1f}% {pr:8.4f} {roc:8.4f}")


if __name__ == "__main__":
    main()
