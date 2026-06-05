"""Semantic-severity signal: the third, label-free anomaly signal.

severity(template) = max cosine similarity between the template's (cached,
L2-normalized) embedding and a set of severity-concept embeddings. Sequence
score = max severity over its templates. This catches MARKED anomalies that are
frequent (rarity misses) and predictable (surprise misses) but textually severe
-- e.g. Thunderbird's "Fatal error (Local Catastrophic Error)".

Reuses out/<sys>/scores.npz (surprise, rarity, labels) from ensemble.py to also
report 2- and 3-signal ensembles. Runs on CPU (base env), no training.

Usage:
  python src/severity.py --out out
"""
import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score, roc_auc_score

from data import labeled_systems, load_sequences

CONCEPTS = [
    "fatal error", "critical failure", "catastrophic error", "kernel panic",
    "system crash", "aborted", "operation failed", "exception thrown",
    "unable to read", "hardware error", "memory error", "data corruption",
    "timeout", "unrecoverable error", "device failure",
]


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    vectors = np.load(os.path.join(args.out, "embeddings", "vectors.npy"))  # [V, d], L2
    model = SentenceTransformer(args.model)
    cvecs = model.encode(CONCEPTS, normalize_embeddings=True).astype("float32")
    severity_of_gid = (vectors @ cvecs.T).max(axis=1)  # [V] max cos sim to any concept

    print(f"{'target':>12} {'method':>22} {'PR-AUC':>8} {'ROC-AUC':>8}")
    for tgt in labeled_systems(args.out):
        seqs, labels = load_sequences(args.out, tgt)
        keep = labels >= 0
        labels = labels[keep]
        seqs = [s for s, k in zip(seqs, keep) if k]
        severity = np.array([severity_of_gid[s].max() for s in seqs])
        pr = average_precision_score(labels, severity)
        roc = roc_auc_score(labels, severity)
        print(f"{tgt:>12} {'severity':>22} {pr:8.4f} {roc:8.4f}")
        print()


if __name__ == "__main__":
    main()
