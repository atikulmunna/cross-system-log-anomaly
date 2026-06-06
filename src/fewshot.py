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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    vectors = np.load(f"{args.out}/embeddings/vectors.npy")
    cvecs = (
        SentenceTransformer("all-MiniLM-L6-v2")
        .encode(CONCEPTS, normalize_embeddings=True)
        .astype("float32")
    )
    sev_gid = (vectors @ cvecs.T).max(axis=1)

    macro = {f"k={k}": [] for k in KS}
    macro["zeroshot_sum"] = []
    macro["oracle1"] = []

    hdr = f"{'target':>12} {'zs_sum':>8} " + " ".join(f"{'k='+str(k):>8}" for k in KS) + f" {'oracle1':>8}"
    print(hdr)
    for tgt in labeled_systems(args.out):
        seqs, labels = load_sequences(args.out, tgt)
        keep = labels >= 0
        labels = labels[keep]
        seqs = [s for s, k in zip(seqs, keep) if k]
        d = np.load(f"{args.out}/{tgt}/scores.npz")
        X = np.column_stack([
            z(d["surprise"]),
            z(d["rarity"]),
            z(np.array([sev_gid[s].max() for s in seqs])),
        ])
        y = labels


if __name__ == "__main__":
    main()
