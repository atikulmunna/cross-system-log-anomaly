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

        zs_sum = roc_auc_score(y, X.sum(axis=1))
        oracle1 = max(roc_auc_score(y, X[:, j]) for j in range(3))
        macro["zeroshot_sum"].append(zs_sum)
        macro["oracle1"].append(oracle1)

        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        row = {}
        for k in KS:
            rocs = []
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                sp = rng.choice(pos, k, replace=False)
                sn = rng.choice(neg, k, replace=False)
                sup = np.concatenate([sp, sn])
                mask = np.ones(len(y), bool)
                mask[sup] = False
                clf = LogisticRegression(max_iter=1000).fit(X[sup], y[sup])
                proba = clf.predict_proba(X[mask])[:, 1]
                rocs.append(roc_auc_score(y[mask], proba))
            row[k] = float(np.mean(rocs))
            macro[f"k={k}"].append(row[k])
        print(f"{tgt:>12} {zs_sum:8.3f} " + " ".join(f"{row[k]:8.3f}" for k in KS) + f" {oracle1:8.3f}")

    print("\n=== MACRO ROC ===")
    print(f"  zero-shot sum : {np.mean(macro['zeroshot_sum']):.3f}")
    for k in KS:
        print(f"  few-shot k={k:<3}: {np.mean(macro[f'k={k}']):.3f}")
    print(f"  oracle1 (UB)  : {np.mean(macro['oracle1']):.3f}")


if __name__ == "__main__":
    main()
