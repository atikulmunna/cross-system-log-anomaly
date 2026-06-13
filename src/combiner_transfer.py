"""Cross-system combiner transfer: a zero-shot signal selector.

Label-free selection from each signal's distribution shape failed (adaptive.py).
But the few-shot combiner (fewshot.py) shows a LINEAR mix of the three signals is
all that's needed. This tests whether those weights TRANSFER across systems: fit
one logistic combiner on the labeled TRAINING systems (each system's signals
z-scored on its own), then apply it to the held-out target. No target labels are
used, so it stays zero-shot.

If it beats the equal-weight sum, the §4.4 negative result gets a positive
resolution: a real zero-shot selector. If not, it is a second failed selector.

Reuses out/<sys>/scores.npz (surprise, rarity, labels) + recomputed severity.

Usage:
  python src/combiner_transfer.py --out out
"""
import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sentence_transformers import SentenceTransformer

from data import labeled_systems, load_sequences
from severity import CONCEPTS


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def system_matrix(out_root, sys, sev_gid):
    """Per-system z-scored [surprise, rarity, severity] matrix + labels."""
    seqs, labels = load_sequences(out_root, sys)
    keep = labels >= 0
    labels = labels[keep]
    seqs = [s for s, k in zip(seqs, keep) if k]
    d = np.load(f"{out_root}/{sys}/scores.npz")
    X = np.column_stack([
        z(d["surprise"]),
        z(d["rarity"]),
        z(np.array([sev_gid[s].max() for s in seqs])),
    ])
    return X, labels


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

    systems = labeled_systems(args.out)
    data = {s: system_matrix(args.out, s, sev_gid) for s in systems}

    macro = {"transfer": [], "sum": [], "oracle1": []}
    names = ["surprise", "rarity", "severity"]
    print(
        f"{'target':>12} {'transfer':>9} {'sum':>7} {'oracle1':>8} "
        f"{'PR(tr)':>7}   learned weights (sur, rar, sev)"
    )
    for tgt in systems:
        Xte, yte = data[tgt]
        Xtr = np.vstack([data[s][0] for s in systems if s != tgt])
        ytr = np.concatenate([data[s][1] for s in systems if s != tgt])

        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]

        roc_t = roc_auc_score(yte, proba)
        pr_t = average_precision_score(yte, proba)
        roc_sum = roc_auc_score(yte, Xte.sum(axis=1))
        roc_oracle = max(roc_auc_score(yte, Xte[:, j]) for j in range(3))
        macro["transfer"].append(roc_t)
        macro["sum"].append(roc_sum)
        macro["oracle1"].append(roc_oracle)

        w = clf.coef_.ravel()
        wstr = ", ".join(f"{n}={wi:+.2f}" for n, wi in zip(names, w))
        print(
            f"{tgt:>12} {roc_t:9.3f} {roc_sum:7.3f} {roc_oracle:8.3f} "
            f"{pr_t:7.3f}   {wstr}"
        )

    print("\n=== MACRO ROC ===")
    print(f"  transfer combiner   : {np.mean(macro['transfer']):.3f}")
    print(f"  equal-weight sum    : {np.mean(macro['sum']):.3f}")
    print(f"  single-signal oracle: {np.mean(macro['oracle1']):.3f}")
    verdict = (
        "beats the sum -> zero-shot selector works"
        if np.mean(macro["transfer"]) > np.mean(macro["sum"]) + 1e-9
        else "does not beat the sum -> a second failed selector"
    )
    print(f"  verdict: transfer {verdict}")


if __name__ == "__main__":
    main()
