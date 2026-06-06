"""Label-free adaptive signal selection.

Each signal's own score distribution tells us if it found something: a signal
with real anomalies has a heavy upper tail far beyond its normal spread; a noise
signal is bland/unimodal. We measure tail separation (p99-p50)/IQR per signal
(NO labels) and use it to (a) hard-select the best signal and (b) soft-weight a
combination. Goal: close the fixed-fusion (~0.81) -> oracle (~0.94) macro gap.

Usage:
  python src/adaptive.py --out out
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sentence_transformers import SentenceTransformer

from data import labeled_systems, load_sequences
from severity import CONCEPTS


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def tail_sep(x):
    """Heavy-tail separation: (p99 - p50) / IQR. High when a signal has a small
    set of extreme outliers (candidate anomalies) above its bulk."""
    p99, p50, p25, p75 = np.percentile(x, [99, 50, 25, 75])
    return (p99 - p50) / (p75 - p25 + 1e-9)


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

    macro = {k: [] for k in ["sum", "hard", "soft", "oracle1"]}
    macro_pr = {k: [] for k in macro}

    for tgt in labeled_systems(args.out):
        seqs, labels = load_sequences(args.out, tgt)
        keep = labels >= 0
        labels = labels[keep]
        seqs = [s for s, k in zip(seqs, keep) if k]

        d = np.load(f"{args.out}/{tgt}/scores.npz")
        sig = {
            "surprise": z(d["surprise"]),
            "rarity": z(d["rarity"]),
            "severity": z(np.array([sev_gid[s].max() for s in seqs])),
        }
        names = list(sig)
        rocs = {k: roc_auc_score(labels, v) for k, v in sig.items()}
        meta = {k: tail_sep(v) for k, v in sig.items()}

        hard_k = max(meta, key=meta.get)
        oracle_k = max(rocs, key=rocs.get)

        print(f"\n== {tgt} (base {labels.mean()*100:.1f}%) ==")
        for k in names:
            print(f"  {k:>9}: ROC={rocs[k]:.3f}  tail_sep={meta[k]:6.2f}")
        flag = "OK" if hard_k == oracle_k else "MISS"
        print(f"  meta picks '{hard_k}' | oracle '{oracle_k}'  [{flag}]")


if __name__ == "__main__":
    main()
