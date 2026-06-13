"""Aggregate multi-seed ensemble runs into mean +/- 95% CI.

Reads out/<sys>/scores_seed*.npz written by `ensemble.py --seed N`, recomputes
the deterministic semantic-severity signal, and reports per-system and macro
ROC-AUC with t-interval 95% confidence intervals across seeds.

Only the surprise signal (and the 3-signal sum that uses it) carries seed
variance; rarity and severity are deterministic, so their CIs are ~0 by
construction -- that contrast is the point.

Usage:
  python src/aggregate_seeds.py --out out
"""
import argparse
import glob
import os

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer

from data import labeled_systems, load_sequences
from severity import CONCEPTS

METHODS = ["surprise", "rarity", "severity", "sum"]


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def ci95(vals):
    """Return (mean, half-width) for a 95% t-interval."""
    a = np.asarray(vals, dtype=float)
    m = a.mean()
    if len(a) < 2:
        return m, 0.0
    se = a.std(ddof=1) / np.sqrt(len(a))
    return m, se * stats.t.ppf(0.975, len(a) - 1)


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
    # per-seed macro accumulator: macro_by_seed[method][seed_index]
    macro = {m: [] for m in METHODS}
    n_seeds = None

    print(f"{'target':>12} {'method':>10}   mean +/- 95% CI   (seeds)")
    for tgt in systems:
        seqs, labels = load_sequences(args.out, tgt)
        keep = labels >= 0
        labels = labels[keep]
        seqs = [s for s, k in zip(seqs, keep) if k]
        severity = z(np.array([sev_gid[s].max() for s in seqs]))

        files = sorted(glob.glob(f"{args.out}/{tgt}/scores_seed*.npz"))
        if not files:
            print(f"{tgt:>12}   (no scores_seed*.npz -- run ensemble.py --seed N)")
            continue
        per_seed = {m: [] for m in METHODS}
        for f in files:
            d = np.load(f)
            sur, rar = z(d["surprise"]), z(d["rarity"])
            vals = {
                "surprise": roc_auc_score(labels, sur),
                "rarity": roc_auc_score(labels, rar),
                "severity": roc_auc_score(labels, severity),
                "sum": roc_auc_score(labels, sur + rar + severity),
            }
            for m in METHODS:
                per_seed[m].append(vals[m])
        n_seeds = len(files)
        for m in METHODS:
            mean, half = ci95(per_seed[m])
            macro[m].append(per_seed[m])
            print(f"{tgt:>12} {m:>10}   {mean:.3f} +/- {half:.3f}   (n={n_seeds})")
        print()

    if not any(macro[m] for m in METHODS):
        return

    print("=== MACRO ROC (mean +/- 95% CI over seeds) ===")
    for m in METHODS:
        # macro per seed = mean across systems for that seed, then CI over seeds
        per_system = np.array(macro[m])              # [n_systems, n_seeds]
        macro_per_seed = per_system.mean(axis=0)     # [n_seeds]
        mean, half = ci95(macro_per_seed)
        print(f"  {m:>9}: {mean:.3f} +/- {half:.3f}")


if __name__ == "__main__":
    main()
