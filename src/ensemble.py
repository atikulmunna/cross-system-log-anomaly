"""Do surprise and rarity catch DIFFERENT anomalies? (complementarity test)

For each labeled target: train the surprise model on all other systems, then
score the target with (a) sequential surprise, (b) template rarity, (c) their
z-score ensemble. Reports PR/ROC for each, the rank correlation between the two
signals, and how many true anomalies surprise catches in its top-P that rarity
misses (P = number of positives). If the ensemble beats rarity alone, the
learned model earns its keep.

Usage:
  C:/Users/Munna/anaconda3/envs/logzs/python.exe src/ensemble.py --out out --epochs 3
"""
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from data import all_systems, labeled_systems, load_sequences, load_vectors
from loso import score, train
from model import LogTransformer
from rarity_baseline import rarity_scores


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    vectors = load_vectors(args.out)
    every = all_systems(args.out)
    labeled = labeled_systems(args.out)

    for tgt in labeled:
        train_systems = [s for s in every if s != tgt]
        train_seqs = []
        for s in train_systems:
            seqs, _ = load_sequences(args.out, s, max_len=args.max_len)
            train_seqs.extend(seqs)
        print(f"\n=== target={tgt}  ({len(train_seqs)} train seqs) ===", flush=True)

        model = LogTransformer(
            vectors, d_model=args.d_model, nlayers=args.nlayers, max_len=args.max_len
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        train(model, train_seqs, args.epochs, args.batch_size, args.lr, args.max_len)

        seqs, labels = load_sequences(args.out, tgt, max_len=args.max_len)
        keep = labels >= 0
        labels = labels[keep]
        seqs = [s for s, k in zip(seqs, keep) if k]

        sc = score(model, seqs, args.batch_size, args.max_len)
        is_window = len({len(s) for s in seqs}) == 1
        surprise = sc["max"] if is_window else sc["mean"]
        rarity = rarity_scores(seqs)
        ens = z(surprise) + z(rarity)

        np.savez(
            f"{args.out}/{tgt}/scores.npz",
            surprise=surprise, rarity=rarity, ensemble=ens, labels=labels,
        )

        print(f"  unit={'window' if is_window else 'session'}  base={labels.mean()*100:.1f}%")
        for name, v in [("surprise", surprise), ("rarity", rarity), ("ensemble", ens)]:
            pr = average_precision_score(labels, v)
            roc = roc_auc_score(labels, v)
            print(f"  [{name:>8}] PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}", flush=True)

        rho = spearmanr(surprise, rarity).correlation
        P = int(labels.sum())
        pos = set(np.where(labels == 1)[0])
        top_sur = set(np.argsort(-surprise)[:P])
        top_rar = set(np.argsort(-rarity)[:P])
        tp_sur = len(top_sur & pos)
        tp_rar = len(top_rar & pos)
        only_sur = len((top_sur & pos) - top_rar)  # caught by surprise, not rarity
        print(
            f"  spearman(surprise,rarity)={rho:.3f}  "
            f"top-{P}: rarity TP={tp_rar}, surprise TP={tp_sur}, "
            f"surprise-only TP={only_sur}",
            flush=True,
        )


if __name__ == "__main__":
    main()
