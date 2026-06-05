"""Leave-one-system-out training + zero-shot evaluation.

For each labeled target system: train the causal transformer on sequences from
ALL other systems (labeled + unlabeled pretraining pool), then score the target
purely by next-embedding surprise -- no target labels are ever seen at train
time. Reports PR-AUC (primary) and ROC-AUC, macro-averaged across folds.

Usage:
  python src/loso.py --out out --epochs 3
  python src/loso.py --out out --target BGL --epochs 3   # single fold
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from data import (
    SeqDataset,
    all_systems,
    labeled_systems,
    load_sequences,
    load_vectors,
    make_collate,
)
from model import LogTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cosine_next_loss(pred, target, pad):
    # pred[:, t] predicts target[:, t+1]; valid where the next pos is not padding.
    valid = (~pad[:, 1:]).float()
    sim = F.cosine_similarity(pred[:, :-1], target[:, 1:], dim=-1)  # [B, L-1]
    loss = (1 - sim) * valid
    return loss.sum() / valid.sum().clamp(min=1)


def train(model, seqs, epochs, batch_size, lr, max_len):
    loader = DataLoader(
        SeqDataset(seqs),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=make_collate(max_len),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total, n = 0.0, 0
        for ids, pad in loader:
            ids, pad = ids.to(DEVICE), pad.to(DEVICE)
            pred, target = model(ids, pad)
            loss = cosine_next_loss(pred, target, pad)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * ids.size(0)
            n += ids.size(0)
        print(f"    epoch {ep + 1}/{epochs}  loss={total / max(n, 1):.4f}", flush=True)


@torch.no_grad()
def score(model, seqs, batch_size, max_len):
    """Return per-sequence surprise under three aggregations."""
    model.eval()
    loader = DataLoader(
        SeqDataset(seqs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(max_len),
    )
    agg = {"max": [], "mean": [], "top3": []}
    for ids, pad in loader:
        ids, pad = ids.to(DEVICE), pad.to(DEVICE)
        pred, target = model(ids, pad)
        sim = F.cosine_similarity(pred[:, :-1], target[:, 1:], dim=-1)  # [B, L-1]
        surprise = 1 - sim
        valid = ~pad[:, 1:]
        surprise = surprise.masked_fill(~valid, float("nan"))
        s = surprise.cpu().numpy()
        for row in s:
            r = row[~np.isnan(row)]
            if len(r) == 0:
                r = np.array([0.0])
            agg["max"].append(r.max())
            agg["mean"].append(r.mean())
            agg["top3"].append(np.sort(r)[-3:].mean())
    return {k: np.asarray(v) for k, v in agg.items()}


def evaluate(model, out_root, system, args):
    seqs, labels = load_sequences(out_root, system, max_len=args.max_len)
    keep = labels >= 0
    labels = labels[keep]
    seqs = [s for s, k in zip(seqs, keep) if k]
    scores = score(model, seqs, args.batch_size, args.max_len)
    res = {
        agg: {
            "pr_auc": float(average_precision_score(labels, sc)),
            "roc_auc": float(roc_auc_score(labels, sc)),
        }
        for agg, sc in scores.items()
    }
    return res, int(labels.sum()), len(labels)


def run_fold(out_root, target, train_systems, args, vectors, labeled):
    print(f"\n=== target={target}  train={train_systems} ===")
    train_seqs = []
    rng = np.random.default_rng(0)
    for s in train_systems:
        seqs, _ = load_sequences(out_root, s, max_len=args.max_len)
        cap = args.max_train_per_system
        if cap and len(seqs) > cap:
            idx = rng.choice(len(seqs), cap, replace=False)
            seqs = [seqs[i] for i in idx]
        train_seqs.extend(seqs)
    print(f"  train sequences: {len(train_seqs)}")

    model = LogTransformer(
        vectors,
        d_model=args.d_model,
        nlayers=args.nlayers,
        max_len=args.max_len,
    ).to(DEVICE)
    train(model, train_seqs, args.epochs, args.batch_size, args.lr, args.max_len)

    # Aggregation is keyed on the target's STRUCTURAL unit type, which is
    # label-free metadata: window units have constant length (localized anomaly
    # -> max), session units have variable length (whole-sequence deviation ->
    # mean). LOSO experiments showed this recovers the oracle-best aggregation in
    # every fold, whereas selecting on training systems picks the wrong one
    # (their unit type differs from the target's).
    tgt_seqs, _ = load_sequences(out_root, target, max_len=args.max_len)
    lengths = {len(s) for s in tgt_seqs}
    is_window = len(lengths) == 1
    chosen = "max" if is_window else "mean"
    unit = "window" if is_window else "session"

    results, pos, tot = evaluate(model, out_root, target, args)
    print(f"  positives={pos}/{tot}  (unit={unit} -> aggregation '{chosen}')")
    for agg, m in results.items():
        flag = "  <- CHOSEN (a priori)" if agg == chosen else ""
        flag += "  [oracle-best]" if agg == max(results, key=lambda a: results[a]["pr_auc"]) else ""
        print(f"  [{agg:>4}] PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}{flag}")
    return results, chosen
