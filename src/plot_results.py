"""Render result figures from the saved per-sequence scores.

Reads out/<sys>/scores.npz (surprise, rarity, ensemble, labels) written by
ensemble.py, recomputes the semantic-severity signal from the cached template
embeddings, and renders two figures into out/figs/:

  signal_roc.png    : zero-shot ROC-AUC of each label-free signal per system
  fewshot_curve.png : macro ROC of the logistic combiner vs k labels/class,
                      against the zero-shot sum and the single-signal oracle

Usage:
  python src/plot_results.py --out out
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from data import labeled_systems, load_sequences  # noqa: E402
from severity import CONCEPTS  # noqa: E402

KS = [5, 10, 25]
SEEDS = 20


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def load_signals(out_root, tgt, sev_gid):
    seqs, labels = load_sequences(out_root, tgt)
    keep = labels >= 0
    labels = labels[keep]
    seqs = [s for s, k in zip(seqs, keep) if k]
    d = np.load(f"{out_root}/{tgt}/scores.npz")
    sur = z(d["surprise"])
    rar = z(d["rarity"])
    sev = z(np.array([sev_gid[s].max() for s in seqs]))
    return sur, rar, sev, labels


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
    roc = {s: {} for s in systems}
    fewshot = {s: {} for s in systems}
    macro = {"sum": [], "oracle": [], **{k: [] for k in KS}}

    for tgt in systems:
        sur, rar, sev, y = load_signals(args.out, tgt, sev_gid)
        X = np.column_stack([sur, rar, sev])
        roc[tgt]["surprise"] = roc_auc_score(y, sur)
        roc[tgt]["rarity"] = roc_auc_score(y, rar)
        roc[tgt]["severity"] = roc_auc_score(y, sev)
        roc[tgt]["sum"] = roc_auc_score(y, X.sum(axis=1))

        macro["sum"].append(roc[tgt]["sum"])
        macro["oracle"].append(max(roc[tgt][s] for s in ("surprise", "rarity", "severity")))

        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        for k in KS:
            rs = []
            for seed in range(SEEDS):
                rng = np.random.default_rng(seed)
                sup = np.concatenate([
                    rng.choice(pos, k, replace=False),
                    rng.choice(neg, k, replace=False),
                ])
                mask = np.ones(len(y), bool)
                mask[sup] = False
                clf = LogisticRegression(max_iter=1000).fit(X[sup], y[sup])
                rs.append(roc_auc_score(y[mask], clf.predict_proba(X[mask])[:, 1]))
            fewshot[tgt][k] = float(np.mean(rs))
            macro[k].append(fewshot[tgt][k])

    figs = os.path.join(args.out, "figs")
    os.makedirs(figs, exist_ok=True)

    # Figure 1: per-signal zero-shot ROC per system.
    sigs = ["surprise", "rarity", "severity", "sum"]
    x = np.arange(len(systems))
    w = 0.2
    plt.figure(figsize=(8, 4.5))
    for i, sig in enumerate(sigs):
        plt.bar(x + (i - 1.5) * w, [roc[s][sig] for s in systems], w, label=sig)
    plt.axhline(0.5, color="gray", ls="--", lw=0.8)
    plt.xticks(x, systems)
    plt.ylim(0.3, 1.02)
    plt.ylabel("ROC-AUC (zero-shot)")
    plt.title("Label-free signals per held-out system")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "signal_roc.png"), dpi=130)
    plt.close()

    # Figure 2: few-shot macro ROC vs k.
    ks_axis = [0] + KS
    curve = [np.mean(macro["sum"])] + [np.mean(macro[k]) for k in KS]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks_axis, curve, "o-", label="logistic combiner (macro)")
    plt.axhline(
        np.mean(macro["oracle"]), color="green", ls="--",
        label=f"single-signal oracle ({np.mean(macro['oracle']):.2f})",
    )
    plt.xticks(ks_axis, ["0 (sum)"] + [str(k) for k in KS])
    plt.xlabel("labeled examples per class (k)")
    plt.ylabel("macro ROC-AUC")
    plt.title("Few-shot resolves label-free signal selection")
    plt.ylim(0.75, 1.0)
    for xk, yk in zip(ks_axis, curve):
        plt.annotate(f"{yk:.3f}", (xk, yk), textcoords="offset points", xytext=(0, 7))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figs, "fewshot_curve.png"), dpi=130)
    plt.close()

    print("wrote", os.path.join(figs, "signal_roc.png"))
    print("wrote", os.path.join(figs, "fewshot_curve.png"))
    print(
        "macro: sum=%.3f k=25=%.3f oracle=%.3f"
        % (np.mean(macro["sum"]), np.mean(macro[25]), np.mean(macro["oracle"]))
    )


if __name__ == "__main__":
    main()
