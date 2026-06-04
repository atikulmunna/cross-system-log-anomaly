"""Loading sequences + the shared embedding matrix for training/eval."""
import glob
import os

import numpy as np


def load_vectors(out_root):
    return np.load(os.path.join(out_root, "embeddings", "vectors.npy"))


def all_systems(out_root):
    return sorted(
        os.path.basename(os.path.dirname(p))
        for p in glob.glob(os.path.join(out_root, "*", "sequences.npz"))
    )


def labeled_systems(out_root):
    out = []
    for s in all_systems(out_root):
        z = np.load(os.path.join(out_root, s, "sequences.npz"), allow_pickle=True)
        if (z["labels"] >= 0).any():
            out.append(s)
    return out


def load_sequences(out_root, system, min_len=2, max_len=512):
    """Return (list[int64 array], labels array) filtered to usable lengths."""
    z = np.load(os.path.join(out_root, system, "sequences.npz"), allow_pickle=True)
    seqs, labels = [], []
    for s, lab in zip(z["sequences"], z["labels"]):
        if len(s) >= min_len:
            seqs.append(np.asarray(s[:max_len], dtype=np.int64))
            labels.append(int(lab))
    return seqs, np.asarray(labels, dtype=np.int64)
