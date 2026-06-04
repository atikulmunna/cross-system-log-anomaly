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
