"""Loading sequences + the shared embedding matrix for training/eval."""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


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


class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def make_collate(max_len=512):
    def collate(batch):
        L = min(max(len(s) for s in batch), max_len)
        B = len(batch)
        ids = np.zeros((B, L), dtype=np.int64)
        pad = np.ones((B, L), dtype=bool)
        for i, s in enumerate(batch):
            s = s[:L]
            ids[i, : len(s)] = s
            pad[i, : len(s)] = False
        return torch.from_numpy(ids), torch.from_numpy(pad)

    return collate
