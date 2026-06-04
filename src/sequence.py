"""Build per-system (template-sequence, label) units (pipeline step 3)."""
import argparse
import os

import numpy as np
import pandas as pd


def _vocab_map(out_root):
    v = pd.read_parquet(os.path.join(out_root, "embeddings", "vocab.parquet"))
    return dict(zip(v["template"].astype(str), v["global_id"].astype(int)))


def build_system(system, out_root, window=100, stride=50, hdfs_labels=None):
    df = pd.read_parquet(os.path.join(out_root, system, "parsed.parquet"))
    df = df.sort_values("idx").reset_index(drop=True)
    gid_of = _vocab_map(out_root)
    df["gid"] = df["template"].astype(str).map(gid_of).astype("int32")

    seqs, labels = [], []
    gids = df["gid"].to_numpy(dtype="int32")
    has_label = "label" in df.columns
    line_lbl = df["label"].to_numpy() if has_label else None
    n = len(gids)
    last = max(n - window + 1, 1)
    for start in range(0, last, stride):
        seqs.append(gids[start : start + window])
        labels.append(int(line_lbl[start : start + window].max()) if has_label else -1)

    out_path = os.path.join(out_root, system, "sequences.npz")
    np.savez(out_path, sequences=np.array(seqs, dtype=object),
             labels=np.array(labels, dtype="int64"))
    print(f"[{system}] sequences={len(seqs)} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--hdfs-labels", default=None)
    args = ap.parse_args()
    build_system(args.system, args.out, args.window, args.stride, args.hdfs_labels)


if __name__ == "__main__":
    main()
