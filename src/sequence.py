"""Build per-system (template-sequence, label) units (pipeline step 3).

Unified interface, per-system cut (the locked design decision):
  HDFS         -> session grouping by block id; label from anomaly_label.csv
  BGL / Thunderbird / OpenStack -> fixed-count sliding window (W, stride);
                  window label = max over its line labels (1 if any anomaly)

Output per system: <out>/<system>/sequences.npz

Usage:
  python src/sequence.py --system BGL --out out --window 100 --stride 50
  python src/sequence.py --system HDFS --out out --hdfs-labels path/anomaly_label.csv
"""
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

    if system == "HDFS":
        block_label = {}
        if hdfs_labels and os.path.exists(hdfs_labels):
            lab = pd.read_csv(hdfs_labels)  # columns: BlockId, Label
            block_label = {
                b: (1 if str(v).strip().lower() == "anomaly" else 0)
                for b, v in zip(lab["BlockId"], lab["Label"])
            }
        for blk, grp in df.dropna(subset=["blk_id"]).groupby("blk_id", sort=False):
            seqs.append(grp["gid"].to_numpy(dtype="int32"))
            labels.append(block_label.get(blk, -1))
    else:
        gids = df["gid"].to_numpy(dtype="int32")
        has_label = "label" in df.columns
        line_lbl = df["label"].to_numpy() if has_label else None
        n = len(gids)
        last = max(n - window + 1, 1)
        for start in range(0, last, stride):
            seqs.append(gids[start : start + window])
            if has_label:
                labels.append(int(line_lbl[start : start + window].max()))
            else:
                labels.append(-1)

    out_path = os.path.join(out_root, system, "sequences.npz")
    labels = np.array(labels, dtype="int64")
    np.savez(out_path, sequences=np.array(seqs, dtype=object), labels=labels)

    n_known = int((labels >= 0).sum())
    n_pos = int((labels == 1).sum())
    lengths = [len(s) for s in seqs]
    print(
        f"[{system}] sequences={len(seqs)} labeled={n_known} positives={n_pos} "
        f"len(min/med/max)={min(lengths)}/{int(np.median(lengths))}/{max(lengths)} "
        f"-> {out_path}"
    )


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
