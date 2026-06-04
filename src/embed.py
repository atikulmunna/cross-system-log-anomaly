"""Embed unique log templates into a shared semantic space (pipeline step 2).

Gathers every unique template string across all parsed systems, embeds each
ONCE with a frozen sentence-transformer (all-MiniLM-L6-v2), L2-normalizes, and
caches the result. Because identical template strings map to the same global id,
the space is shared across systems -- this is what makes cross-system transfer
possible. The embedder is frozen; this runs once on CPU.

Output (in <out>/embeddings/):
  vocab.parquet : [global_id, template]
  vectors.npy   : float32 [num_templates, dim], row i == global_id i, L2-normed

Usage:
  python src/embed.py --out out
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def build(out_root, model_name="all-MiniLM-L6-v2", batch_size=256):
    files = sorted(glob.glob(os.path.join(out_root, "*", "templates.csv")))
    if not files:
        raise RuntimeError(f"No templates.csv under {out_root}. Run parse.py first.")

    templates = set()
    for f in files:
        # keep_default_na=False so empty-string templates stay "" (not NaN);
        # otherwise they vanish here and fail to map in sequence.py.
        col = pd.read_csv(f, keep_default_na=False)["template"].astype(str)
        templates.update(col.tolist())
    templates = sorted(templates)
    print(f"collected {len(templates)} unique templates from {len(files)} systems")

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        templates,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype("float32")

    emb_dir = os.path.join(out_root, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)
    pd.DataFrame({"global_id": range(len(templates)), "template": templates}).to_parquet(
        os.path.join(emb_dir, "vocab.parquet")
    )
    np.save(os.path.join(emb_dir, "vectors.npy"), vecs)
    print(f"embedded {len(templates)} templates dim={vecs.shape[1]} -> {emb_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    build(args.out, args.model, args.batch_size)


if __name__ == "__main__":
    main()
