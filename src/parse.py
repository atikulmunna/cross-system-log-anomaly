"""Parse raw LogHub logs into structured templates with Drain3.

Pipeline step 1. For each line: split header via the per-system format regex,
feed the Content to Drain3 to mine an event template, and record the cluster id.
Templates are resolved to their FINAL form after the full pass (Drain refines
clusters as it learns), so every line maps to a stable template string.

Output per system (in <out>/<system>/):
  parsed.parquet : one row per line [idx, content, template_id, template,
                   label?, blk_id?]
  templates.csv  : [template_id, template] (unique, final)

Usage:
  python src/parse.py --system BGL --input loghub/BGL/BGL_2k.log --out out
"""
import argparse
import os
import re

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from formats import LOG_FORMATS, LABEL_SYSTEMS, generate_logformat_regex

BLK_RE = re.compile(r"(blk_-?\d+)")  # HDFS block id, for session grouping


def parse_file(system, input_path, out_root, sim_th=0.4, depth=4):
    if system not in LOG_FORMATS:
        raise ValueError(f"Unknown system {system!r}. Known: {sorted(LOG_FORMATS)}")
    _, regex = generate_logformat_regex(LOG_FORMATS[system])

    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = sim_th
    cfg.drain_depth = depth
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)

    rows = []
    n_total = n_matched = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fh:
        for idx, raw in enumerate(fh):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            n_total += 1
            m = regex.match(line)
            if not m:
                continue
            n_matched += 1
            fields = m.groupdict()
            content = (fields.get("Content") or "").strip()
            res = miner.add_log_message(content)
            # content is intentionally not stored: at 10M+ lines it dominates
            # memory/disk and nothing downstream needs it (templates suffice).
            row = {
                "idx": idx,
                "template_id": res["cluster_id"],
            }
            if system in LABEL_SYSTEMS:
                row["label"] = 0 if (fields.get("Label", "-") == "-") else 1
            if system == "HDFS":
                found = BLK_RE.findall(content)
                row["blk_id"] = found[0] if found else None
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No lines matched the {system} format. Check LOG_FORMATS[{system!r}]."
        )

    # Resolve every line to its final template (clusters evolve during the pass).
    id_to_template = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    df = pd.DataFrame(rows)
    df["template"] = df["template_id"].map(id_to_template)

    out_dir = os.path.join(out_root, system)
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "parsed.parquet"))
    templates = (
        df[["template_id", "template"]]
        .drop_duplicates("template_id")
        .sort_values("template_id")
    )
    templates.to_csv(os.path.join(out_dir, "templates.csv"), index=False)

    match_rate = 100 * n_matched / max(n_total, 1)
    print(
        f"[{system}] lines={n_total} matched={n_matched} ({match_rate:.1f}%) "
        f"templates={len(templates)} -> {out_dir}"
    )
    if match_rate < 90:
        print(f"  WARNING: low match rate for {system}; format may need a fix.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--sim-th", type=float, default=0.4)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()
    parse_file(args.system, args.input, args.out, args.sim_th, args.depth)


if __name__ == "__main__":
    main()
