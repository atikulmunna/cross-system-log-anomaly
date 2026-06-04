"""Parse raw LogHub logs into Drain3 templates (pipeline step 1)."""
import argparse
import os

import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from formats import LOG_FORMATS, generate_logformat_regex


def parse_file(system, input_path, out_root, sim_th=0.4, depth=4):
    _, regex = generate_logformat_regex(LOG_FORMATS[system])
    cfg = TemplateMinerConfig()
    cfg.drain_sim_th = sim_th
    cfg.drain_depth = depth
    cfg.profiling_enabled = False
    miner = TemplateMiner(config=cfg)

    rows = []
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fh:
        for idx, raw in enumerate(fh):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            m = regex.match(line)
            if not m:
                continue
            content = (m.groupdict().get("Content") or "").strip()
            res = miner.add_log_message(content)
            rows.append({"idx": idx, "template_id": res["cluster_id"]})

    df = pd.DataFrame(rows)
    id_to_template = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    df["template"] = df["template_id"].map(id_to_template)
    out_dir = os.path.join(out_root, system)
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, "parsed.parquet"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    parse_file(args.system, args.input, args.out)


if __name__ == "__main__":
    main()
