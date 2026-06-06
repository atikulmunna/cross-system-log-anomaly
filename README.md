# Cross-System Zero-Shot Log Anomaly Detection

This is a log anomaly model that scores a system it has **never seen during
training**. The interesting part isn't scale, it's the generalization trick and
an honest **leave-one-system-out (LOSO)** evaluation that most papers quietly
skip.

> **Full write-up: [REPORT.md](REPORT.md).** The short version: zero-shot
> transfer collapses and trivial baselines beat the learned model. Anomalies
> fall into a taxonomy (contextual, point, marked), and each type is caught by a
> different label-free signal (surprise, rarity, severity). Picking the right
> signal without labels turns out to fail, but 5 to 25 labels per system fix it
> and get you to **macro ROC 0.94**. Everything runs on one 8 GB GPU for $0.

## Core idea

Naive cross-system transfer fails because models latch onto each system's own
*vocabulary*. The way around it is to never work on raw tokens at all, and
operate in a **shared semantic space** instead.

```
raw logs (N systems)
  -> Drain3 parse ............ templates ("Receiving block <*> src: <*>")
       -> frozen MiniLM embed . shared 384-d vectors (cross-system!)
            -> group into sequences . (template-seq, label)
                 -> small causal transformer over embeddings
                      -> surprise (predicted vs actual next embedding) = score
```

Surprise is **unsupervised**, and that's exactly why it carries over to a system
the model has never seen.

## Locked design decisions

| Decision | Choice | Why |
|---|---|---|
| Granularity | Per-system cut, unified `(seq, label)` interface | HDFS uses block sessions (keeps comparability with the literature); BGL/TB/OpenStack use fixed-count windows |
| Window | 100 lines, stride 50 | counting by lines keeps sequence lengths comparable despite very different log rates |
| Embedder | `all-MiniLM-L6-v2`, frozen, cached | fast on CPU ($0), strong, and easy to swap out for an ablation |
| Representation | embedding **regression**, not template-id classification | template ids aren't shared across systems, but the vector space is |
| Metric | PR-AUC primary, ROC-AUC secondary, macro-avg | holds up across systems with very different anomaly base rates |

## Datasets (labeled)

| System | Unit | Labels |
|---|---|---|
| HDFS  | block session | `anomaly_label.csv` (2.9% of 575k blocks anomalous) |
| BGL   | 100-line window | inline (`-` = normal) |
| Thunderbird | 100-line window | inline; using the first 5M-line slice (the full file is 32 GB) |
| OpenStack | VM-instance session | 4 abnormal instance UUIDs in `anomaly_labels.txt` |

Unlabeled systems (Spark, Hadoop, Zookeeper, HPC, Apache, Linux, OpenSSH, and so
on) can be dropped into the **pretraining pool** to strengthen generalization.

## Pipeline

```
python src/parse.py    --system HDFS --input data/HDFS/HDFS.log --out out
python src/embed.py    --out out                       # build shared vector cache
python src/sequence.py --system HDFS --out out --hdfs-labels data/HDFS/anomaly_label.csv
python src/loso.py     --out out                       # train+eval, leave-one-system-out
```

Outputs land in `out/<system>/` (`parsed.parquet`, `templates.csv`,
`sequences.npz`) and `out/embeddings/` (`vocab.parquet`, `vectors.npy`).

## Evaluation protocol (the headline)

| Setting | Meaning |
|---|---|
| Oracle | trained on the target system, so this is the upper bound |
| Naive transfer | train on one system, apply raw to another, and watch it collapse |
| **LOSO zero-shot** | train on all-but-target with no target labels; this is the headline number |
| Few-shot | LOSO plus fine-tuning on k=10/50 target examples, the practical sweet spot |
