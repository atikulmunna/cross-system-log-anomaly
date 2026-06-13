# Cross-System Zero-Shot Log Anomaly Detection: An Honest Benchmark, a Signal Taxonomy, and a Few-Shot Resolution

## Abstract

Log anomaly detection is usually trained and evaluated on a single system, where
reported F1 scores routinely clear 0.99. That's a sign the benchmark is
exhausted, not that the problem is solved. We instead study the more realistic
**leave-one-system-out (LOSO)** setting: train on N−1 systems and detect
anomalies on a held-out system with **no labels from it**. Four findings come out
of this. (1) A learned sequential model does transfer across systems, and broad
multi-system pretraining helps (HDFS ROC 0.65 to 0.77, BGL 0.42 to 0.59). (2) A
**trivial, training-free template-rarity baseline beats the learned model** on 2
of 3 held-out systems, which is a reality check most papers leave out. (3)
Anomalies fall into a **taxonomy** of *contextual*, *point*, and *marked*, and
each type is detected by a different label-free signal (sequential surprise,
template rarity, semantic severity). Each signal is near-perfect on its own type
and noise off-type (for instance, semantic severity reaches ROC 0.9999 on
Thunderbird). (4) Choosing the right signal per system **without labels fails**
(distribution shape is not the same as anomaly relevance), but **5 to 25 labels
per system** fully resolve it, reaching **macro ROC 0.94** and recovering
cross-signal complementarity that no single signal has. Every experiment runs on
a single 8 GB laptop GPU.

---

## 1. Motivation

Here's the production reality: a new service ships with **zero labeled anomaly
history**. Same-system anomaly detection assumes a labeled history that doesn't
exist yet. So the useful question is really about *cross-system transfer*: can a
detector trained on existing systems work on a brand-new one? This is rarely
reported honestly, because the scores collapse. We measure that collapse and then
fix it.

## 2. Data

11 LogHub systems. Three carry anomaly labels and serve as held-out targets, and
eight unlabeled systems form a pretraining pool.

| Role | Systems |
|---|---|
| Labeled targets | HDFS, BGL, Thunderbird |
| Pretraining pool | Zookeeper, Apache, Linux, Mac, HealthApp, Proxifier, OpenSSH, HPC |

| Target | Unit | #Sequences | Anomaly rate | #Templates |
|---|---|---|---|---|
| HDFS | block session | 575,061 | 2.9% | 47 |
| BGL | 100-line window | 94,268 | 10.2% | 1,822 |
| Thunderbird | 100-line window | 99,999 | 46.2%* | 1,040 |

\*Thunderbird is the first 5M-line slice (the full file is 32 GB). Its
window-level rate is inflated by window labeling; the line-level alert rate is
4.5%, clustered in bursts. ROC (base-rate invariant) is the honest metric for
this fold.

## 3. Method

**Shared semantic space.** Raw logs become Drain3 templates, which then go
through a *frozen* `all-MiniLM-L6-v2` embedding (384-d, L2-normalized) cached
once. Identical template strings map to the same vector, so the space is shared
across systems, and that's the prerequisite for transfer. Total vocabulary:
32,550 templates.

**Unit interface.** Every system becomes `(template-sequence, binary label)`.
HDFS uses block sessions (variable length); window systems use fixed 100-line
windows (stride 50). The model and metric are identical regardless of unit.

**Three label-free anomaly signals.**
- **Surprise** (contextual): a small causal Transformer over the sequence of
  *embeddings* predicts the next template's embedding, and the anomaly score is
  the cosine distance between predicted and actual. This is regression, not
  classification, because template ids are not shared across systems but the
  vector space is. Aggregation is keyed on unit type (window goes to max, session
  goes to mean), which is label-free metadata.
- **Rarity** (point): `-log p(template)` from the target's own frequencies, with
  the sequence score being the max rarity. No training.
- **Severity** (marked): max cosine similarity of each template's embedding to a
  set of 15 severity-concept phrases ("fatal error", "kernel panic", and so on),
  with the sequence score being the max over templates. No training.

The surprise Transformer is deliberately tiny: 3 layers, `d_model` 128, 4 heads,
operating on 384-d vectors (~1M parameters), trained 3 epochs with AdamW. A full
LOSO fold is a few GPU-minutes, and the rarity and severity signals need no
training at all.

**Metric.** PR-AUC (primary) and ROC-AUC, macro-averaged across folds. PR-AUC
holds up across the very different base rates, and ROC is base-rate invariant.

## 4. Results

*Provenance.* The surprise model is retrained per fold, and the per-system signal
analyses in §4.2 to §4.5 read the per-sequence scores saved by `ensemble.py`.
Run-to-run ROC variance from random initialization is around 0.02 to 0.04. For
example, the surprise-HDFS fold reads 0.769 in the §4.1 sweep and 0.732 in the
§4.3 run, which is the same model under different seeds. Single-run figures are
reported without confidence intervals; only the few-shot results (§4.5) are
averaged (20 draws).

### 4.1 The learned model transfers, and pretraining breadth helps

LOSO with the surprise Transformer (type-keyed aggregation):

| Target | 3-system pretrain ROC | 11-system pretrain ROC |
|---|---|---|
| HDFS | 0.653 | **0.769** |
| BGL | 0.419 | **0.588** |
| Macro | 0.552 | **0.610** |

Broad pretraining improves transfer, which is evidence for the "foundation"
hypothesis. (Training loss falls cleanly each fold, so the gap is in *signal
transfer*, not optimization.)

### 4.2 A trivial baseline beats the learned model

Zero-shot, training-free template rarity vs. the Transformer (PR / ROC):

| Target | Rarity PR / ROC | Surprise PR / ROC |
|---|---|---|
| HDFS | **0.798 / 0.887** | 0.192 / 0.732 |
| BGL | **0.305 / 0.777** | 0.249 / 0.695 |
| Thunderbird | 0.419 / 0.356 | 0.410 / 0.450 |

A one-line heuristic dominates on HDFS and BGL. HDFS being "easy for rarity",
since anomalous blocks contain distinctive rare events, is consistent with the
literature. Any learned method has to justify itself against this.

### 4.3 Signals are complementary; an anomaly taxonomy

The surprise and rarity signals are **uncorrelated on HDFS** (Spearman −0.03) and
combine almost perfectly:

| Target | corr(surprise,rarity) | surprise | rarity | severity | best fusion |
|---|---|---|---|---|---|
| HDFS | −0.03 | 0.732 | 0.887 | 0.597 | **0.985** (surprise+rarity) |
| BGL | 0.75 | 0.695 | 0.777 | **0.848** | 0.848 (severity) |
| Thunderbird | 0.30 | 0.450 | 0.356 | **0.9999** | 0.9999 (severity) |

(ROC-AUC.) This gives a taxonomy:

| Anomaly type | Example | Signal | Result |
|---|---|---|---|
| Contextual | unusual event sequence | surprise (+ rarity) | HDFS 0.985 |
| Point | rare event template | rarity | HDFS/BGL 0.78 to 0.89 |
| Marked | frequent but textually severe | semantic severity | TB 0.9999, BGL 0.848 |

(Severity PR-AUC: HDFS 0.044, BGL 0.321, TB 0.9999, so noise on HDFS but decisive
on TB.) Thunderbird's dominant alert ("Fatal error (Local Catastrophic Error)")
is *frequent* (4.5% of lines) and *predictable*, so it is invisible to rarity
(not rare) and to surprise (learned away), yet trivially caught by severity.

### 4.4 Negative result: label-free signal selection fails

No fixed fusion wins everywhere, since naive summing dilutes the right signal
(fusing anything into TB's perfect severity drags it down to about 0.75). We
tried to select the right signal per system from each signal's
**score-distribution shape** (tail separation, and 2-component GMM separation).
Both fail:

| Method | Macro ROC |
|---|---|
| Equal-weight sum | 0.809 |
| Adaptive hard-select (GMM) | 0.764 |
| Adaptive soft-weight (GMM) | 0.793 |
| Single-signal oracle | 0.912 |

**Lesson:** a signal can be bimodal for reasons that have nothing to do with
anomalies. HDFS severity has the *highest* GMM separation yet is
anomaly-irrelevant. Distribution shape is not anomaly relevance, and selection is
unsolvable from the unlabeled distribution alone.

A second selector also fails: **cross-system combiner transfer**. The few-shot
result (next section) shows a linear mix of the three signals is what is needed,
so we fit one logistic combiner on the labeled *training* systems (each system's
signals z-scored on its own) and apply it to the held-out target. No target
labels are used, so it is still zero-shot.

| Method | Macro ROC |
|---|---|
| Equal-weight sum | 0.809 |
| Transferred combiner | **0.496** |
| Single-signal oracle | 0.912 |

It does *worse than chance on average*. With only three labeled systems, each
dominated by a different anomaly type, leave-one-out always trains on two types
and tests on the third. The combiner learns the training systems' signal
preference and *anti-transfers* it: trained on {BGL, Thunderbird} it learns
"trust severity, distrust rarity" and applies that to HDFS, whose best signal is
rarity, scoring 0.187 (below random). The learned per-target weights make this
explicit (surprise, rarity, severity):

| Target | transfer ROC | learned weights |
|---|---|---|
| BGL | 0.831 | sev +1.34 (correct) |
| HDFS | 0.187 | sev +3.25, rarity **-0.96** (wrong signal up-weighted) |
| Thunderbird | 0.470 | rarity +1.02, sev +0.25 (under-weights severity) |

So the right combination is genuinely *system-specific*: weights transfer only
between systems of the same anomaly type, which a tiny, type-diverse pool never
provides. This is further evidence that a few target labels (next section), not
clever zero-shot transfer, is the practical fix. A larger pool covering each type
might let transfer work; that is left to future work.

### 4.5 Few-shot resolves it

A logistic combiner over the three z-scored signals, trained on *k* labeled
examples per class per system, evaluated on the remainder (mean of 20 draws):

| Target | zero-shot sum | k=5 | k=10 | k=25 | single-oracle |
|---|---|---|---|---|---|
| BGL | 0.799 | 0.817 | 0.824 | 0.835 | 0.848 |
| HDFS | 0.961 | 0.942 | 0.967 | **0.988** | 0.887 |
| Thunderbird | 0.665 | 0.989 | 0.996 | **0.999** | 1.000 |
| **Macro** | **0.809** | **0.916** | **0.929** | **0.941** | 0.912 |

(ROC-AUC.) Five labeled examples per class already lift macro from 0.81 to 0.92,
and k=25 reaches 0.94. On HDFS the combiner *exceeds* the single-signal oracle by
learning the surprise+rarity complementarity. Thunderbird goes from 0.665 to
0.999 by learning to trust severity from a handful of labels. On the already-easy
HDFS fold, k=5 dips slightly below the fixed sum (0.942 vs 0.961), because with so
few labels the learned weights are noisy, but it overtakes by k=10.

### 4.6 The macro progression

| Regime | Macro ROC |
|---|---|
| Surprise Transformer alone (LOSO) | 0.61 |
| Equal-weight 3-signal sum (zero-shot) | 0.81 |
| Label-free adaptive selection | ≤0.81 (fails) |
| Single-signal oracle (upper bound) | 0.91 |
| **Few-shot k=25** | **0.94** |

### 4.7 Stability across seeds

Re-running the full LOSO ensemble under four random seeds (model initialization
and training-batch order) gives 95% t-interval confidence intervals. Only the
*surprise* signal, and the fusions that use it, varies; rarity and severity are
deterministic, so their CIs are exactly zero by construction (which is the point
of the contrast).

| Signal (macro ROC) | mean +/- 95% CI |
|---|---|
| surprise | 0.609 +/- 0.055 |
| rarity | 0.673 +/- 0.000 |
| severity | 0.815 +/- 0.000 |
| equal-weight sum | **0.809 +/- 0.007** |

The headline equal-weight sum is stable to +/-0.007 because the deterministic
rarity and severity dominate it; the learned surprise signal is the noisy
component, widest on BGL (0.594 +/- 0.144) and smallest on HDFS (0.729 +/-
0.056). So the single-run figures elsewhere in this report are representative,
with the caveat that any surprise-only number carries roughly +/-0.05 to 0.14 of
seed noise. (Reproduce with `ensemble.py --seed N` then `aggregate_seeds.py`.)

## 5. Discussion & limitations

- **BGL is genuinely the hardest** target (best around 0.85); its 1,822 templates
  are far more diverse than HDFS (47) or TB (1,040).
- **Thunderbird** is a 5M-line slice and its window base rate is inflated, so ROC
  is reported as the honest metric.
- **HealthApp** parses poorly under Drain (28k over-fragmented templates) yet did
  not degrade results, so the shared space is robust to a noisy pool member.
- **Seed variance.** The zero-shot signals now carry 95% CIs over four seeds
  (§4.7): the equal-weight sum is stable (+/-0.007) and rarity/severity are
  deterministic, but the surprise signal alone swings by +/-0.05 to 0.14, so any
  surprise-only figure should be read as approximate. Few-shot is separately
  averaged over 20 support draws. Extending to more seeds and adding CIs to the
  few-shot table is straightforward future work.
- **Three labeled folds.** Conclusions rest on HDFS, BGL, and Thunderbird; a
  fourth fold (OpenStack) would strengthen the macro estimates.
- The contribution is a **framework, a benchmark, and honest negatives**, not a
  single SOTA number.

## 6. Reproduction

Environment: Python 3.12, `torch 2.11+cu128` (Blackwell sm_120), `drain3`,
`sentence-transformers`, `scikit-learn`. Single 8 GB GPU; total cost $0.

```
# 1. parse each system (Drain3 templates)
python src/parse.py --system <S> --input data/<S>/<S>.log --out out
# 2. shared embedding cache (frozen MiniLM)
python src/embed.py --out out
# 3. per-system sequences
python src/sequence.py --system <S> --out out [--hdfs-labels data/HDFS/anomaly_label.csv]
# 4. LOSO surprise transformer (GPU)
python src/loso.py --out out --epochs 3
# 5. signals + analysis
python src/rarity_baseline.py --out out
python src/ensemble.py  --out out --epochs 3     # saves out/<S>/scores.npz
python src/severity.py  --out out
python src/adaptive.py  --out out                # negative result
python src/fewshot.py   --out out                # capstone
```

## 7. Conclusion

Cross-system zero-shot log anomaly detection collapses under honest evaluation,
and trivial baselines embarrass the learned models. But anomalies form a small
taxonomy, each type served by a distinct label-free signal, and the open
difficulty is *selecting* among them without labels. That selection fails from
distribution shape but is resolved by 5 to 25 labels per system, reaching macro
ROC 0.94. The practical recipe for a new system: compute three cheap signals,
label a handful of examples, and fit a logistic combiner.

### Future work
- OpenStack as a 4th labeled fold (per-UUID session grouping).
- Scale the pretraining pool (Spark, Hadoop, Windows, Android).
- Learn severity concepts rather than hand-specify them.
- A learned, label-free signal selector that uses anomaly relevance, not just
  distribution shape.
