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
