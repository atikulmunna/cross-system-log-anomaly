# Cross-System Zero-Shot Log Anomaly Detection

This is a log anomaly model that scores a system it has **never seen during
training**. The interesting part isn't scale, it's the generalization trick and
an honest **leave-one-system-out (LOSO)** evaluation that most papers quietly
skip.
