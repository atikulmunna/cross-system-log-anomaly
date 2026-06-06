"""Label-free adaptive signal selection.

Each signal's own score distribution tells us if it found something: a signal
with real anomalies has a heavy upper tail far beyond its normal spread; a noise
signal is bland/unimodal. We measure tail separation (p99-p50)/IQR per signal
(NO labels) and use it to (a) hard-select the best signal and (b) soft-weight a
combination. Goal: close the fixed-fusion (~0.81) -> oracle (~0.94) macro gap.

Usage:
  python src/adaptive.py --out out
"""
import argparse

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sentence_transformers import SentenceTransformer

from data import labeled_systems, load_sequences
from severity import CONCEPTS


def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def tail_sep(x):
    """Heavy-tail separation: (p99 - p50) / IQR. High when a signal has a small
    set of extreme outliers (candidate anomalies) above its bulk."""
    p99, p50, p25, p75 = np.percentile(x, [99, 50, 25, 75])
    return (p99 - p50) / (p75 - p25 + 1e-9)
