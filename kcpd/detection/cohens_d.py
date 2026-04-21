"""
Cohen's d effect size for pairwise regime comparison.

Computes the multi-feature Root Mean Square (RMS) Cohen's d between
two time-series segments.  For each feature dimension, the standardized
mean difference is computed using the pooled standard deviation:

    d_j = |μ_A^j - μ_B^j| / s_pooled^j

    s_pooled^j = sqrt( ((n_A - 1)·Var(A^j) + (n_B - 1)·Var(B^j)) / (n_A + n_B - 2) )

The final score is the RMS across all features:

    Cohen's d = sqrt( (1/N) · Σ_j  d_j² )

This captures the *magnitude of mean shift* in feature space between
two regimes, complementing DTW (shape) and RuLSIF (distribution).

Reference: Section III.E (Regime Divergence Index — Cohen's d component)
           and Algorithm 1 (Step 5b) of the paper.
"""

from __future__ import annotations

import numpy as np


def cohens_d(seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    """
    Multi-feature RMS Cohen's d between two segments.

    Parameters
    ----------
    seg_a : ndarray, shape (T_a, N)
        Feature matrix for segment A.
    seg_b : ndarray, shape (T_b, N)
        Feature matrix for segment B.

    Returns
    -------
    d_rms : float
        Root-mean-square Cohen's d across all N features.
    """
    N = seg_a.shape[1]
    d_vals = np.zeros(N)

    for j in range(N):
        a = seg_a[:, j]
        b = seg_b[:, j]
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) < 3 or len(b) < 3:
            continue
        pooled = np.sqrt(
            ((len(a) - 1) * np.var(a, ddof=1)
             + (len(b) - 1) * np.var(b, ddof=1))
            / (len(a) + len(b) - 2)
        )
        if pooled > 1e-12:
            d_vals[j] = abs(np.mean(a) - np.mean(b)) / pooled

    return float(np.sqrt(np.mean(d_vals ** 2)))
