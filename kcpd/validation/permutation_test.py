"""
Permutation-based statistical validation.

Validates the hypothesis that segments containing affective events
(e.g., crash, barrel roll) have significantly higher RDI scores than
segments without events.

Algorithm (Section III.F):
  1. Partition segments into two groups:
       - Event segments: contain at least one affective event.
       - Non-event segments: contain no events.
  2. Compute the observed test statistic:
       Δ_obs = mean(RDI_event) − mean(RDI_non-event)
  3. Permutation test (n_perm = 10,000):
       a. Randomly reassign the event/non-event labels.
       b. Recompute Δ under the null hypothesis.
  4. p-value = fraction of permuted Δ values ≥ Δ_obs.

Additionally computes the rank-biserial correlation as a
non-parametric effect size:
    r_rb = 2U / (n₁·n₂) − 1
where U is the Mann-Whitney U statistic.

Reference: Section III.F (Statistical Validation) of the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def permutation_test(
    rdi_scores: np.ndarray,
    is_event: np.ndarray,
    n_perm: int = 10_000,
    seed: int | None = 42,
) -> dict:
    """
    Permutation test for event → RDI association.

    Parameters
    ----------
    rdi_scores : ndarray, shape (n_seg,)
        Consensus RDI score for each segment.
    is_event : ndarray of bool, shape (n_seg,)
        True if the segment contains at least one affective event.
    n_perm : int
        Number of permutations (default 10,000).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    result : dict with keys:
        ``observed_delta`` : float — Δ_obs = mean(event) − mean(non-event)
        ``p_value`` : float — one-sided p-value
        ``null_distribution`` : ndarray — (n_perm,) permuted Δ values
        ``effect_size`` : float — rank-biserial correlation
        ``n_event`` : int — number of event segments
        ``n_nonevent`` : int — number of non-event segments
    """
    rng = np.random.default_rng(seed)

    event_mask = np.asarray(is_event, dtype=bool)
    scores_event = rdi_scores[event_mask]
    scores_nonevent = rdi_scores[~event_mask]

    n_event = len(scores_event)
    n_nonevent = len(scores_nonevent)

    if n_event == 0 or n_nonevent == 0:
        return {
            "observed_delta": 0.0,
            "p_value": 1.0,
            "null_distribution": np.zeros(n_perm),
            "effect_size": 0.0,
            "n_event": n_event,
            "n_nonevent": n_nonevent,
        }

    observed_delta = float(np.mean(scores_event) - np.mean(scores_nonevent))

    # Effect size: rank-biserial correlation via Mann-Whitney U
    u, _ = mannwhitneyu(scores_event, scores_nonevent, alternative="greater")
    effect_size = float(2 * u / (n_event * n_nonevent) - 1)

    # Permutation null distribution
    null_deltas = np.empty(n_perm)
    n_total = len(rdi_scores)

    for i in range(n_perm):
        perm = rng.permutation(n_total)
        perm_event = rdi_scores[perm[:n_event]]
        perm_nonevent = rdi_scores[perm[n_event:]]
        null_deltas[i] = np.mean(perm_event) - np.mean(perm_nonevent)

    p_value = float(np.mean(null_deltas >= observed_delta))

    return {
        "observed_delta": observed_delta,
        "p_value": p_value,
        "null_distribution": null_deltas,
        "effect_size": effect_size,
        "n_event": n_event,
        "n_nonevent": n_nonevent,
    }


def within_subject_z_normalize(
    rdi_scores: np.ndarray,
) -> np.ndarray:
    """
    Within-subject z-normalization of RDI scores.

    Converts raw RDI to z-scores relative to the subject's own
    baseline distribution, enabling cross-subject comparison.

    Parameters
    ----------
    rdi_scores : ndarray, shape (n_seg,)

    Returns
    -------
    z_scores : ndarray, shape (n_seg,)
    """
    mu = np.mean(rdi_scores)
    sigma = np.std(rdi_scores)
    if sigma < 1e-12:
        return np.zeros_like(rdi_scores)
    return (rdi_scores - mu) / sigma
