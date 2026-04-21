"""
Regime Divergence Index (RDI) — consensus outlier scoring.

The RDI combines three complementary pairwise comparison metrics
(DTW distance, RuLSIF divergence, Cohen's d) into a single
consensus score that quantifies how "different" each regime segment
is from all others.

Algorithm (Section III.E, Algorithm 1 Steps 5–7):
  1. For each pair of segments (i, j), compute:
       - DTW distance   (shape similarity)
       - RuLSIF score   (distributional shift)
       - Cohen's d      (mean shift magnitude)
  2. For each segment i, average its distance to all other segments.
  3. Normalize each metric to [0, 1].
  4. Average the three normalized scores → consensus RDI.

Segments with high RDI are "outlier regimes" — physiological states
that differ substantially from baseline, likely driven by external
events (e.g., driving hazards).

Reference: Section III.E (Regime Divergence Index) and
           Algorithm 1 (Steps 5–7) of the paper.
"""

from __future__ import annotations

import numpy as np

from .dtw import accelerated_dtw
from .rulsif import rulsif
from .cohens_d import cohens_d


# ═══════════════════════════════════════════════════════════════════════════════
#  Segment construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_segments(features: np.ndarray,
                   boundaries: list[int]) -> list[dict]:
    """
    Split the feature matrix into segments defined by CP boundaries.

    Parameters
    ----------
    features : ndarray, shape (T, N)
    boundaries : list of int — sorted CP indices

    Returns
    -------
    segments : list of dict
        Each dict has keys: ``data``, ``start``, ``end``, ``mean``, ``std``.
    """
    T = features.shape[0]
    edges = [0] + sorted(boundaries) + [T]
    segments = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        seg_data = features[s:e]
        segments.append({
            "data": seg_data,
            "start": s,
            "end": e,
            "mean": np.nanmean(seg_data, axis=0),
            "std": np.nanstd(seg_data, axis=0),
        })
    return segments


# ═══════════════════════════════════════════════════════════════════════════════
#  Pairwise comparison
# ═══════════════════════════════════════════════════════════════════════════════

def _seg_dtw_distance(seg_a: dict, seg_b: dict) -> float:
    """DTW distance normalized by path length."""
    d, _, _, _ = accelerated_dtw(seg_a["data"], seg_b["data"], "euclidean")
    norm = max(len(seg_a["data"]), len(seg_b["data"]))
    return d / norm if norm > 0 else 0.0


def _seg_rulsif_divergence(seg_a: dict, seg_b: dict,
                            alpha: float = 0.5) -> float:
    """RuLSIF divergence between two segment distributions."""
    x_a = seg_a["data"].T  # (N, T_a)
    x_b = seg_b["data"].T
    if x_a.shape[1] < 5 or x_b.shape[1] < 5:
        return 0.0
    try:
        return max(0.0, rulsif(x_a, x_b, alpha=alpha))
    except Exception:
        return 0.0


def _seg_cohens_d(seg_a: dict, seg_b: dict) -> float:
    """RMS Cohen's d across all features."""
    return cohens_d(seg_a["data"], seg_b["data"])


def compute_pairwise_distances(
    segments: list[dict],
) -> dict[str, np.ndarray]:
    """
    Compute pairwise distance matrices for all segment pairs.

    Three methods are used:
      - ``DTW``: Dynamic time warping distance (shape)
      - ``RuLSIF``: Density-ratio divergence (distribution)
      - ``Cohen's d``: Standardized mean difference (effect size)

    Parameters
    ----------
    segments : list of dict
        Output of :func:`build_segments`.

    Returns
    -------
    dist_matrices : dict of str → ndarray (n_seg, n_seg)
        Symmetric pairwise distance matrices.
    """
    n = len(segments)
    methods = {
        "DTW": _seg_dtw_distance,
        "RuLSIF": _seg_rulsif_divergence,
        "Cohen's d": _seg_cohens_d,
    }
    dist_matrices = {name: np.zeros((n, n)) for name in methods}

    for i in range(n):
        for j in range(i + 1, n):
            for name, func in methods.items():
                d = func(segments[i], segments[j])
                dist_matrices[name][i, j] = d
                dist_matrices[name][j, i] = d

    return dist_matrices


# ═══════════════════════════════════════════════════════════════════════════════
#  Consensus RDI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rdi(dist_matrices: dict[str, np.ndarray],
                n_segments: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Compute the Regime Divergence Index (RDI) for each segment.

    Algorithm:
      1. For each method, compute each segment's mean distance to all others.
      2. Normalize each method's scores to [0, 1].
      3. Average across methods → consensus RDI.

    Parameters
    ----------
    dist_matrices : dict of str → (n_seg, n_seg) arrays
    n_segments : int

    Returns
    -------
    per_method : dict of str → (n_seg,) arrays
        Unnormalized per-method mean distances.
    consensus : ndarray, shape (n_seg,)
        Final RDI score in [0, 1].
    """
    per_method = {}
    for name, dm in dist_matrices.items():
        avg_dist = np.sum(dm, axis=1) / max(1, n_segments - 1)
        per_method[name] = avg_dist

    # Normalize each to [0, 1] and average
    norm_scores = np.zeros((len(per_method), n_segments))
    for mi, scores in enumerate(per_method.values()):
        smax = scores.max()
        norm_scores[mi] = scores / smax if smax > 0 else scores

    consensus = np.mean(norm_scores, axis=0)
    return per_method, consensus


def map_segments_to_events(
    segments: list[dict],
    time_vec: np.ndarray,
    event_times: np.ndarray,
    event_names: list[str],
) -> list[list[str]]:
    """
    For each segment, identify which known events fall within it.

    Parameters
    ----------
    segments : list of dict — from :func:`build_segments`
    time_vec : (T,) — window center times in seconds
    event_times : array of float — event timestamps
    event_names : list of str — event labels

    Returns
    -------
    seg_events : list of list of str
        Events contained in each segment.
    """
    seg_events = []
    for seg in segments:
        t_start = time_vec[seg["start"]]
        t_end = time_vec[min(seg["end"] - 1, len(time_vec) - 1)]
        events_in = [en for et, en in zip(event_times, event_names)
                     if t_start <= et <= t_end]
        seg_events.append(events_in)
    return seg_events
