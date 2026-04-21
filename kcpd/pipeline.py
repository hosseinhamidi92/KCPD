"""
End-to-end pipeline — Algorithm 1 from the paper.

Orchestrates the full unsupervised regime divergence detection:

  1. **Load & preprocess** — Read raw physiological signals, downsample,
     denoise, and compute per-modality Signal Quality Indices (SQI).
  2. **Feature extraction** — Sliding window (W=30 s, Δ=1 s) over five
     modalities to produce a (T×19) feature matrix.
  3. **KernelCPD segmentation** — Gaussian RBF kernel change point
     detection with penalty β=1 and minimum segment length m=10.
  4. **Build segments** — Split the feature matrix at detected CPs.
  5. **Pairwise comparison** — For every segment pair, compute:
       (a) DTW distance        — shape similarity
       (b) RuLSIF divergence   — distributional shift
       (c) Cohen's d           — mean shift magnitude
  6. **Consensus RDI** — Normalize each metric to [0,1], average to
     obtain the Regime Divergence Index.
  7. **Within-subject z-normalization** — Convert RDI to z-scores for
     cross-subject comparison.
  8. **Statistical validation** — Permutation test (n=10,000) comparing
     event vs. non-event segment RDI scores.

Reference: Algorithm 1 and Section III of the paper.

Usage
-----
::

    from kcpd.pipeline import run_pipeline

    results = run_pipeline(
        ecg, ppg, resp, eda, temp,
        event_times=[120.0, 250.0, 380.0],
        event_names=["Crash", "Barrel", "Brake"],
    )

    # results["consensus_rdi"]  — (n_seg,) RDI scores
    # results["z_rdi"]          — z-normalized RDI
    # results["permutation"]    — dict with p_value, effect_size, etc.
"""

from __future__ import annotations

import numpy as np

from .detection.kernel_cpd import segment_signal
from .detection.rdi import build_segments, compute_pairwise_distances, compute_rdi
from .detection.rdi import map_segments_to_events
from .validation.permutation_test import permutation_test, within_subject_z_normalize


def run_pipeline(
    ecg: np.ndarray,
    ppg: np.ndarray,
    resp: np.ndarray,
    eda: np.ndarray,
    temp: np.ndarray,
    ecg_fs: int = 200,
    ppg_fs: int = 200,
    resp_fs: int = 100,
    eda_fs: int = 4,
    temp_fs: int = 4,
    window_sec: int = 30,
    step_sec: int = 1,
    penalty: float = 1.0,
    min_size: int = 10,
    event_times: list[float] | None = None,
    event_names: list[str] | None = None,
    affective_events: set[str] | None = None,
    n_perm: int = 10_000,
) -> dict:
    """
    Run the full KCPD pipeline (Algorithm 1).

    Parameters
    ----------
    ecg, ppg, resp, eda, temp : 1-D arrays
        Raw physiological signals at their native sampling rates.
    ecg_fs, ppg_fs, resp_fs, eda_fs, temp_fs : int
        Sampling rates in Hz.
    window_sec : int
        Feature extraction window length (seconds).  Default: 30.
    step_sec : int
        Feature extraction step size (seconds).  Default: 1.
    penalty : float
        KernelCPD penalty β.  Default: 1.0.
    min_size : int
        Minimum segment length m.  Default: 10.
    event_times : list of float or None
        Known event timestamps (seconds from recording start).
    event_names : list of str or None
        Labels for each event.
    affective_events : set of str or None
        Which event names count as "affective" for validation.
        Default: {"Crash", "Barrel", "Brake"}.
    n_perm : int
        Number of permutation test iterations.

    Returns
    -------
    dict with keys:
        ``features`` : FeatureResult — extracted features
        ``boundaries`` : list of int — CP indices
        ``segments`` : list of dict — segment data
        ``dist_matrices`` : dict — pairwise distances per method
        ``per_method_rdi`` : dict — per-method mean distances
        ``consensus_rdi`` : ndarray — consensus RDI per segment
        ``z_rdi`` : ndarray — z-normalized RDI
        ``seg_events`` : list of list of str — events per segment
        ``permutation`` : dict — permutation test results
    """
    from .preprocessing.feature_extraction import extract_features

    if affective_events is None:
        affective_events = {"Crash", "Barrel", "Brake"}
    if event_times is None:
        event_times = []
    if event_names is None:
        event_names = []

    # Step 1–2: Feature extraction (Algorithm 1, lines 1–2)
    feat_result = extract_features(
        ecg, ppg, resp, eda, temp,
        ecg_fs=ecg_fs, ppg_fs=ppg_fs, resp_fs=resp_fs,
        eda_fs=eda_fs, temp_fs=temp_fs,
        window_sec=window_sec, step_sec=step_sec,
    )
    features = feat_result.matrix
    time_vec = feat_result.time_vec

    # Step 3: KernelCPD segmentation (Algorithm 1, lines 3–4)
    boundaries = segment_signal(features, penalty=penalty, min_size=min_size)

    # Step 4: Build segments
    segments = build_segments(features, boundaries)
    n_seg = len(segments)

    # Step 5: Pairwise comparison (Algorithm 1, lines 5a–5c)
    dist_matrices = compute_pairwise_distances(segments)

    # Step 6: Consensus RDI (Algorithm 1, lines 6–7)
    per_method, consensus = compute_rdi(dist_matrices, n_seg)

    # Step 7: Within-subject z-normalization
    z_rdi = within_subject_z_normalize(consensus)

    # Event mapping
    seg_events = map_segments_to_events(
        segments, time_vec, np.asarray(event_times), event_names)

    # Step 8: Permutation test (Section III.F)
    is_event = np.array([
        any(e in affective_events for e in evts)
        for evts in seg_events
    ])
    perm_result = permutation_test(consensus, is_event, n_perm=n_perm)

    return {
        "features": feat_result,
        "boundaries": boundaries,
        "segments": segments,
        "dist_matrices": dist_matrices,
        "per_method_rdi": per_method,
        "consensus_rdi": consensus,
        "z_rdi": z_rdi,
        "seg_events": seg_events,
        "permutation": perm_result,
    }
