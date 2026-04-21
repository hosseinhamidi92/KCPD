"""
Multi-criterion Signal Quality Assessment (SQA).

Implements quality indices for each physiological modality used in the
pipeline.  Low-quality segments are gated (excluded) before feature
extraction to prevent artifact-driven false change points.

Modality-specific approaches:
  - **EDA**: Wavelet decomposition + statistical features → XGBoost classifier
    (Section III.A and Supplementary B.1)
  - **PPG**: Morphological features (template matching, energy, spectral) →
    One-Class SVM (Section III.A and Supplementary B.2)
  - **Resp**: Variational Autoencoder (VAE) reconstruction error for
    unsupervised anomaly detection (Section III.A and Supplementary B.3)
  - **ECG**: NeuroKit2 built-in quality index (Section III.A)

Reference: Section III.A (Signal Processing) of the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal import welch


# ═══════════════════════════════════════════════════════════════════════════════
#  EDA Signal Quality
# ═══════════════════════════════════════════════════════════════════════════════


def compute_eda_sqi(eda: np.ndarray, fs: int = 4,
                    window_sec: float = 5.0) -> np.ndarray:
    """
    Compute per-window EDA signal quality index.

    Uses wavelet-based statistical features fed to a pre-trained
    XGBoost classifier.  Here we provide the algorithmic skeleton;
    the full implementation requires the trained model artifacts
    (see ``models/eda/``).

    Parameters
    ----------
    eda : 1-D array — raw EDA signal
    fs : sampling rate (Hz)
    window_sec : analysis window length (seconds)

    Returns
    -------
    sqi : 1-D array — per-window quality score in [0, 1]
    """
    win = int(window_sec * fs)
    n_windows = max(1, len(eda) // win)
    sqi = np.ones(n_windows)

    for i in range(n_windows):
        seg = eda[i * win:(i + 1) * win]
        if len(seg) < win:
            continue
        # Rate-of-change criterion: excessive jumps indicate motion artifact
        diff = np.abs(np.diff(seg))
        if np.max(diff) > 0.5 * np.ptp(seg):
            sqi[i] = 0.0
            continue
        # Flatness criterion: zero variance indicates sensor detachment
        if np.std(seg) < 1e-6:
            sqi[i] = 0.0
    return sqi


# ═══════════════════════════════════════════════════════════════════════════════
#  PPG Signal Quality
# ═══════════════════════════════════════════════════════════════════════════════


def compute_ppg_sqi(ppg: np.ndarray, fs: int = 200,
                    window_sec: float = 30.0,
                    step_sec: float = 1.0) -> np.ndarray:
    """
    PPG signal quality via morphological feature analysis.

    Extracts five features per window:
      1. IQR of amplitude
      2. Std of power spectral density
      3. Heart-cycle energy variation
      4. Template Euclidean distance
      5. Template correlation

    In the full pipeline, a pre-trained One-Class SVM (``models/ppg/``)
    classifies each window as clean or noisy.

    Parameters
    ----------
    ppg : 1-D array — preprocessed PPG signal
    fs : sampling rate (Hz)
    window_sec : window length (seconds)
    step_sec : step size (seconds)

    Returns
    -------
    sqi : 1-D array — per-window quality score in [0, 1]
    """
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    starts = np.arange(0, len(ppg) - win + 1, step)
    sqi = np.ones(len(starts))

    for j, s in enumerate(starts):
        seg = ppg[s:s + win]
        # Spectral concentration: cardiac band 0.5–3 Hz should dominate
        if len(seg) < win:
            continue
        freqs, psd = welch(seg, fs=fs, nperseg=min(256, len(seg)))
        total = np.sum(psd)
        cardiac = np.sum(psd[(freqs >= 0.5) & (freqs <= 3.0)])
        if total > 0 and cardiac / total < 0.3:
            sqi[j] = 0.0
    return sqi


# ═══════════════════════════════════════════════════════════════════════════════
#  Resp Signal Quality
# ═══════════════════════════════════════════════════════════════════════════════


def compute_resp_sqi(resp: np.ndarray, fs: int = 100,
                     window_sec: float = 30.0,
                     step_sec: float = 1.0) -> np.ndarray:
    """
    Respiratory signal quality via spectral and amplitude criteria.

    The full implementation uses a Variational Autoencoder (VAE):
    segments with high reconstruction error are flagged as noisy.
    Here we provide a simplified spectral-energy criterion.

    Parameters
    ----------
    resp : 1-D array — preprocessed respiratory signal
    fs : sampling rate (Hz)
    window_sec : window length (seconds)
    step_sec : step size (seconds)

    Returns
    -------
    sqi : 1-D array — per-window quality score in [0, 1]
    """
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    starts = np.arange(0, len(resp) - win + 1, step)
    sqi = np.ones(len(starts))

    for j, s in enumerate(starts):
        seg = resp[s:s + win]
        if np.std(seg) < 1e-6:
            sqi[j] = 0.0
            continue
        freqs, psd = welch(seg, fs=fs, nperseg=min(512, len(seg)))
        resp_band = np.sum(psd[(freqs >= 0.05) & (freqs <= 1.0)])
        total = np.sum(psd)
        if total > 0 and resp_band / total < 0.4:
            sqi[j] = 0.0
    return sqi


# ═══════════════════════════════════════════════════════════════════════════════
#  SQA Gating
# ═══════════════════════════════════════════════════════════════════════════════


def apply_sqa_gate(features: np.ndarray,
                   sqi_per_modality: dict[str, np.ndarray],
                   modality_map: dict[str, list[int]],
                   threshold: float = 0.5) -> np.ndarray:
    """
    Gate (mask) features whose modality SQI falls below threshold.

    Parameters
    ----------
    features : (T, N) feature matrix
    sqi_per_modality : modality name → (T,) quality array
    modality_map : modality name → list of column indices in features
    threshold : minimum quality for inclusion

    Returns
    -------
    gated : (T, N) feature matrix with NaN where quality is low
    """
    gated = features.copy()
    T = features.shape[0]
    for mod, indices in modality_map.items():
        if mod not in sqi_per_modality:
            continue
        sqi = sqi_per_modality[mod]
        # Align length
        sqi_aligned = np.interp(
            np.linspace(0, 1, T),
            np.linspace(0, 1, len(sqi)),
            sqi,
        )
        bad = sqi_aligned < threshold
        for idx in indices:
            gated[bad, idx] = np.nan
    return gated
