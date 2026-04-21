"""
RuLSIF — Relative Unconstrained Least-Squares Importance Fitting.

Estimates the Pearson divergence between two sample sets by fitting
a density ratio model using Gaussian kernels.  The bandwidth is
selected via the median heuristic.

Given reference samples X_de and test samples X_nu, RuLSIF estimates:

    r̂(x) = p_nu(x) / (α·p_nu(x) + (1-α)·p_de(x))

and returns the Pearson divergence:

    PE = E_nu[r̂(x)] − 0.5·(α·E_nu[r̂²(x)] + (1−α)·E_de[r̂²(x)]) − 0.5

Higher values indicate greater distributional shift between the two
segments, suggesting a regime transition.

Based on: Liu et al. (2013) — "Change-point detection in time-series data
by relative density-ratio estimation."

Reference: Section III.E (Regime Divergence Index — RuLSIF component)
           and Algorithm 1 (Step 5c) of the paper.
"""

from __future__ import annotations

import numpy as np


def _comp_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance matrix between columns of x and y."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    Gx = np.sum(x * x, axis=0)
    Gy = np.sum(y * y, axis=0)
    return Gx[:, None] + Gy[None, :] - 2 * x.T @ y


def _comp_med(x: np.ndarray) -> float:
    """Median heuristic for Gaussian kernel bandwidth σ."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    _, n = x.shape
    G = np.sum(x * x, axis=0)
    T_mat = np.tile(G, (n, 1))
    dist2 = T_mat - 2 * x.T @ x + T_mat.T
    dist2 -= np.tril(dist2)
    R = dist2.ravel()
    return float(np.sqrt(0.5 * np.median(R[R > 0])))


def _kernel_gau(dist2: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel: exp(-dist² / (2σ²))."""
    return np.exp(-dist2 / (2 * sigma ** 2))


def rulsif(x_de: np.ndarray, x_nu: np.ndarray,
           alpha: float = 0.5) -> float:
    """
    RuLSIF divergence estimate (fixed σ and λ for speed).

    Parameters
    ----------
    x_de : ndarray, shape (d, n_de)
        Reference ("denominator") samples — columns are observations.
    x_nu : ndarray, shape (d, n_nu)
        Test ("numerator") samples — columns are observations.
    alpha : float
        Mixing parameter α ∈ [0, 1].  Default 0.5.

    Returns
    -------
    rPE : float
        Pearson divergence estimate.  Higher = more distributional shift.
    """
    if x_nu.ndim == 1:
        x_nu = x_nu.reshape(1, -1)
    if x_de.ndim == 1:
        x_de = x_de.reshape(1, -1)

    _, n_nu = x_nu.shape
    _, n_de = x_de.shape

    # Select kernel centers from test set
    b = min(100, n_nu)
    idx = np.random.permutation(n_nu)
    x_ce = x_nu[:, idx[:b]]
    n_ce = x_ce.shape[1]

    # Median heuristic for bandwidth
    x_all = np.concatenate((x_nu, x_de), axis=1)
    med = _comp_med(x_all)
    sigma_list = med * np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    lambda_list = 10.0 ** np.arange(-3, 2)

    dist2_de = _comp_dist(x_de, x_ce)
    dist2_nu = _comp_dist(x_nu, x_ce)

    # Use fixed middle σ and λ (no cross-validation — much faster)
    sigma = sigma_list[2]
    lam = lambda_list[2]

    k_de = _kernel_gau(dist2_de, sigma).T  # (n_ce, n_de)
    k_nu = _kernel_gau(dist2_nu, sigma).T  # (n_ce, n_nu)

    # Solve for importance weights θ
    H = ((1 - alpha) / n_de) * k_de @ k_de.T + (alpha / n_nu) * k_nu @ k_nu.T
    h = np.mean(k_nu, axis=1)
    theta = np.linalg.solve(H + np.eye(n_ce) * lam, h)

    # Pearson divergence estimate
    g_nu = theta @ k_nu
    g_de = theta @ k_de
    rPE = (np.mean(g_nu)
           - 0.5 * (alpha * np.mean(g_nu ** 2)
                     + (1 - alpha) * np.mean(g_de ** 2))
           - 0.5)
    return float(rPE)


def sliding_rulsif(features: np.ndarray,
                   n_half_window: int = 25,
                   alpha: float = 0.5) -> np.ndarray:
    """
    Sliding-window RuLSIF change detection score.

    At each time step t, compares the feature distribution in
    [t−n, t) vs [t, t+n) using RuLSIF divergence.

    Parameters
    ----------
    features : ndarray, shape (T, N)
        Feature matrix (T time steps, N features).
    n_half_window : int
        Half-window size.
    alpha : float
        RuLSIF mixing parameter.

    Returns
    -------
    scores : ndarray, shape (T,)
        RuLSIF score at each time step.  NaN at edges where the
        window extends beyond the signal.
    """
    T, N = features.shape
    X = features.T  # (N, T)
    scores = np.full(T, np.nan)
    n = n_half_window

    for t in range(n, T - n):
        Y = X[:, t - n:t + n]
        stds = np.std(Y, axis=1, keepdims=True)
        stds[stds == 0] = 1.0
        Y = Y / np.tile(stds, (1, 2 * n))
        ref = Y[:, n:]   # second half (reference)
        test = Y[:, :n]  # first half (test)
        scores[t] = rulsif(ref, test, alpha)

    return scores
