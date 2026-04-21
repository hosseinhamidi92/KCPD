"""
Dynamic Time Warping (DTW) distance.

Computes the DTW alignment distance between two multivariate time
series segments.  Used as one of three pairwise regime comparison
metrics in the Regime Divergence Index (RDI) computation.

DTW captures *shape similarity* between segments of potentially
different lengths by finding the optimal temporal alignment.

Reference: Section III.E (Regime Divergence Index — DTW component)
           and Algorithm 1 (Step 5a) of the paper.
"""

from __future__ import annotations

import numpy as np
from numpy import zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf


def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Compute Dynamic Time Warping distance between two sequences.

    Parameters
    ----------
    x : array-like, shape (N1,) or (N1, M)
        First time series.
    y : array-like, shape (N2,) or (N2, M)
        Second time series.
    dist : callable
        Pairwise distance function ``dist(x_i, y_j) -> float``.
    warp : int
        Number of diagonal warping shifts (default 1).
    w : float
        Sakoe-Chiba band width (``inf`` for no constraint).
    s : float
        Slope weight for off-diagonal moves (≥1 biases toward diagonal).

    Returns
    -------
    distance : float
        Minimum DTW distance.
    cost : ndarray
        Local cost matrix.
    acc_cost : ndarray
        Accumulated cost matrix.
    path : tuple of arrays
        Optimal warping path (row_indices, col_indices).
    """
    assert len(x) and len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    r, c = len(x), len(y)

    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf

    D1 = D0[1:, 1:]
    for i in range(r):
        for j in range(c):
            if isinf(w) or (max(0, i - w) <= j <= min(c, i + w)):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()

    for i in range(r):
        jrange = range(c) if isinf(w) else range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j] * s,
                             D0[i, min(j + k, c)] * s]
            D1[i, j] += min(min_list)

    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)

    return D1[-1, -1], C, D1, path


def accelerated_dtw(x, y, dist="euclidean", warp=1):
    """
    Fast DTW using scipy.spatial.distance.cdist for batch distance computation.

    This is the variant used in the pipeline for computing pairwise
    segment distances (Algorithm 1, Step 5a).

    Parameters
    ----------
    x : array-like, shape (N1,) or (N1, M)
        First time series.
    y : array-like, shape (N2,) or (N2, M)
        Second time series.
    dist : str or callable
        Distance metric for ``cdist`` (default ``"euclidean"``).
    warp : int
        Warping factor.

    Returns
    -------
    distance : float
        Minimum DTW distance.
    cost : ndarray
        Local cost matrix.
    acc_cost : ndarray
        Accumulated cost matrix.
    path : tuple of arrays
        Optimal warping path.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    assert len(x) and len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)

    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()

    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)

    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)

    return D1[-1, -1], C, D1, path


def _traceback(D):
    """Trace back the optimal warping path from the accumulated cost matrix."""
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while i > 0 or j > 0:
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)
