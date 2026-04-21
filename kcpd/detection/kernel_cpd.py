"""
Kernel-based Change Point Detection (KernelCPD).

Segments multivariate physiological time series into regimes using
a Gaussian RBF kernel in a Reproducing Kernel Hilbert Space (RKHS).

The method solves a penalized cost minimization problem via dynamic
programming.  The penalty parameter β controls the trade-off between
model fit and the number of change points:

    argmin_{t_1,...,t_K}  Σ_k  C(y_{t_k:t_{k+1}})  +  β · K

where C(·) is the kernel-based segment cost computed from the Gram
matrix of the Gaussian RBF kernel:

    k(x, x') = exp(-||x - x'||² / (2σ²))

The bandwidth σ is set via the median heuristic:
    σ = median{ ||x_i - x_j|| : i < j }

Reference: Section III.D (Kernel-based Change Point Detection) and
           Algorithm 1 (Steps 3–4) of the paper.
"""

from __future__ import annotations

from math import floor
import numpy as np
import ruptures as rpt
from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator


class KernelCPD(BaseEstimator):
    """
    Dynamic-programming change point detection with configurable cost model.

    This is a thin wrapper around the ruptures framework, implementing
    the exact penalized segmentation via dynamic programming (DP).

    Parameters
    ----------
    model : str
        Cost model name (default ``"rbf"`` for Gaussian RBF kernel).
    min_size : int
        Minimum segment length in samples (paper: m = 10).
    jump : int
        Grid spacing for candidate change points.
    params : dict or None
        Additional parameters passed to the cost function.
    """

    def __init__(self, model: str = "rbf", min_size: int = 10,
                 jump: int = 5, params: dict | None = None):
        if params is None:
            self.cost = cost_factory(model=model)
        else:
            self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples: int | None = None

    def _seg(self, pen: float) -> dict:
        """Run penalized dynamic programming segmentation."""
        partitions = {0: {(0, 0): 0}}
        admissible: list[int] = []

        ind = [k for k in range(0, self.n_samples, self.jump)
               if k >= self.min_size]
        ind += [self.n_samples]

        for bkp in ind:
            new_adm_pt = floor((bkp - self.min_size) / self.jump) * self.jump
            admissible.append(new_adm_pt)

            subproblems = []
            for t in admissible:
                try:
                    tmp = partitions[t].copy()
                except KeyError:
                    continue
                tmp[(t, bkp)] = self.cost.error(t, bkp) + pen
                subproblems.append(tmp)

            partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            admissible = [
                t for t, p in zip(admissible, subproblems)
                if sum(p.values()) <= sum(partitions[bkp].values()) + pen
            ]

        best = partitions[self.n_samples]
        del best[(0, 0)]
        return best

    def fit(self, signal: np.ndarray) -> "KernelCPD":
        """Fit the cost model on the input signal."""
        self.cost.fit(signal)
        self.n_samples = signal.shape[0] if signal.ndim > 1 else len(signal)
        return self

    def predict(self, pen: float) -> list[int]:
        """Predict change points for a given penalty."""
        partition = self._seg(pen)
        return sorted(e for _, e in partition.keys())

    def fit_predict(self, signal: np.ndarray, pen: float) -> list[int]:
        """Fit and predict in one call."""
        return self.fit(signal).predict(pen)


# ═══════════════════════════════════════════════════════════════════════════════
#  Convenience wrappers matching paper parameters
# ═══════════════════════════════════════════════════════════════════════════════

def segment_signal(features: np.ndarray, penalty: float = 1.0,
                   min_size: int = 10) -> list[int]:
    """
    Segment a multivariate feature matrix using KernelCPD.

    This is the primary entry point used by the pipeline
    (Algorithm 1, Step 3).

    Parameters
    ----------
    features : ndarray, shape (T, N)
        Windowed feature matrix (T time steps, N features).
    penalty : float
        Penalty β controlling number of change points.
        Paper default: β = 1.0.
    min_size : int
        Minimum segment length m in samples.
        Paper default: m = 10.

    Returns
    -------
    boundaries : list of int
        Sorted change point indices (excluding T).
    """
    algo = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(features)
    cps = algo.predict(pen=penalty)
    return [c for c in cps if c < len(features)]


def multi_scale_cpd(features: np.ndarray,
                    penalties: tuple[float, ...] = (5, 10, 20),
                    min_size: int = 5) -> dict[float, list[int]]:
    """
    Run KernelCPD at multiple penalty levels for multi-scale analysis.

    Lower penalty → more change points (higher sensitivity).
    Higher penalty → fewer change points (more conservative).

    Parameters
    ----------
    features : ndarray, shape (T, N)
    penalties : tuple of float
    min_size : int

    Returns
    -------
    dict : penalty → list of CP indices
    """
    algo = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(features)
    return {pen: [c for c in algo.predict(pen=pen) if c < len(features)]
            for pen in penalties}
