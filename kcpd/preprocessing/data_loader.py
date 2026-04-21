"""
Data loading and normalization utilities.

Handles Biopac .txt file parsing and signal-level normalization.

Reference: Section II (Data Collection) of the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


class DataProcessor:
    """Parse Biopac .txt data files into per-channel arrays."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def process_file(self, lines_to_skip: int = 20):
        """
        Read a Biopac text export, skipping header lines.

        Returns
        -------
        list of (channel_index, length, data_list)
        """
        columns: list[list[float]] = []
        max_columns = 0

        with open(self.file_path, "r") as fh:
            for _ in range(lines_to_skip):
                next(fh, None)
            for line in fh:
                row = [float(c) for c in line.split()]
                max_columns = max(max_columns, len(row))
                while len(columns) < max_columns:
                    columns.append([])
                for i, val in enumerate(row):
                    columns[i].append(val)
                for i in range(len(row), max_columns):
                    columns[i].append(None)

        return [(i, len(col), col) for i, col in enumerate(columns)]


class SkinTempProcessor:
    """Low-pass Butterworth filter for skin temperature signals."""

    def __init__(self, cutoff: float = 0.1, fs: int = 4):
        self.cutoff = cutoff
        self.fs = fs

    def process_skin_temp_signal(self, signal: np.ndarray) -> np.ndarray:
        nyquist = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyquist
        b, a = butter(5, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, signal)


class DataNormalizer:
    """Min-max and z-score normalization."""

    @staticmethod
    def normalize(data: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]."""
        mn, mx = np.min(data), np.max(data)
        rng = mx - mn
        return (data - mn) / rng if rng > 0 else np.zeros_like(data)

    @staticmethod
    def standardize(data: np.ndarray) -> np.ndarray:
        """Z-score standardization (mean=0, std=1)."""
        mu, sigma = np.mean(data), np.std(data)
        return (data - mu) / sigma if sigma > 0 else np.zeros_like(data)


def downsample_signal(signal: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Downsample by integer factor (point decimation)."""
    factor = fs_in // fs_out
    if factor <= 1:
        return np.asarray(signal)
    return np.asarray(signal[::factor])
