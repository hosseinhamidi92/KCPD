#!/usr/bin/env python3
"""
Signal Quality Index (SQI) 2×2 panel figure.

Panels: EDA, ECG, PPG, Resp — each with raw signal, SQI overlay, and
zoomed inset highlighting an artefact transition.

Requires: Biopac .txt data, timestamps Excel.

Usage:
    MPLBACKEND=Agg python polished/fig_sqi.py [--subject P4S]
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    import scienceplots  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurokit2 as nk
from scipy.signal import welch, hilbert
from scipy.stats import kurtosis as _kurtosis
from Load.LoadData import DataProcessor, DataNormalizer
from Utils.func import SignalProcessor
from PreProcessing import EDA_SQA

# ── SQA helpers ──────────────────────────────────────────────────────────

def _flag_flatline(sig, fs, min_duration_sec=2.0, std_threshold=1e-6):
    win = int(min_duration_sec * fs)
    if win < 2 or len(sig) < win:
        return np.zeros(len(sig), dtype=bool)
    flags = np.zeros(len(sig), dtype=bool)
    for start in range(0, len(sig) - win + 1, max(1, win // 2)):
        if np.nanstd(sig[start:start + win]) < std_threshold:
            flags[start:start + win] = True
    return flags


def _flag_clipping(sig, percentile_margin=0.5):
    valid = sig[~np.isnan(sig)] if np.isnan(sig).any() else sig
    if len(valid) < 10:
        return np.zeros(len(sig), dtype=bool)
    lo, hi = np.percentile(valid, [percentile_margin, 100 - percentile_margin])
    return (sig <= lo) | (sig >= hi)


def _rolling_amplitude_bad(sig, fs, win_sec=5.0, z_thresh=3.0):
    n, win = len(sig), int(win_sec * fs)
    step = max(1, win // 2)
    starts = list(range(0, n - win + 1, step))
    ranges = np.array([np.ptp(sig[s:s + win]) for s in starts])
    if len(ranges) < 3:
        return np.zeros(n, dtype=bool)
    med = np.median(ranges)
    mad = np.median(np.abs(ranges - med)) * 1.4826
    if mad < 1e-10:
        return np.zeros(n, dtype=bool)
    bad = np.zeros(n, dtype=bool)
    for i, s in enumerate(starts):
        if (ranges[i] - med) / mad > z_thresh:
            bad[s:s + win] = True
    return bad


def _ecg_sqi(ecg_fn, ecg_fs):
    q = nk.ecg_quality(ecg_fn, sampling_rate=ecg_fs)
    return (~((q < 0.7) | _flag_flatline(ecg_fn, ecg_fs))).astype(float)


def _ppg_sqi(ppg_fn, ppg_raw, ppg_fs):
    q = nk.ppg_quality(ppg_fn, sampling_rate=ppg_fs, method="templatematch")
    bad = (q < np.mean(q) * 0.8) | _rolling_amplitude_bad(ppg_fn, ppg_fs) | \
          _flag_flatline(ppg_fn, ppg_fs) | _flag_clipping(ppg_raw)
    return (~bad).astype(float)


def _resp_sqi(resp_fn, resp_fs):
    n = len(resp_fn)
    win_sp = int(10 * resp_fs)
    step_sp = int(1 * resp_fs)
    bad_spec = np.zeros(n, dtype=bool)
    for s in range(0, n - win_sp + 1, step_sp):
        chunk = resp_fn[s:s + win_sp]
        if np.nanstd(chunk) < 1e-10:
            bad_spec[s:s + win_sp] = True
            continue
        freqs, psd = welch(chunk, fs=resp_fs, nperseg=min(len(chunk), int(4 * resp_fs)))
        total = np.trapz(psd, freqs)
        if total < 1e-15:
            bad_spec[s:s + win_sp] = True
            continue
        m = (freqs >= 0.1) & (freqs <= 0.7)
        if m.any() and np.trapz(psd[m], freqs[m]) / total < 0.30:
            bad_spec[s:s + win_sp] = True

    envelope = np.abs(hilbert(resp_fn))
    win_env = int(5 * resp_fs)
    env_smooth = np.convolve(envelope, np.ones(win_env) / win_env, mode="same")
    bad_env = env_smooth < np.median(env_smooth) * 0.4

    bad = bad_spec | bad_env | _rolling_amplitude_bad(resp_fn, resp_fs, 10.0, 2.5) | \
          _flag_flatline(resp_fn, resp_fs, 3.0) | _flag_clipping(resp_fn)
    return (~bad).astype(float)


# ── Load & compute ───────────────────────────────────────────────────────

def load_signals_and_sqi(subject_id):
    from datetime import datetime as _dt

    biodata = DataProcessor(os.path.join("data", f"{subject_id}.txt")).process_file(lines_to_skip=20)
    df_ts = pd.read_excel("data/Experiment_2_Stamps_S.xlsx", sheet_name=subject_id)
    df_ts["timestamp"] = df_ts.apply(
        lambda r: _dt(int(r["Year"]), int(r["Month"]), int(r["Day"]),
                      int(r.get("Hour", 0) if pd.notna(r.get("Hour")) else 0),
                      int(r.get("Minute", 0) if pd.notna(r.get("Minute")) else 0),
                      int(r.get("Second", 0) if pd.notna(r.get("Second")) else 0)),
        axis=1)
    start_time = df_ts.loc[df_ts["Event"] == "Data Start", "timestamp"].iloc[0]
    df_ts["sec"] = (df_ts["timestamp"] - start_time).dt.total_seconds()
    video_start = df_ts.loc[df_ts["Event"] == "Video Start", "sec"].iloc[0]
    study_end = df_ts.loc[df_ts["Event"] == "Survey 3 End", "sec"].iloc[0]
    if study_end < 0:
        study_end = -1
    CLIP = [int(video_start), int(study_end)]

    Bfs = 2000
    ecg_fs, ppg_fs, resp_fs, eda_fs = 200, 200, 100, 4

    ecg_raw = SignalProcessor.downsample_signal(biodata[2][-1][CLIP[0]*Bfs:CLIP[1]*Bfs], Bfs, ecg_fs)
    ppg_raw = SignalProcessor.downsample_signal(biodata[1][-1][CLIP[0]*Bfs:CLIP[1]*Bfs], Bfs, ppg_fs)
    resp_raw = SignalProcessor.downsample_signal(biodata[4][-1][CLIP[0]*Bfs:CLIP[1]*Bfs], Bfs, resp_fs)
    eda_raw = SignalProcessor.downsample_signal(biodata[3][-1][CLIP[0]*Bfs:CLIP[1]*Bfs], Bfs, eda_fs)

    ecg_fn = DataNormalizer().standardize_data(nk.ecg_clean(ecg_raw, sampling_rate=ecg_fs))
    ppg_fn = DataNormalizer().standardize_data(nk.ppg_clean(ppg_raw, sampling_rate=ppg_fs, method="nabian2018"))
    resp_fn = DataNormalizer().standardize_data(nk.rsp_clean(resp_raw, sampling_rate=resp_fs, method="khodadad2018"))
    eda_fn = DataNormalizer().normalize_data(nk.eda_clean(eda_raw, sampling_rate=eda_fs))

    ecg_sqi = _ecg_sqi(ecg_fn, ecg_fs)
    ppg_sqi = _ppg_sqi(ppg_fn, ppg_raw, ppg_fs)
    resp_sqi = _resp_sqi(resp_fn, resp_fs)

    # EDA SQI via wavelet + XGBoost
    ts_eda = np.arange(len(eda_fn)) / eda_fs
    df_eda = pd.DataFrame({"EDA": eda_fn[1:], "Time": ts_eda[1:]})
    EDA_SQA.compute_eda_artifacts(df_eda)
    eda_data = pd.read_csv("output/artifacts-removed.csv")
    q_eda = SignalProcessor.replace_consecutive_ones(eda_data["Artifact"], consecutive_one=60)
    eda_sqi = (np.array(q_eda) == 0).astype(float)
    if len(eda_sqi) < len(eda_fn):
        eda_sqi = np.pad(eda_sqi, (0, len(eda_fn) - len(eda_sqi)), constant_values=1)
    elif len(eda_sqi) > len(eda_fn):
        eda_sqi = eda_sqi[:len(eda_fn)]

    return dict(ecg=ecg_fn, ecg_fs=ecg_fs, ecg_sqi=ecg_sqi,
                ppg=ppg_fn, ppg_fs=ppg_fs, ppg_sqi=ppg_sqi,
                resp=resp_fn, resp_fs=resp_fs, resp_sqi=resp_sqi,
                eda=eda_fn, eda_fs=eda_fs, eda_sqi=eda_sqi)


# ── Inset helper ─────────────────────────────────────────────────────────

SIG_COLOR, SQI_COLOR = "#0044CC", "#e01b24"


def _find_inset_region(sqi, fs, target_sec=40):
    step = max(1, int(fs))
    sqi_1hz = sqi[::step]
    transitions = np.where(np.diff(sqi_1hz) != 0)[0]
    if len(transitions) == 0:
        mid = len(sqi) // (2 * int(fs))
        return max(0, mid - target_sec // 2), mid + target_sec // 2
    mid_idx = len(sqi_1hz) // 2
    best = transitions[np.argmin(np.abs(transitions - mid_idx))]
    return max(0, best - target_sec // 2), best + target_sec // 2


def _add_inset(ax, t_sec, sig, sqi, fs, inset_bounds, inset_pos):
    t_s, t_e = inset_bounds
    ax.add_patch(Rectangle((t_s, ax.get_ylim()[0]), t_e - t_s,
                            ax.get_ylim()[1] - ax.get_ylim()[0],
                            lw=1.2, edgecolor="black", facecolor="none", ls="--", zorder=5))
    ax_ins = ax.inset_axes(inset_pos)
    si, ei = int(t_s * fs), min(int(t_e * fs), len(sig))
    tc, sc, qc = t_sec[si:ei], sig[si:ei], sqi[si:ei]
    sv = sc[~np.isnan(sc)] if np.isnan(sc).any() else sc
    shi, slo = (np.max(sv), np.min(sv)) if len(sv) else (1, 0)
    skip = max(1, len(tc) // 4000)
    ax_ins.plot(tc[::skip], sc[::skip], color=SIG_COLOR, lw=0.3, alpha=0.8)
    ax_ins.plot(tc[::skip], (slo + qc * (shi - slo))[::skip], color=SQI_COLOR, lw=0.6, alpha=0.9)
    ax_ins.set_xlim(t_s, t_e)
    ax_ins.tick_params(labelsize=5, length=2, pad=1)
    ax_ins.text(0.03, 0.92, "SQI", transform=ax_ins.transAxes, fontsize=5,
                color=SQI_COLOR, fontweight="bold", va="top")
    for sp in ax_ins.spines.values():
        sp.set_linewidth(0.5); sp.set_color("grey")


# ── Plot ─────────────────────────────────────────────────────────────────

def plot_sqi_figure(data, output_path, fmt="pdf"):
    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.5))
        configs = [
            (0, 0, "eda",  "eda_fs",  "eda_sqi",  "EDA",  [0.55, 0.50, 0.42, 0.45]),
            (0, 1, "ecg",  "ecg_fs",  "ecg_sqi",  "ECG",  [0.05, 0.05, 0.42, 0.40]),
            (1, 0, "ppg",  "ppg_fs",  "ppg_sqi",  "PPG",  [0.05, 0.55, 0.42, 0.40]),
            (1, 1, "resp", "resp_fs", "resp_sqi", "Resp", [0.55, 0.05, 0.42, 0.40]),
        ]
        for row, col, sk, fk, qk, ylabel, ipos in configs:
            ax = axes[row, col]
            sig, fs, sqi = data[sk], data[fk], data[qk]
            t = np.arange(len(sig)) / fs
            skip = max(1, len(sig) // 8000)
            sv = sig[~np.isnan(sig)] if np.isnan(sig).any() else sig
            shi = np.percentile(sv, 99.5) if len(sv) else 1
            slo = np.percentile(sv, 0.5) if len(sv) else 0
            sqi_s = slo + sqi * (shi - slo)
            ax.plot(t[::skip], sig[::skip], color=SIG_COLOR, lw=0.25, alpha=0.85, rasterized=True)
            ax.plot(t[::skip], sqi_s[::skip], color=SQI_COLOR, lw=0.5, alpha=0.9, label="SQI")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_ylim(slo - 0.1 * (shi - slo), shi + 0.1 * (shi - slo))
            ax.legend(loc="upper right", fontsize=6, frameon=False)
            _add_inset(ax, t, sig, sqi, fs, _find_inset_region(sqi, fs), ipos)
        for ax in axes[1]:
            ax.set_xlabel("time (s)", fontsize=8)
        plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        out = f"{output_path}.{fmt}" if not output_path.endswith(f".{fmt}") else output_path
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="P4S")
    parser.add_argument("--output", default="output/figures/sqi_figure")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    args = parser.parse_args()
    data = load_signals_and_sqi(args.subject)
    plot_sqi_figure(data, args.output, fmt=args.format)
