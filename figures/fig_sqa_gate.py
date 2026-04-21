#!/usr/bin/env python3
"""
SQA-Weighted Fusion Gate figure (panel c2).

Top: Fused activation with/without SQA gating + difference signal.
Bottom: Compact SQA quality heatmap strip per modality.

Requires: Biopac .txt data, timestamps Excel.

Usage:
    MPLBACKEND=Agg python polished/fig_sqa_gate.py [--subject P4S]
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
import matplotlib.colors as mcolors

try:
    import scienceplots  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neurokit2 as nk
from scipy.signal import welch, hilbert
from Load.LoadData import DataProcessor, SkinTempProcessor, DataNormalizer
from Utils.func import SignalProcessor
from PreProcessing import EDA_SQA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODALITY_ORDER = ["ECG", "PPG", "Resp", "EDA", "Temp"]
AFFECTIVE_EVENTS = {"Crash", "Barrel", "Clear Text", "Fog Start", "Brake"}


# ── SQA helpers (compact) ───────────────────────────────────────────────

def _flag_flatline(sig, fs, min_dur=2.0, thr=1e-6):
    win = int(min_dur * fs)
    if win < 2 or len(sig) < win:
        return np.zeros(len(sig), dtype=bool)
    f = np.zeros(len(sig), dtype=bool)
    for s in range(0, len(sig) - win + 1, max(1, win // 2)):
        if np.nanstd(sig[s:s + win]) < thr:
            f[s:s + win] = True
    return f


def _flag_clipping(sig, pm=0.5):
    v = sig[~np.isnan(sig)] if np.isnan(sig).any() else sig
    if len(v) < 10:
        return np.zeros(len(sig), dtype=bool)
    lo, hi = np.percentile(v, [pm, 100 - pm])
    return (sig <= lo) | (sig >= hi)


def _rolling_amp_bad(sig, fs, win_sec=5.0, z_thr=3.0):
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
        if (ranges[i] - med) / mad > z_thr:
            bad[s:s + win] = True
    return bad


def _ecg_sqi(ecg_fn, fs):
    q = nk.ecg_quality(ecg_fn, sampling_rate=fs)
    return (~((q < 0.7) | _flag_flatline(ecg_fn, fs))).astype(float)


def _ppg_sqi(ppg_fn, ppg_raw, fs):
    q = nk.ppg_quality(ppg_fn, sampling_rate=fs, method="templatematch")
    bad = (q < np.mean(q) * 0.8) | _rolling_amp_bad(ppg_fn, fs) | \
          _flag_flatline(ppg_fn, fs) | _flag_clipping(ppg_raw)
    return (~bad).astype(float)


def _resp_sqi(resp_fn, fs):
    n = len(resp_fn)
    win_sp, step_sp = int(10 * fs), int(1 * fs)
    bad_spec = np.zeros(n, dtype=bool)
    for s in range(0, n - win_sp + 1, step_sp):
        c = resp_fn[s:s + win_sp]
        if np.nanstd(c) < 1e-10:
            bad_spec[s:s + win_sp] = True; continue
        fr, psd = welch(c, fs=fs, nperseg=min(len(c), int(4 * fs)))
        tot = np.trapz(psd, fr)
        if tot < 1e-15:
            bad_spec[s:s + win_sp] = True; continue
        m = (fr >= 0.1) & (fr <= 0.7)
        if m.any() and np.trapz(psd[m], fr[m]) / tot < 0.30:
            bad_spec[s:s + win_sp] = True
    env = np.abs(hilbert(resp_fn))
    win_e = int(5 * fs)
    env_s = np.convolve(env, np.ones(win_e) / win_e, mode="same")
    bad = bad_spec | (env_s < np.median(env_s) * 0.4) | \
          _rolling_amp_bad(resp_fn, fs, 10.0, 2.5) | \
          _flag_flatline(resp_fn, fs, 3.0) | _flag_clipping(resp_fn)
    return (~bad).astype(float)


def _eda_sqi(eda_fn, fs):
    ts = np.arange(len(eda_fn)) / fs
    df = pd.DataFrame({"EDA": eda_fn[1:], "Time": ts[1:]})
    EDA_SQA.compute_eda_artifacts(df)
    eda_data = pd.read_csv(os.path.join(ROOT, "output", "artifacts-removed.csv"))
    q = SignalProcessor.replace_consecutive_ones(eda_data["Artifact"], consecutive_one=60)
    sqi = (np.array(q) == 0).astype(float)
    if len(sqi) < len(eda_fn):
        sqi = np.pad(sqi, (0, len(eda_fn) - len(sqi)), constant_values=1)
    elif len(sqi) > len(eda_fn):
        sqi = sqi[:len(eda_fn)]
    return sqi


# ── Data loading ─────────────────────────────────────────────────────────

def load_data_and_features(subject_id, window_sec=30, step_sec=1, buffer_sec=60):
    from scripts.experiment_focus import extract_hires_features
    from datetime import datetime as _dt

    data_dir = os.path.join(ROOT, "data")
    biodata = DataProcessor(os.path.join(data_dir, f"{subject_id}.txt")).process_file(lines_to_skip=20)
    df_ts = pd.read_excel(os.path.join(data_dir, "Experiment_2_Stamps_S.xlsx"),
                          sheet_name=subject_id)
    df_ts["timestamp"] = df_ts.apply(
        lambda r: _dt(int(r["Year"]), int(r["Month"]), int(r["Day"]),
                      int(r["Hour"]) if pd.notna(r["Hour"]) else 0,
                      int(r["Minute"]) if pd.notna(r["Minute"]) else 0,
                      int(r["Second"]) if pd.notna(r["Second"]) else 0),
        axis=1)
    start_time = df_ts.loc[df_ts["Event"] == "Data Start", "timestamp"].iloc[0]
    df_ts["sec"] = (df_ts["timestamp"] - start_time).dt.total_seconds()
    video_start = df_ts.loc[df_ts["Event"] == "Video Start", "sec"].iloc[0]
    study_end = df_ts.loc[df_ts["Event"] == "Survey 3 End", "sec"].iloc[0]
    if study_end < 0:
        study_end = -1
    CLIP = [int(video_start), int(study_end)]

    Bfs = 2000
    ecg_fs, ppg_fs, resp_fs, eda_fs, temp_fs = 200, 200, 100, 4, 4

    ds = {}
    for idx, name, tfs in [(2, "ecg", ecg_fs), (1, "ppg", ppg_fs),
                            (4, "resp", resp_fs), (3, "eda", eda_fs), (5, "temp", temp_fs)]:
        ds[name] = SignalProcessor.downsample_signal(
            biodata[idx][-1][CLIP[0]*Bfs:CLIP[1]*Bfs], Bfs, tfs)

    temp_f = SkinTempProcessor().process_skin_temp_signal(ds["temp"])
    ecg_fn = DataNormalizer().standardize_data(nk.ecg_clean(ds["ecg"], sampling_rate=ecg_fs))
    ppg_fn = DataNormalizer().standardize_data(nk.ppg_clean(ds["ppg"], sampling_rate=ppg_fs, method="nabian2018"))
    resp_fn = DataNormalizer().standardize_data(nk.rsp_clean(ds["resp"], sampling_rate=resp_fs, method="khodadad2018"))
    eda_fn = DataNormalizer().normalize_data(nk.eda_clean(ds["eda"], sampling_rate=eda_fs))

    ecg_sqi = _ecg_sqi(ecg_fn, ecg_fs)
    ppg_sqi = _ppg_sqi(ppg_fn, ds["ppg"], ppg_fs)
    resp_sqi = _resp_sqi(resp_fn, resp_fs)
    eda_sqi = _eda_sqi(eda_fn, eda_fs)
    temp_sqi = (~_flag_flatline(np.array(temp_f, dtype=float), temp_fs, 5.0)).astype(float)

    exp_s_row = df_ts.loc[df_ts["Event"] == "Experiment Start"]
    exp_e_row = df_ts.loc[df_ts["Event"] == "Experiment End"]
    exp_s_rel = exp_s_row["sec"].iloc[0] - video_start
    exp_e_rel = exp_e_row["sec"].iloc[0] - video_start
    crop_s = max(0, exp_s_rel - buffer_sec)
    crop_e = exp_e_rel + buffer_sec

    def _crop(sig, fs):
        return sig[int(crop_s * fs):min(int(crop_e * fs), len(sig))]

    feat = extract_hires_features(
        _crop(ecg_fn, ecg_fs), _crop(ppg_fn, ppg_fs), _crop(resp_fn, resp_fs),
        _crop(eda_fn, eda_fs), _crop(np.array(temp_f, dtype=float), temp_fs),
        ecg_fs=ecg_fs, ppg_fs=ppg_fs, resp_fs=resp_fs,
        eda_fs=eda_fs, temp_fs=temp_fs,
        window_sec=window_sec, step_sec=step_sec)
    feat["time_vec"] = feat["time_vec"] + crop_s

    sqi_map = {"ECG": (ecg_sqi, ecg_fs), "PPG": (ppg_sqi, ppg_fs),
               "Resp": (resp_sqi, resp_fs), "EDA": (eda_sqi, eda_fs),
               "Temp": (temp_sqi, temp_fs)}
    n_win = len(feat["time_vec"])
    sqa_win = {}
    for mod, (sq, fs) in sqi_map.items():
        sq_crop = _crop(sq, fs)
        quality = np.ones(n_win)
        for wi in range(n_win):
            tc = feat["time_vec"][wi] - crop_s
            ws = int(max(0, (tc - window_sec / 2)) * fs)
            we = int(min(len(sq_crop), (tc + window_sec / 2) * fs))
            if we > ws:
                quality[wi] = np.mean(sq_crop[ws:we])
        sqa_win[mod] = quality

    evt_times = df_ts["sec"].values - video_start
    m = evt_times >= 0
    return dict(feat=feat, sqa_window=sqa_win, event_times=evt_times[m],
                event_names=df_ts["Event"].values[m],
                exp_start=exp_s_rel, exp_end=exp_e_rel, subject_id=subject_id)


# ── Fusion computation ───────────────────────────────────────────────────

def _exp_mask(tv, es, ee):
    return (tv >= es) & (tv <= ee)


def _compute_fusion(d):
    feat = d["feat"]
    tv = np.array(feat["time_vec"])
    mod_map = feat["modality_map"]
    sqa = d["sqa_window"]
    mask = _exp_mask(tv, d["exp_start"], d["exp_end"])
    t_exp = tv[mask]
    matrix = feat["matrix"]

    rows_w, rows_ng, labels = [], [], []
    for mod in MODALITY_ORDER:
        if mod not in mod_map:
            continue
        cols = mod_map[mod]
        mm = matrix[mask][:, cols]
        ng = np.nanmean(mm, axis=1)
        rows_ng.append(ng)
        wm = np.copy(mm)
        if mod in sqa:
            wm = wm * sqa[mod][mask][:, np.newaxis]
        rows_w.append(np.nanmean(wm, axis=1))
        labels.append(mod)

    return dict(t_exp=t_exp, rows_w=rows_w, rows_ng=rows_ng, labels=labels,
                overall_w=np.nanmean(rows_w, axis=0),
                overall_ng=np.nanmean(rows_ng, axis=0), mask=mask)


# ── Plot ─────────────────────────────────────────────────────────────────

def build_panel_c2(d, output_dir, fmt="png"):
    cd = _compute_fusion(d)
    t_exp = cd["t_exp"]
    tv = np.array(d["feat"]["time_vec"])
    sqa = d["sqa_window"]
    mask = _exp_mask(tv, d["exp_start"], d["exp_end"])

    sqa_rows, sqa_labels = [], []
    for mod in MODALITY_ORDER:
        if mod in sqa:
            sqa_rows.append(sqa[mod][mask])
            sqa_labels.append(mod)
    sqa_matrix = np.array(sqa_rows)

    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10})
    fig, (ax_top, ax_sqa) = plt.subplots(
        2, 1, figsize=(7.0, 3.5), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0], "hspace": 0.10})

    ow, ong = cd["overall_w"], cd["overall_ng"]
    ax_top.plot(t_exp, ong, color="#888888", lw=1.0, label="Without SQA Gate", alpha=0.85)
    ax_top.fill_between(t_exp, 0, ong, color="#888888", alpha=0.08)
    ax_top.plot(t_exp, ow, color="#c0392b", lw=1.0, label="With SQA Gate", alpha=0.9)
    ax_top.fill_between(t_exp, 0, ow, color="#c0392b", alpha=0.12)
    ax_top.fill_between(t_exp, ong, ow, where=(ow > ong), color="#c0392b", alpha=0.18, interpolate=True)
    ax_top.fill_between(t_exp, ong, ow, where=(ow <= ong), color="#2980b9", alpha=0.18, interpolate=True)

    diff = np.array(ow) - np.array(ong)
    ax_diff = ax_top.twinx()
    ax_diff.plot(t_exp, diff, color="#2980b9", lw=0.8, label="Difference", alpha=0.8, ls="--")
    ax_diff.fill_between(t_exp, 0, diff, color="#2980b9", alpha=0.10)
    ax_diff.set_ylabel("Difference", fontsize=12, color="#2980b9")
    ax_diff.tick_params(labelsize=10, colors="#2980b9")
    ax_diff.spines["right"].set_color("#2980b9")
    h1, l1 = ax_top.get_legend_handles_labels()
    h2, l2 = ax_diff.get_legend_handles_labels()
    ax_top.legend(h1 + h2, l1 + l2, fontsize=10, loc="upper right", framealpha=0.7)
    ax_top.set_ylabel("Fused Activation", fontsize=12)
    ax_top.set_ylim(bottom=0)

    # SQA heatmap
    cmap_rb = mcolors.LinearSegmentedColormap.from_list("sqa_rb", [(1, 1, 1), (0.85, 0.1, 0.1)])
    n_rows = len(sqa_labels)
    extent = [t_exp[0], t_exp[-1], n_rows - 0.5, -0.5]
    im = ax_sqa.imshow(1.0 - sqa_matrix, aspect="auto", extent=extent,
                       interpolation="nearest", cmap=cmap_rb, vmin=0, vmax=1)
    ax_sqa.set_yticks(range(n_rows))
    ax_sqa.set_yticklabels(sqa_labels, fontsize=10, fontweight="bold")
    ax_sqa.tick_params(axis="y", length=0, pad=3)
    ax_sqa.set_xlabel("Time (s)", fontsize=12)
    for r in range(1, n_rows):
        ax_sqa.axhline(y=r - 0.5, color="white", lw=1.0)

    bbox = ax_sqa.get_position()
    cb_ax = fig.add_axes([bbox.x0 - 0.06, bbox.y0, 0.01, bbox.height])
    cb = fig.colorbar(im, cax=cb_ax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["0", "1"], fontsize=10)
    cb.ax.tick_params(length=2, pad=2)
    cb.ax.yaxis.set_ticks_position("left")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"panel_c2_fusion_{d['subject_id']}.{fmt}")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="P4S")
    parser.add_argument("--format", default="png", choices=["pdf", "png"])
    args = parser.parse_args()

    d = load_data_and_features(args.subject)
    build_panel_c2(d, os.path.join(ROOT, "output", "figures"), fmt=args.format)
