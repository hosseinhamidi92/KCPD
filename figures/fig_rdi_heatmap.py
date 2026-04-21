#!/usr/bin/env python3
"""
Two-panel RDI heatmap strips.

  (a) WESAD S3 — RDI with stress condition
  (b) P19S (driving) — RDI with affective events

Caches pipeline output to .pkl; subsequent runs only re-plot.

Usage:
    MPLBACKEND=Agg python polished/fig_rdi_heatmap.py
"""

import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

try:
    import scienceplots  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "output/segment_outlier_figures"
CACHE_DIR = os.path.join(OUT_DIR, "_cache")

RDI_CMAP = LinearSegmentedColormap.from_list("rdi", [
    (0.0, "#2C3E50"), (0.3, "#2980B9"), (0.5, "#27AE60"),
    (0.7, "#F39C12"), (0.85, "#E74C3C"), (1.0, "#8E44AD"),
])


# ── Caching ──────────────────────────────────────────────────────────────

def _cache_path(subject_id):
    return os.path.join(CACHE_DIR, f"{subject_id}_rdi_cache.pkl")


def _save_cache(subject_id, d):
    os.makedirs(CACHE_DIR, exist_ok=True)
    so = d["segment_outlier"]
    cache = dict(
        subject_id=d["subject_id"],
        time_vec=np.array(d["time_vec"]),
        all_event_times=np.array(d["all_event_times"]),
        all_event_names=list(d["all_event_names"]),
        exp_start=d["exp_start"], exp_end=d["exp_end"],
        segments=[(s["start"], s["end"]) for s in so["segments"]],
        consensus=so["consensus"],
    )
    if "_wesad_spans" in d:
        cache["_wesad_spans"] = d["_wesad_spans"]
    with open(_cache_path(subject_id), "wb") as f:
        pickle.dump(cache, f)


def _load_cache(subject_id):
    with open(_cache_path(subject_id), "rb") as f:
        return pickle.load(f)


def _load_wesad(subject_id):
    cp = _cache_path(f"WESAD_{subject_id}")
    if os.path.exists(cp):
        return _load_cache(f"WESAD_{subject_id}")
    from scripts.wesad_analysis import load_wesad_subject, run_wesad_segment_outlier
    d = load_wesad_subject(subject_id, data_dir="data/WESAD")
    d = run_wesad_segment_outlier(d, penalty=1.0, min_size=10)
    _save_cache(f"WESAD_{subject_id}", d)
    return _load_cache(f"WESAD_{subject_id}")


def _load_driving(subject_id):
    cp = _cache_path(subject_id)
    if os.path.exists(cp):
        return _load_cache(subject_id)
    from scripts.segment_outlier import run_segment_outlier_pipeline
    d = run_segment_outlier_pipeline(
        subject_id, "data", "data", "Experiment_2_Stamps_S.xlsx",
        penalty=1.0, min_size=10, buffer_sec=99999, window_sec=30, step_sec=1)
    _save_cache(subject_id, d)
    return _load_cache(subject_id)


# ── Plot ─────────────────────────────────────────────────────────────────

def _expand_consensus(c):
    td = c["time_vec"]
    ct = np.full(len(td), np.nan)
    for (s, e), score in zip(c["segments"], c["consensus"]):
        ct[s:e] = score
    return td, ct


def _plot_rdi_panel(ax, ax_cb, c):
    td, ct = _expand_consensus(c)
    im = ax.imshow(ct[np.newaxis, :], aspect="auto", cmap=RDI_CMAP, vmin=0, vmax=1,
                   extent=[td[0], td[-1], 0, 1], interpolation="bilinear")
    ax.set_xlim(td[0], td[-1])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    plt.colorbar(im, cax=ax_cb, orientation="vertical")
    ax_cb.set_ylabel("RDI", fontsize=8, rotation=90, labelpad=8, va="center")
    ax_cb.tick_params(labelsize=7)


def _make_figure(c_top, c_bottom):
    fig = plt.figure(figsize=(7, 2.0), dpi=200)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 0.015], hspace=0.9, wspace=0.05)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[:, 1])
    _plot_rdi_panel(ax_a, ax_cb, c_top)
    _plot_rdi_panel(ax_b, ax_cb, c_bottom)
    return fig


if __name__ == "__main__":
    c3 = _load_wesad("S3")
    c19 = _load_driving("P19S")

    plt.style.use(["science", "no-latex"])
    os.makedirs(OUT_DIR, exist_ok=True)

    for ext in ("png", "pdf"):
        fig = _make_figure(c3, c19)
        path = os.path.join(OUT_DIR, f"fig_rdi_wesad_panel.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved → {path}")
