#!/usr/bin/env python3
"""
Feature response direction & latency figure (2-panel).

  (a) Per-modality mean z-score trajectory around events
  (b) Per-feature mean ΔZ bar chart grouped by modality

Loads pre-computed data from feature_response_cache.json.

Usage:
    MPLBACKEND=Agg python polished/fig_feature_response.py [--replot]
"""

import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_PATH = "output/feature_response/feature_response_cache.json"
OUT_DIR = "output/figures"
PAGE_W = 7.16

MOD_ORDER = ["ECG", "PPG", "Resp", "EDA", "Temp"]
MOD_COLORS = {"ECG": "#1f77b4", "PPG": "#ff7f0e", "Resp": "#2ca02c",
              "EDA": "#9467bd", "Temp": "#d62728"}
MOD_LS = {"ECG": "-", "PPG": "--", "Resp": "-.", "EDA": ":", "Temp": "-"}

_FEAT_TO_MOD = {
    "SDNN": "ECG", "RMSSD": "ECG", "HF Power": "ECG", "LF/HF": "ECG",
    "SampEn": "ECG", "pNN50": "ECG", "Inst HR": "ECG",
    "Pulse Rate": "PPG", "PAV": "PPG", "PTT": "PPG",
    "Breath Rate": "Resp", "RVT": "Resp", "Inst Breath Rate": "Resp",
    "SCL": "EDA", "SCR Freq": "EDA", "Phasic Amp": "EDA",
    "SCR Rise Time": "EDA",
    "Skin Temp": "Temp", "dSkin Temp": "Temp",
}


def _load(replot=False):
    if replot and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    # Fall back to full computation
    from scripts.plot_feature_response import _collect_feature_trajectories
    return _collect_feature_trajectories()


def _plot(data):
    trajs = {mod: [(np.array(t["offsets"]), np.array(t["scores"]))
                   for t in data["trajectories"][mod]]
             for mod in MOD_ORDER}
    feat_dirs = data["feature_directions"]

    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 8, "font.family": "serif",
        "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(PAGE_W, PAGE_W * 0.42))
    ax_a.set_box_aspect(0.85)

    # Panel (a): trajectory
    for mod in MOD_ORDER:
        if not trajs[mod]:
            continue
        all_off = sorted(set(o for offs, _ in trajs[mod] for o in offs))
        means = np.array([np.mean([sc[of == off][0] for of, sc in trajs[mod] if off in of])
                          for off in all_off])
        sems = np.array([np.std([sc[of == off][0] for of, sc in trajs[mod] if off in of])
                         / max(1, np.sqrt(sum(1 for of, _ in trajs[mod] if off in of)))
                         for off in all_off])
        ax_a.fill_between(all_off, means - sems, means + sems, alpha=0.10, color=MOD_COLORS[mod])
        ax_a.plot(all_off, means, color=MOD_COLORS[mod], lw=1.4, ls=MOD_LS[mod],
                  marker="o", markersize=3, label=f"{mod} (n={len(trajs[mod])})")

    ax_a.axvline(0, color="black", ls="--", lw=0.8)
    ax_a.axhline(0, color="gray", ls=":", lw=0.5)
    ax_a.set_xlabel("Segment Offset from Event")
    ax_a.set_ylabel("Mean Feature Z-Score")
    ax_a.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax_a.text(0.02, 1.05, "(a)", transform=ax_a.transAxes, fontsize=10, fontweight="bold", va="bottom")

    # Panel (b): bar chart
    feat_data = []
    for fname, deltas in feat_dirs.items():
        mod = _FEAT_TO_MOD.get(fname, "?")
        if mod not in MOD_ORDER:
            continue
        feat_data.append((mod, fname, np.mean(deltas),
                          np.std(deltas) / np.sqrt(len(deltas)) if len(deltas) > 1 else 0))
    feat_data.sort(key=lambda x: (MOD_ORDER.index(x[0]), -abs(x[2])))

    y_pos = np.arange(len(feat_data))
    ax_b.barh(y_pos, [f[2] for f in feat_data],
              xerr=[f[3] for f in feat_data], height=0.65,
              color=[MOD_COLORS[f[0]] for f in feat_data], alpha=0.75,
              edgecolor=[MOD_COLORS[f[0]] for f in feat_data], lw=0.6,
              error_kw=dict(elinewidth=0.7, capsize=2, capthick=0.7))
    ax_b.axvline(0, color="black", lw=0.8)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([f[1] for f in feat_data], fontsize=7)
    ax_b.set_xlabel("Mean $\\Delta$Z at Event")
    ax_b.invert_yaxis()

    prev_mod = None
    group_starts = []
    for i, (mod, *_) in enumerate(feat_data):
        if mod != prev_mod:
            group_starts.append((i, mod))
            prev_mod = mod
    for gi, (si, mod) in enumerate(group_starts):
        ei = group_starts[gi + 1][0] if gi + 1 < len(group_starts) else len(feat_data)
        ax_b.text(ax_b.get_xlim()[1] * 1.02, (si + ei - 1) / 2, mod,
                  ha="left", va="center", fontsize=8, fontweight="bold", color=MOD_COLORS[mod])
    ax_b.text(0.02, 1.05, "(b)", transform=ax_b.transAxes, fontsize=10, fontweight="bold", va="bottom")

    fig.tight_layout(w_pad=2.0)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_feature_response.{ext}"),
                    dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {OUT_DIR}/fig_feature_response.png/.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    _plot(_load(replot=args.replot))
