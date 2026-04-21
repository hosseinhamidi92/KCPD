#!/usr/bin/env python3
"""
Session overview: 3-panel figure for a single subject.

  (a) Phase timeline with affective event markers
  (b) Normalised feature heatmap (19 features) + segment boundaries
  (c) Consensus outlier intensity strip

Requires: Biopac .txt data, timestamps Excel.

Usage:
    MPLBACKEND=Agg python polished/fig_session_overview.py [--subject P4S]
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

try:
    import scienceplots  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.segment_outlier import run_segment_outlier_pipeline, _get_session_phases

STYLE = ["science", "ieee", "no-latex"]
DPI = 600

PHASE_COLORS = {
    "Video": "#AED6F1", "Survey 1": "#D5DBDB", "Practice": "#ABEBC6",
    "Survey 2": "#D5DBDB", "Experiment": "#F9E79F", "Survey 3": "#D5DBDB",
}
EVENT_COLORS = {
    "Crash": "#e74c3c", "Barrel": "#e67e22",
    "Fog Start": "#8e44ad", "Brake": "#795548",
}
AFFECTIVE = set(EVENT_COLORS)

RDI_CMAP = LinearSegmentedColormap.from_list("consensus", [
    (0.0, "#2C3E50"), (0.3, "#2980B9"), (0.5, "#27AE60"),
    (0.7, "#F39C12"), (0.85, "#E74C3C"), (1.0, "#8E44AD"),
])


def generate_figure(d, out_path):
    so = d["segment_outlier"]
    td = np.array(d["time_vec"])
    features = d["Used"]
    labels = so["labels"]
    segments = so["segments"]
    consensus = so["consensus"]
    T = len(td)

    consensus_time = np.full(T, np.nan)
    for si, seg in enumerate(segments):
        consensus_time[seg["start"]:seg["end"]] = consensus[si]

    phases = _get_session_phases(d)

    with plt.style.context(STYLE):
        plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                             "xtick.labelsize": 8, "ytick.labelsize": 8})
        fig = plt.figure(figsize=(7.16, 4.2))
        gs = gridspec.GridSpec(3, 2, height_ratios=[0.07, 0.78, 0.15],
                               width_ratios=[0.97, 0.03], hspace=0.08, wspace=0.03)

        ax_phase = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_phase)
        ax_cb = fig.add_subplot(gs[1, 1])
        ax_cons = fig.add_subplot(gs[2, 0], sharex=ax_phase)
        fig.add_subplot(gs[0, 1]).axis("off")
        ax_cons_cb = fig.add_subplot(gs[2, 1])

        t_min, t_max = td[0], td[-1]

        # Panel (a) — phase bar
        ax_phase.set_xlim(t_min, t_max)
        ax_phase.set_ylim(0, 1)
        for lbl, ps, pe, pc in phases:
            ax_phase.axvspan(ps, pe, color=pc, alpha=0.7)
            ax_phase.text((ps + pe) / 2, 0.5, lbl, ha="center", va="center",
                          fontsize=6, fontweight="bold", clip_on=True)
        for et, en in zip(d["all_event_times"], d["all_event_names"]):
            if en not in AFFECTIVE:
                continue
            col = EVENT_COLORS.get(en, "gray")
            ax_phase.axvline(et, color=col, lw=1.2, zorder=5)
            ax_phase.annotate(en, xy=(et, 1.0), xytext=(0, 4),
                              textcoords="offset points", ha="center", va="bottom",
                              fontsize=5.5, color=col, fontweight="bold",
                              arrowprops=dict(arrowstyle="-", color=col, lw=0.6),
                              clip_on=False)
        ax_phase.set_yticks([])
        ax_phase.tick_params(axis="x", labelbottom=False)
        for s in ("top", "right", "left"):
            ax_phase.spines[s].set_visible(False)

        # Panel (b) — feature heatmap
        im = ax_heat.imshow(features.T, aspect="auto",
                            extent=[t_min, t_max, len(labels) - 0.5, -0.5],
                            cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
        ax_heat.set_yticks(range(len(labels)))
        ax_heat.set_yticklabels(labels, fontsize=5.5)
        ax_heat.tick_params(axis="x", labelbottom=False)
        for cp in so["boundaries"]:
            ax_heat.axvline(td[min(cp, T - 1)], color="black", lw=0.3, ls=":", alpha=0.5)
        for et, en in zip(d["all_event_times"], d["all_event_names"]):
            if en in AFFECTIVE:
                ax_heat.axvline(et, color=EVENT_COLORS.get(en, "gray"), lw=0.8, ls="--", alpha=0.8)
        ax_heat.axvspan(d["exp_start"], d["exp_end"], color="#F9E79F", alpha=0.08)
        cb = fig.colorbar(im, cax=ax_cb)
        cb.set_label("Normalised value", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        # Panel (c) — consensus strip
        im_c = ax_cons.imshow(consensus_time[np.newaxis, :], aspect="auto",
                              extent=[t_min, t_max, 0, 1], cmap=RDI_CMAP,
                              vmin=0, vmax=1, interpolation="nearest")
        for et, en in zip(d["all_event_times"], d["all_event_names"]):
            if en in AFFECTIVE:
                ax_cons.axvline(et, color=EVENT_COLORS.get(en, "gray"), lw=0.8, ls="--", alpha=0.9)
        ax_cons.set_yticks([])
        ax_cons.set_ylabel("Outlier\nintensity", fontsize=7, rotation=0, labelpad=15, va="center")
        ax_cons.set_xlabel("Time (s)", fontsize=9)
        ax_cons.set_xlim(t_min, t_max)
        cb_c = fig.colorbar(im_c, cax=ax_cons_cb)
        cb_c.set_label("Score", fontsize=6)
        cb_c.ax.tick_params(labelsize=5)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="P4S")
    parser.add_argument("--output", default="output/figures/fig_session_overview.png")
    args = parser.parse_args()

    d = run_segment_outlier_pipeline(
        args.subject, "data", "data", "Experiment_2_Stamps_S.xlsx",
        penalty=1.0, min_size=10, buffer_sec=99999, window_sec=30, step_sec=1,
        affective_set={"Crash", "Barrel", "Fog Start", "Brake"})
    generate_figure(d, args.output)
