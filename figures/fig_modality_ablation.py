#!/usr/bin/env python3
"""
Modality ablation figure (2-panel).

  (a) Solo-modality AUC bars vs full-fusion baseline
  (b) Per-event solo AUC heatmap

Loads pre-computed results from loo_cache.json.

Usage:
    MPLBACKEND=Agg python polished/fig_modality_ablation.py [--replot]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
except ImportError:
    pass

CACHE_PATH = "output/modality_loo/loo_cache.json"
OUT_DIR = "output/figures"
COL_W = 3.5

MOD_ORDER = ["ECG", "PPG", "Resp", "EDA", "Temp"]
MOD_COLORS = {"ECG": "#1f77b4", "PPG": "#ff7f0e", "Resp": "#2ca02c",
              "EDA": "#9467bd", "Temp": "#d62728"}
EVENT_ORDER = ["Barrel", "Brake", "Crash", "Slow Car"]


def _load(replot=False):
    if replot and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    from scripts.plot_modality_loo import _compute_loo
    return _compute_loo()


def _plot(results):
    base = results["baseline"]
    solo = results.get("solo", {})

    baseline_overall = 0.73  # hardcoded from main detection figure (trimmed)

    # Solo overall AUC per modality
    solo_overall = {}
    for mod in MOD_ORDER:
        if mod not in solo:
            continue
        aucs = [solo[mod][e]["mean_auc"] for e in EVENT_ORDER if e in solo[mod]]
        if aucs:
            solo_overall[mod] = np.mean(aucs)

    # Per-subject stats
    base_subj = {}
    for evt, info in base.items():
        if evt not in EVENT_ORDER:
            continue
        for sid, auc in zip(info["per_subj_sids"], info["per_subj_aucs"]):
            base_subj.setdefault(sid, []).append(auc)
    base_mean = {s: np.mean(a) for s, a in base_subj.items()}

    mod_stats = {}
    for mod in MOD_ORDER:
        if mod not in solo:
            continue
        solo_subj = {}
        for evt, info in solo[mod].items():
            if evt not in EVENT_ORDER:
                continue
            for sid, auc in zip(info["per_subj_sids"], info["per_subj_aucs"]):
                solo_subj.setdefault(sid, []).append(auc)
        solo_mean = {s: np.mean(a) for s, a in solo_subj.items()}
        common = sorted(set(base_mean) & set(solo_mean))
        if not common:
            continue
        b = np.array([base_mean[s] for s in common])
        s = np.array([solo_mean[s] for s in common])
        _, p_t = ttest_rel(b, s)
        mod_stats[mod] = dict(solo_aucs=s.tolist(), sem=float(s.std() / np.sqrt(len(s))),
                              p_ttest=float(p_t), n=len(common))

    sorted_mods = sorted(solo_overall, key=lambda m: solo_overall[m], reverse=True)

    pe_matrix = {}
    for mod in MOD_ORDER:
        if mod not in solo:
            continue
        for evt in EVENT_ORDER:
            if evt in solo[mod]:
                pe_matrix[(mod, evt)] = solo[mod][evt]["mean_auc"]

    with plt.style.context(["science", "ieee", "no-latex"]):
        plt.rcParams.update({"font.size": 10, "axes.labelsize": 11,
                             "xtick.labelsize": 10, "ytick.labelsize": 10,
                             "legend.fontsize": 9, "font.family": "serif",
                             "axes.linewidth": 0.8})

        fig, (ax_bar, ax_heat) = plt.subplots(
            1, 2, figsize=(2 * COL_W, COL_W * 0.95),
            gridspec_kw={"width_ratios": [1.2, 1]})

        # Panel (a)
        x = np.arange(len(sorted_mods))
        means = [solo_overall[m] for m in sorted_mods]
        sems = [mod_stats[m]["sem"] for m in sorted_mods]
        colors = [MOD_COLORS[m] for m in sorted_mods]

        ax_bar.bar(x, means, width=0.55, color=colors, alpha=0.75,
                   edgecolor=colors, lw=0.8, zorder=3)
        ax_bar.errorbar(x, means, yerr=sems, fmt="none", ecolor="black",
                        elinewidth=0.8, capsize=3, capthick=0.8, zorder=4)
        rng = np.random.default_rng(42)
        for i, mod in enumerate(sorted_mods):
            aucs = mod_stats[mod]["solo_aucs"]
            ax_bar.scatter(i + rng.uniform(-0.15, 0.15, len(aucs)), aucs,
                           s=10, alpha=0.4, color=MOD_COLORS[mod], edgecolors="none", zorder=5)
        ax_bar.axhline(baseline_overall, color="black", ls="-", lw=1.2, zorder=2,
                       label=f"All modalities ({baseline_overall:.2f})")
        ax_bar.axhline(0.5, color="gray", ls="--", lw=0.6, zorder=1, label="Chance (0.50)")
        for i, mod in enumerate(sorted_mods):
            ax_bar.text(i, means[i] + sems[i] + 0.01, f"{means[i]:.02f}",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(sorted_mods)
        ax_bar.set_xlabel("Modality (Solo)")
        ax_bar.set_ylabel("Mean AUC")
        ax_bar.set_ylim(0.4, max(means) + 0.15)
        ax_bar.legend(loc="lower right", fontsize=7, framealpha=0.8)
        ax_bar.text(-0.02, 1.04, "(a)", transform=ax_bar.transAxes,
                    fontsize=11, fontweight="bold", va="bottom")

        # Panel (b)
        present_evts = [e for e in EVENT_ORDER
                        if any((m, e) in pe_matrix for m in MOD_ORDER)]
        z = np.zeros((len(MOD_ORDER), len(present_evts)))
        for i, mod in enumerate(MOD_ORDER):
            for j, evt in enumerate(present_evts):
                z[i, j] = pe_matrix.get((mod, evt), 0.0)

        im = ax_heat.imshow(z, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=0.85)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                tc = "white" if z[i, j] < 0.5 or z[i, j] > 0.75 else "black"
                ax_heat.text(j, i, f"{z[i,j]:.02f}", ha="center", va="center",
                             fontsize=10, color=tc, fontweight="bold")
        ax_heat.set_xticks(range(len(present_evts)))
        ax_heat.set_xticklabels(present_evts, rotation=30, ha="right")
        ax_heat.set_yticks(range(len(MOD_ORDER)))
        ax_heat.set_yticklabels(MOD_ORDER)
        cb = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cb.set_label("AUC", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax_heat.text(-0.02, 1.04, "(b)", transform=ax_heat.transAxes,
                     fontsize=11, fontweight="bold", va="bottom")

        plt.tight_layout()
        os.makedirs(OUT_DIR, exist_ok=True)
        for fmt in ("png", "pdf"):
            fig.savefig(os.path.join(OUT_DIR, f"fig_modality_loo.{fmt}"),
                        dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {OUT_DIR}/fig_modality_loo.png/.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    _plot(_load(replot=args.replot))
