#!/usr/bin/env python3
"""
Permutation-test null distribution figure (1×4 panel).

Panels:
  (a) Affective vs Non-Event
  (b) Affective vs Practice
  (c) Affective vs Video
  (d) Experiment-only affective vs non-event

Loads pre-computed results from multisubject_cache.json.

Usage:
    MPLBACKEND=Agg python polished/fig_permutation.py [--dataset surprise]
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

EVENT_MERGE_MAP = {
    "Brake 1": "Brake", "Brake 2": "Brake", "Brake 3": "Brake",
    "Brake 4": "Brake", "Brake 5": "Brake",
    "Single Slow Car": "Slow Car", "Double Slow Car": "Slow Car",
}

CACHE_PATHS = {
    "surprise": "output/causal_validation_figures/multisubject_cache.json",
    "irritation": "output/causal_validation_irritation/multisubject_cache.json",
}


def _merge_events(evts):
    seen = set()
    merged = []
    for e in evts:
        c = EVENT_MERGE_MAP.get(e, e)
        if c not in seen:
            seen.add(c)
            merged.append(c)
    return merged


def _load_cache(dataset="surprise"):
    cp = CACHE_PATHS[dataset]
    if not os.path.exists(cp):
        sys.exit(f"Cache not found: {cp}")
    with open(cp) as f:
        c = json.load(f)
    for r in c["all_seg_records"]:
        r["events"] = _merge_events(r["events"])
    for r in c["per_subject"]:
        for k in ("consensus", "aff_mask", "null_means"):
            if k in r and isinstance(r[k], list):
                r[k] = np.array(r[k])
    null_aff = np.array(c["null_aff"])
    null_exp = np.array(c["null_exp"]) if c["null_exp"] else np.array([])
    p_exp = c["p_exp"] if c["p_exp"] is not None else np.nan
    phase_tests = {}
    for k, v in c.get("phase_tests", {}).items():
        v["null"] = np.array(v["null"])
        phase_tests[k] = v
    return c, null_aff, null_exp, p_exp, phase_tests


def plot_permutation(c, null_aff, null_exp, p_exp, phase_tests, out_dir):
    plt.style.use(["science", "ieee", "no-latex"])
    COL_W = 3.5
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "font.family": "serif",
        "axes.linewidth": 0.8, "lines.linewidth": 1.5,
    })

    prac = phase_tests.get("Practice")
    vid = phase_tests.get("Video")

    panels = [
        dict(label="(a)", null=null_aff,
             obs_aff=c["obs_aff_mean"], obs_other=c["obs_non_mean"],
             p=c["p_pooled"], r_rb=c.get("r_rb_pooled", 0.0),
             other_label="Non-Event", hist_color="#4a90d9"),
    ]
    if prac:
        panels.append(dict(label="(b)", null=np.array(prac["null"]),
                           obs_aff=prac["obs_aff"], obs_other=prac["obs_other"],
                           p=prac["p"], r_rb=prac.get("r_rb", 0.0),
                           other_label="Practice", hist_color="#ABEBC6"))
    else:
        panels.append(None)
    if vid:
        panels.append(dict(label="(c)", null=np.array(vid["null"]),
                           obs_aff=vid["obs_aff"], obs_other=vid["obs_other"],
                           p=vid["p"], r_rb=vid.get("r_rb", 0.0),
                           other_label="Video", hist_color="#AED6F1"))
    else:
        panels.append(None)
    if len(null_exp) > 0 and not np.isnan(p_exp):
        panels.append(dict(label="(d)", null=null_exp,
                           obs_aff=c["obs_exp_aff"], obs_other=c["obs_exp_non"],
                           p=p_exp, r_rb=c.get("r_rb_exp", 0.0),
                           other_label="Non-Event (exp)", hist_color="#F0B27A"))
    else:
        panels.append(None)

    fig, axes = plt.subplots(1, 4, figsize=(2 * COL_W, 2.5), sharey=False)

    for idx, (ax, panel) in enumerate(zip(axes, panels)):
        ax.set_box_aspect(1)
        if panel is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#888888")
            ax.set_title(f"({'abcd'[idx]})", fontsize=10, fontweight="bold", loc="left")
            ax.set_xlabel("RDI")
            if idx == 0:
                ax.set_ylabel("Count")
            continue

        ax.hist(panel["null"], bins=60, color=panel["hist_color"], alpha=0.6,
                edgecolor="white", linewidth=0.3)
        ax.axvline(panel["obs_aff"], color="#c0392b", lw=1.5, zorder=5)
        ax.axvline(panel["obs_other"], color="#27ae60", lw=1.2, ls="--", zorder=5)

        sig = " ***" if panel["p"] < 0.001 else " **" if panel["p"] < 0.01 else " *" if panel["p"] < 0.05 else ""
        aff_txt = (f"Affective = {panel['obs_aff']:.4f}\n"
                   f"p = {panel['p']:.4f}, $r_{{rb}}$ = {panel['r_rb']:.3f}{sig}")
        oth_txt = f"{panel['other_label']} = {panel['obs_other']:.4f}"

        ax.text(0.97, 0.95, aff_txt, transform=ax.transAxes, fontsize=6,
                color="#c0392b", weight="bold", ha="right", va="top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c0392b", alpha=0.9, lw=0.6))
        ax.text(0.03, 0.65, oth_txt, transform=ax.transAxes, fontsize=6,
                color="#27ae60", weight="bold", ha="left", va="top", zorder=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#27ae60", alpha=0.9, lw=0.6))
        ax.set_title(panel["label"], fontsize=10, fontweight="bold", loc="left", pad=6)
        ax.set_xlabel("RDI")
        if idx == 0:
            ax.set_ylabel("Count")

    fig.tight_layout(w_pad=1.0)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"fig_permutation_multisubject_v2.{ext}")
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_dir}/fig_permutation_multisubject_v2.png/.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="surprise",
                        choices=["surprise", "irritation"])
    args = parser.parse_args()

    out_dir = os.path.dirname(CACHE_PATHS[args.dataset])
    c, null_aff, null_exp, p_exp, phase_tests = _load_cache(args.dataset)
    plot_permutation(c, null_aff, null_exp, p_exp, phase_tests, out_dir)
