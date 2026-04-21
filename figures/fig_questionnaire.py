#!/usr/bin/env python3
"""
Questionnaire Before/After — Quadrant-mean summary for both experiments.

Panels:
  (a) Exp 1: Irritation
  (b) Exp 2: Surprise

Produces: output/questionnaire_quadrant_summary.{png,pdf}
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats

try:
    import scienceplots  # noqa: F401
    STYLE = ["science", "no-latex"]
except Exception:
    STYLE = []

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUT_DIR.mkdir(exist_ok=True)
DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "Questionnaire" / "Questionaire_Data_paper.xlsx"

EXPERIMENTS = {
    "Exp 1: Irritation": {"before": "Relaxation IR", "after": "Scene IR"},
    "Exp 2: Surprise":   {"before": "Relaxation SP", "after": "Scene SP"},
}
QUADRANT_ORDER = {
    "Neg. Valence\nHigh Arousal": ["Distressed", "Impatient", "Tense", "Scared", "Irritable"],
    "Pos. Valence\nHigh Arousal": ["Interested", "Funny", "Surprised", "Excited", "Alert"],
    "Neg. Valence\nLow Arousal":  ["Anxious", "Fatigued", "Bored", "Nervous", "Ashamed"],
    "Pos. Valence\nLow Arousal":  ["Sleepy", "Relaxed", "Confident", "Pleased", "Attentive"],
}

all_sheets = pd.read_excel(DATA_PATH, sheet_name=None)


def _extract_paired(emotion, before_col, after_col):
    b_vals, a_vals = [], []
    for df in all_sheets.values():
        norm_cols = {c.strip().lower(): c for c in df.columns}
        bc = norm_cols.get(before_col.strip().lower())
        ac = norm_cols.get(after_col.strip().lower())
        if bc is None or ac is None:
            continue
        label_col = df.columns[0]
        mask = df[label_col].astype(str).str.strip().str.lower() == emotion.strip().lower()
        row = df.loc[mask]
        if row.empty:
            continue
        bv = pd.to_numeric(row[bc].values[0], errors="coerce")
        av = pd.to_numeric(row[ac].values[0], errors="coerce")
        if pd.notna(bv) and pd.notna(av):
            b_vals.append(bv)
            a_vals.append(av)
    return np.array(b_vals), np.array(a_vals)


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def _holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.ones(m)
    for rank, idx in enumerate(order):
        adjusted[idx] = pvals[idx] * (m - rank)
    adjusted = np.minimum(adjusted, 1.0)
    running_max = 0.0
    for idx in order:
        running_max = max(running_max, adjusted[idx])
        adjusted[idx] = running_max
    return adjusted.tolist()


# ── Build figure ─────────────────────────────────────────────────────────
COL_W = 3.5
COLORS_BEFORE, COLORS_AFTER = "#3498db", "#e74c3c"

with plt.style.context(STYLE):
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "font.family": "serif",
        "axes.linewidth": 0.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(2 * COL_W, 3.0), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for idx, (ax, (exp_name, cols)) in enumerate(zip(axes, EXPERIMENTS.items())):
        q_names, b_means, a_means, b_sems, a_sems = [], [], [], [], []
        raw_p = []
        for qname, emos in QUADRANT_ORDER.items():
            b_all, a_all = [], []
            for emo in emos:
                b, a = _extract_paired(emo, cols["before"], cols["after"])
                b_all.extend(b.tolist())
                a_all.extend(a.tolist())
            b_arr, a_arr = np.array(b_all), np.array(a_all)
            b_means.append(np.mean(b_arr))
            a_means.append(np.mean(a_arr))
            b_sems.append(np.std(b_arr, ddof=1) / np.sqrt(len(b_arr)))
            a_sems.append(np.std(a_arr, ddof=1) / np.sqrt(len(a_arr)))
            q_names.append(qname)
            n = min(len(b_arr), len(a_arr))
            try:
                _, p = stats.wilcoxon(b_arr[:n], a_arr[:n]) if n >= 5 else (0, 1.0)
            except ValueError:
                p = 1.0
            raw_p.append(p)

        adj_p = _holm_bonferroni(raw_p)
        stars_list = [_sig_stars(p) for p in adj_p]

        x = np.arange(len(q_names))
        w = 0.32
        ax.bar(x - w/2, b_means, w, yerr=b_sems, capsize=3,
               color=COLORS_BEFORE, edgecolor="gray", lw=0.5,
               label="Before", error_kw=dict(lw=0.8))
        ax.bar(x + w/2, a_means, w, yerr=a_sems, capsize=3,
               color=COLORS_AFTER, edgecolor="gray", lw=0.5,
               label="After", error_kw=dict(lw=0.8))

        for i, star in enumerate(stars_list):
            if star:
                y_max = max(b_means[i] + b_sems[i], a_means[i] + a_sems[i])
                ax.text(i, y_max + 0.1, star, ha="center", va="bottom",
                        fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(q_names, rotation=20, ha="right")
        ax.set_ylabel("Mean Rating (1\u20135)")
        ax.set_ylim(0, 5.5)
        ax.legend(loc="upper right")
        ax.set_title(f"({'ab'[idx]})", fontsize=10, fontweight="bold", loc="left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = OUT_DIR / f"questionnaire_quadrant_summary.{ext}"
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT_DIR / 'questionnaire_quadrant_summary.png'}")
    plt.close(fig)
