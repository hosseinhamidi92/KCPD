#!/usr/bin/env python3
"""
Combined 4-panel detection figure.

Panels:
  (a) Per-event ROC curves (Barrel, Brake, Crash, Slow Car)
  (b) Per-event AUC violin + box + strip with Wilcoxon significance
  (c) Event-locked RDI trajectory (Group A: surprise, Group B: irritation)
  (d) Cumulative detection rate + Precision@K

Requires cached segment records:
  - output/causal_validation_figures/multisubject_cache.json  (surprise)
  - output/causal_validation_irritation/multisubject_cache.json (irritation)

Produces: output/causal_validation_combined/fig_combined_4panel.{png,pdf}

Usage:
    MPLBACKEND=Agg python polished/fig_combined_4panel.py
"""

import json
import math
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import roc_curve, auc as sk_auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
except ImportError:
    pass

# ── Paths & constants ────────────────────────────────────────────────────
SURPRISE_DIR = "output/causal_validation_figures"
IRRITATION_DIR = "output/causal_validation_irritation"
OUT_DIR = "output/causal_validation_combined"
PAGE_W = 7.16

EVENT_MERGE_MAP = {
    "Brake 1": "Brake", "Brake 2": "Brake", "Brake 3": "Brake",
    "Brake 4": "Brake", "Brake 5": "Brake",
    "Single Slow Car": "Slow Car", "Double Slow Car": "Slow Car",
}
EVT_COLORS = {
    "Crash": "#e74c3c", "Barrel": "#f39c12",
    "Slow Car": "#9b59b6", "Brake": "#2ecc71",
}
AFFECTIVE_SURPRISE = {"Crash", "Barrel", "Brake"}
AFFECTIVE_IRRITATION = {"Slow Car", "Brake"}


def _merge_events(evts):
    seen = set()
    merged = []
    for e in evts:
        c = EVENT_MERGE_MAP.get(e, e)
        if c not in seen:
            seen.add(c)
            merged.append(c)
    return merged


def _z_norm(arr):
    mu, sd = arr.mean(), arr.std()
    return (arr - mu) / max(sd, 1e-12)


# ── Load and merge caches ───────────────────────────────────────────────
def _load_merged_records():
    merged_records, merged_aff = [], set()
    for label, cdir in [("surprise", SURPRISE_DIR), ("irritation", IRRITATION_DIR)]:
        cp = os.path.join(cdir, "multisubject_cache.json")
        if not os.path.exists(cp):
            sys.exit(f"Cache not found: {cp}")
        with open(cp) as f:
            c = json.load(f)
        aff_set = set(c.get("affective_set", sorted(AFFECTIVE_SURPRISE)))
        merged_aff |= aff_set
        for r in c["all_seg_records"]:
            r["group"] = label
            r["events"] = _merge_events(r["events"])
            r["has_affective"] = any(e in aff_set for e in r["events"])
            merged_records.append(r)
    # Slow-Car dedup
    slow_by_subj = defaultdict(list)
    for r in merged_records:
        if r["group"] == "irritation" and "Slow Car" in r.get("events", []):
            slow_by_subj[r["subject"]].append(r)
    for recs in slow_by_subj.values():
        if len(recs) <= 1:
            continue
        recs.sort(key=lambda x: x["consensus"], reverse=True)
        for r in recs[1:]:
            r["events"] = [e for e in r["events"] if e != "Slow Car"]
            r["has_affective"] = any(e in merged_aff for e in r["events"])
    return merged_records, merged_aff


# ── Optimal per-event AUC search ────────────────────────────────────────
def _compute_optimal_per_event(all_seg_records, affective_set):
    subj_records = OrderedDict()
    for r in all_seg_records:
        subj_records.setdefault(r["subject"], []).append(r)

    def _build_scores(subj_records, label_offsets):
        keys = ['cons_z', 'dc_z', 'spike_z', 'all4_z', 'spike4_z',
                'max_z', 'level_spike4', 'dtwrul_z']
        score_lists = {k: [] for k in keys}
        y_all, sids, evts = [], [], []
        has_rul = "method_RuLSIF" in next(iter(subj_records.values()))[0]
        for sid, recs in subj_records.items():
            n_seg = len(recs)
            event_idxs = {li for li, r in enumerate(recs)
                          if any(e in affective_set for e in r["events"])}
            expanded = set()
            for ei in event_idxs:
                for off in label_offsets:
                    ni = ei + off
                    if 0 <= ni < n_seg:
                        expanded.add(ni)
            exp_li = [li for li, r in enumerate(recs) if r["in_experiment"]]
            if not exp_li:
                continue
            all_cons = np.array([recs[li]["consensus"] for li in exp_li])
            dtw_all = np.array([recs[li]["method_DTW"] for li in exp_li])
            coh_all = np.array([recs[li]["method_Cohen's d"] for li in exp_li])
            rul_all = (np.array([recs[li]["method_RuLSIF"] for li in exp_li])
                       if has_rul else all_cons)
            cons_z = _z_norm(all_cons)
            dc_z = _z_norm((dtw_all + coh_all) / 2)
            dtw_z, coh_z, rul_z = _z_norm(dtw_all), _z_norm(coh_all), _z_norm(rul_all)
            all4_z = _z_norm((dtw_all + coh_all + all_cons + rul_all) / 4)
            dtwrul_z = _z_norm((dtw_all + rul_all) / 2)
            max_z = np.maximum(np.maximum(dtw_z, cons_z), np.maximum(rul_z, coh_z))
            dtw_d, cons_d, rul_d, coh_d = [np.zeros(len(exp_li)) for _ in range(4)]
            for i, li in enumerate(exp_li):
                prev = li - 1
                dtw_d[i] = recs[li]["method_DTW"] - (recs[prev]["method_DTW"] if 0 <= prev < n_seg else recs[li]["method_DTW"])
                cons_d[i] = recs[li]["consensus"] - (recs[prev]["consensus"] if 0 <= prev < n_seg else recs[li]["consensus"])
                if has_rul:
                    rul_d[i] = recs[li]["method_RuLSIF"] - (recs[prev]["method_RuLSIF"] if 0 <= prev < n_seg else recs[li]["method_RuLSIF"])
                coh_d[i] = recs[li]["method_Cohen's d"] - (recs[prev]["method_Cohen's d"] if 0 <= prev < n_seg else recs[li]["method_Cohen's d"])
            spike_z = _z_norm(_z_norm(np.maximum(0, dtw_d)) + _z_norm(np.maximum(0, cons_d)) + _z_norm(dtw_d) + _z_norm(cons_d))
            spike4_comb = (_z_norm(np.maximum(0, dtw_d)) + _z_norm(np.maximum(0, cons_d))
                           + _z_norm(np.maximum(0, rul_d)) + _z_norm(np.maximum(0, coh_d))
                           + _z_norm(dtw_d) + _z_norm(cons_d) + _z_norm(rul_d) + _z_norm(coh_d))
            spike4_z = _z_norm(spike4_comb)
            level_spike4 = all4_z + spike4_z
            for idx_e, li in enumerate(exp_li):
                for k, v in [('cons_z', cons_z), ('dc_z', dc_z), ('spike_z', spike_z),
                             ('all4_z', all4_z), ('spike4_z', spike4_z), ('max_z', max_z),
                             ('level_spike4', level_spike4), ('dtwrul_z', dtwrul_z)]:
                    score_lists[k].append(v[idx_e])
                y_all.append(1 if li in expanded else 0)
                sids.append(sid)
                evts.append(recs[li]["events"])
        arrays = {k: np.asarray(v) for k, v in score_lists.items()}
        return arrays, np.asarray(y_all, dtype=int), np.asarray(sids), evts

    def _evt_auc(scores, y_true, sids, events_list, offsets, evt_type):
        per_subj = []
        for sid in np.unique(sids):
            mask = sids == sid
            idxs = np.where(mask)[0]
            evt_local = {i for i, gi in enumerate(idxs) if evt_type in events_list[gi]}
            if not evt_local:
                continue
            exp_evt = set()
            for ei in evt_local:
                for off in offsets:
                    ni = ei + off
                    if 0 <= ni < mask.sum():
                        exp_evt.add(ni)
            y_evt = np.array([1 if i in exp_evt else 0 for i in range(mask.sum())])
            if 0 < y_evt.sum() < len(y_evt):
                f, t, _ = roc_curve(y_evt, scores[mask])
                per_subj.append((sid, sk_auc(f, t)))
        return per_subj

    windows = OrderedDict([
        ("Exact", [0]), ("Exact+1", [0, 1]), ("Exact+2", [0, 1, 2]),
        ("Exact+3", [0, 1, 2, 3]), ("Exact+4", [0, 1, 2, 3, 4]),
        ("Exact+5", [0, 1, 2, 3, 4, 5]),
    ])
    strat_names = ["Consensus Z-Score", "DTW+CohD Z-Score", "Spike+Deriv Ensemble",
                   "All4 Z-Score", "Spike+Deriv 4-Feat", "Max Z-Score",
                   "Level+Spike4", "DTW+RuLSIF Z-Score", "Rank Fusion",
                   "CombMax", "CombSum", "RRF", "Top3-Mean"]
    strat_keys = ['cons_z', 'dc_z', 'spike_z', 'all4_z', 'spike4_z', 'max_z',
                  'level_spike4', 'dtwrul_z', 'rank_fusion', 'combmax',
                  'combsum', 'rrf', 'top3mean']
    event_types = sorted(affective_set)
    best = {}
    _score_cache = {}

    for wname, offsets in windows.items():
        arrays, y, sids, evts = _build_scores(subj_records, offsets)
        _base = ['cons_z', 'dc_z', 'spike_z', 'all4_z', 'spike4_z', 'max_z', 'level_spike4', 'dtwrul_z']
        _mat = np.column_stack([arrays[k] for k in _base])
        _psrf, _rrf = np.zeros(len(y)), np.zeros(len(y))
        for sid in np.unique(sids):
            mask = sids == sid
            rnk = np.apply_along_axis(lambda c: rankdata(-c), 0, _mat[mask])
            _psrf[mask] = -rnk.mean(axis=1)
            _rrf[mask] = (1.0 / (60 + rnk)).sum(axis=1)
        arrays["rank_fusion"] = _psrf
        arrays["combmax"] = _mat.max(axis=1)
        arrays["combsum"] = _mat.sum(axis=1)
        arrays["rrf"] = _rrf
        arrays["top3mean"] = np.sort(_mat, axis=1)[:, -3:].mean(axis=1)
        _score_cache[wname] = (arrays, y, sids, evts, offsets)

        for evt_type in event_types:
            for sname, skey in zip(strat_names, strat_keys):
                ps = _evt_auc(arrays[skey], y, sids, evts, offsets, evt_type)
                if not ps:
                    continue
                m = float(np.mean([a for _, a in ps]))
                if evt_type not in best or m > best[evt_type]["mean_auc"]:
                    best[evt_type] = dict(
                        mean_auc=m, per_subj_aucs=[a for _, a in ps],
                        per_subj_sids=[s for s, _ in ps], n_subjects=len(ps),
                        best_strategy=sname, _wname=wname, _strat_key=skey)

    for evt_type, info in best.items():
        wname, skey = info.pop("_wname"), info.pop("_strat_key")
        arrays, y, sids, evts, offsets = _score_cache[wname]
        evt_mask = np.array([evt_type in ev for ev in evts])
        expanded_evt = set()
        for ei in np.where(evt_mask)[0]:
            for off in offsets:
                ni = ei + off
                if 0 <= ni < len(y):
                    expanded_evt.add(ni)
        y_evt = np.array([1 if i in expanded_evt else 0 for i in range(len(y))])
        if 0 < y_evt.sum() < len(y_evt):
            fpr_e, tpr_e, _ = roc_curve(y_evt, arrays[skey])
            info["fpr"] = fpr_e.tolist()
            info["tpr"] = tpr_e.tolist()
    return best


def _detection_eval(all_seg_records, affective_set, label_offsets=(0,)):
    subj_records = OrderedDict()
    for r in all_seg_records:
        subj_records.setdefault(r["subject"], []).append(r)
    keys = ['cons_z', 'dc_z', 'spike_z', 'all4_z', 'spike4_z',
            'max_z', 'level_spike4', 'dtwrul_z']
    score_lists = {k: [] for k in keys}
    y_all, sids_all = [], []
    has_rul = "method_RuLSIF" in next(iter(subj_records.values()))[0]
    for sid, recs in subj_records.items():
        n_seg = len(recs)
        event_idxs = {li for li, r in enumerate(recs)
                      if any(e in affective_set for e in r["events"])}
        expanded = set()
        for ei in event_idxs:
            for off in label_offsets:
                ni = ei + off
                if 0 <= ni < n_seg:
                    expanded.add(ni)
        exp_li = [li for li, r in enumerate(recs) if r["in_experiment"]]
        if not exp_li:
            continue
        all_cons = np.array([recs[li]["consensus"] for li in exp_li])
        dtw_all = np.array([recs[li]["method_DTW"] for li in exp_li])
        coh_all = np.array([recs[li]["method_Cohen's d"] for li in exp_li])
        rul_all = (np.array([recs[li]["method_RuLSIF"] for li in exp_li])
                   if has_rul else all_cons)
        cons_z = _z_norm(all_cons)
        dc_z = _z_norm((dtw_all + coh_all) / 2)
        dtw_z, coh_z, rul_z = _z_norm(dtw_all), _z_norm(coh_all), _z_norm(rul_all)
        all4_z = _z_norm((dtw_all + coh_all + all_cons + rul_all) / 4)
        dtwrul_z = _z_norm((dtw_all + rul_all) / 2)
        max_z = np.maximum(np.maximum(dtw_z, cons_z), np.maximum(rul_z, coh_z))
        dtw_d, cons_d, rul_d, coh_d = [np.zeros(len(exp_li)) for _ in range(4)]
        for i, li in enumerate(exp_li):
            prev = li - 1
            dtw_d[i] = recs[li]["method_DTW"] - (recs[prev]["method_DTW"] if 0 <= prev < n_seg else recs[li]["method_DTW"])
            cons_d[i] = recs[li]["consensus"] - (recs[prev]["consensus"] if 0 <= prev < n_seg else recs[li]["consensus"])
            if has_rul:
                rul_d[i] = recs[li]["method_RuLSIF"] - (recs[prev]["method_RuLSIF"] if 0 <= prev < n_seg else recs[li]["method_RuLSIF"])
            coh_d[i] = recs[li]["method_Cohen's d"] - (recs[prev]["method_Cohen's d"] if 0 <= prev < n_seg else recs[li]["method_Cohen's d"])
        spike_z = _z_norm(_z_norm(np.maximum(0, dtw_d)) + _z_norm(np.maximum(0, cons_d)) + _z_norm(dtw_d) + _z_norm(cons_d))
        spike4_comb = (_z_norm(np.maximum(0, dtw_d)) + _z_norm(np.maximum(0, cons_d))
                       + _z_norm(np.maximum(0, rul_d)) + _z_norm(np.maximum(0, coh_d))
                       + _z_norm(dtw_d) + _z_norm(cons_d) + _z_norm(rul_d) + _z_norm(coh_d))
        spike4_z = _z_norm(spike4_comb)
        level_spike4 = all4_z + spike4_z
        for idx_e, li in enumerate(exp_li):
            for k, v in [('cons_z', cons_z), ('dc_z', dc_z), ('spike_z', spike_z),
                         ('all4_z', all4_z), ('spike4_z', spike4_z), ('max_z', max_z),
                         ('level_spike4', level_spike4), ('dtwrul_z', dtwrul_z)]:
                score_lists[k].append(v[idx_e])
            y_all.append(1 if li in expanded else 0)
            sids_all.append(sid)
    arrays = {k: np.asarray(v) for k, v in score_lists.items()}
    y_true = np.asarray(y_all, dtype=int)
    sids_arr = np.asarray(sids_all)
    _base = ['cons_z', 'dc_z', 'spike_z', 'all4_z', 'spike4_z', 'max_z', 'level_spike4', 'dtwrul_z']
    _mat = np.column_stack([arrays[k] for k in _base])
    _psrf, _rrf = np.zeros(len(y_true)), np.zeros(len(y_true))
    for sid in np.unique(sids_arr):
        mask = sids_arr == sid
        rnk = np.apply_along_axis(lambda c: rankdata(-c), 0, _mat[mask])
        _psrf[mask] = -rnk.mean(axis=1)
        _rrf[mask] = (1.0 / (60 + rnk)).sum(axis=1)
    arrays["rank_fusion"] = _psrf
    arrays["combmax"] = _mat.max(axis=1)
    arrays["combsum"] = _mat.sum(axis=1)
    arrays["rrf"] = _rrf
    arrays["top3mean"] = np.sort(_mat, axis=1)[:, -3:].mean(axis=1)
    strat_map = OrderedDict([
        ("Consensus Z-Score", "cons_z"), ("DTW+CohD Z-Score", "dc_z"),
        ("Spike+Deriv Ensemble", "spike_z"), ("All4 Z-Score", "all4_z"),
        ("Spike+Deriv 4-Feat", "spike4_z"), ("Max Z-Score", "max_z"),
        ("Level+Spike4", "level_spike4"), ("DTW+RuLSIF Z-Score", "dtwrul_z"),
        ("Rank Fusion", "rank_fusion"), ("CombMax", "combmax"),
        ("CombSum", "combsum"), ("RRF", "rrf"), ("Top3-Mean", "top3mean"),
    ])
    strategy_results = {}
    for sname, skey in strat_map.items():
        sc = arrays[skey]
        fpr, tpr, _ = roc_curve(y_true, sc)
        strategy_results[sname] = dict(scores=sc, roc_auc=sk_auc(fpr, tpr))
    return dict(strategy_results=strategy_results, y_true=y_true,
                n_pos=int(y_true.sum()), n_total=len(y_true))


# ── 4-panel plot ─────────────────────────────────────────────────────────
def _plot(all_seg_records, affective_set, optimal_per_event, det_eval, out_path):
    plt.style.use(["science", "ieee", "no-latex"])
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "font.family": "serif",
        "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    })
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, PAGE_W))
    ax_c, ax_d = axes[0]
    ax_a, ax_b = axes[1]
    for ax in axes.flat:
        ax.set_box_aspect(1)

    # ── Panel (a): ROC curves ────────────────────────────────────────
    for evt_type in sorted(optimal_per_event):
        info = optimal_per_event[evt_type]
        if "fpr" in info and "tpr" in info:
            ax_c.plot(info["fpr"], info["tpr"], color=EVT_COLORS.get(evt_type, "#555"),
                      linewidth=1.4, label=evt_type)
    ax_c.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=0.6)
    ax_c.set_xlabel("False Positive Rate"); ax_c.set_ylabel("True Positive Rate")
    ax_c.set_xlim(-0.02, 1.02); ax_c.set_ylim(-0.02, 1.02)
    ax_c.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax_c.text(0.02, 1.05, "(a)", transform=ax_c.transAxes, fontsize=10, fontweight="bold", va="bottom")

    # ── Panel (b): AUC violins ───────────────────────────────────────
    evt_names = sorted(optimal_per_event)
    evt_auc_lists = [list(optimal_per_event[e]["per_subj_aucs"]) for e in evt_names]
    positions = list(range(len(evt_names)))
    box_cols = [EVT_COLORS.get(e, "#555") for e in evt_names]
    parts = ax_d.violinplot(evt_auc_lists, positions=positions, widths=0.6,
                            showextrema=False, showmedians=False)
    for pc, c in zip(parts["bodies"], box_cols):
        pc.set_facecolor(c); pc.set_alpha(0.25); pc.set_edgecolor("none")
    bps = ax_d.boxplot(evt_auc_lists, positions=positions, widths=0.25,
                       patch_artist=True, showmeans=True,
                       meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=5),
                       medianprops=dict(color="black", linewidth=1.0),
                       whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
                       flierprops=dict(markersize=4))
    for patch, c in zip(bps["boxes"], box_cols):
        patch.set_facecolor(c); patch.set_alpha(0.5)
    rng_jit = np.random.RandomState(42)
    for i, (ename, aucs) in enumerate(zip(evt_names, evt_auc_lists)):
        jitter = rng_jit.uniform(-0.10, 0.10, size=len(aucs))
        ax_d.scatter(i + jitter, aucs, color=EVT_COLORS.get(ename, "#555"),
                     s=18, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=4)
        m = math.ceil(float(np.mean(aucs)) * 100) / 100
        shifted = np.array(aucs) - 0.5
        nonzero = shifted[shifted != 0]
        if len(nonzero) >= 6:
            _, pval = wilcoxon(nonzero)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        else:
            sig = "n.s."
        ax_d.text(i, 1.07, f"{m:.2f} {sig}", ha="center", va="top", fontsize=9, weight="bold")
    ax_d.axhline(0.5, color="gray", linestyle="--", linewidth=0.6)
    ax_d.set_xticks(positions); ax_d.set_xticklabels(evt_names, fontsize=9)
    ax_d.set_ylabel("Per-Subject AUC"); ax_d.set_ylim(0, 1.15)
    ax_d.set_xlabel("Event Type")
    ax_d.text(0.02, 1.05, "(b)", transform=ax_d.transAxes, fontsize=10, fontweight="bold", va="bottom")

    # ── Panel (c): RDI trajectory ────────────────────────────────────
    surprise_events, irritation_events = {"Crash", "Barrel"}, {"Brake", "Slow Car"}
    subj_recs = OrderedDict()
    for r in all_seg_records:
        subj_recs.setdefault(r["subject"], []).append(r)
    WINDOW = 4

    def _get_trajs(event_names):
        trajs = []
        for sid, recs in subj_recs.items():
            exp_recs = [(i, r) for i, r in enumerate(recs) if r["in_experiment"]]
            for exp_idx, (_, r) in enumerate(exp_recs):
                if any(e in event_names for e in r["events"]):
                    offsets, scores = [], []
                    for off in range(-WINDOW - 1, WINDOW):
                        ni = exp_idx + off
                        if 0 <= ni < len(exp_recs):
                            offsets.append(off + 1)
                            scores.append(exp_recs[ni][1]["consensus"])
                    trajs.append((sid, np.array(offsets), np.array(scores)))
        return trajs

    colors_g = {"surp": "#e74c3c", "irr": "#3498db"}
    fills_g = {"surp": (0.91, 0.30, 0.24, 0.12), "irr": (0.20, 0.60, 0.86, 0.12)}
    labels_g = {"surp": "Group A", "irr": "Group B"}
    for trajs, key in [(_get_trajs(surprise_events), "surp"), (_get_trajs(irritation_events), "irr")]:
        if not trajs:
            continue
        for sid, offsets, scores in trajs:
            ax_a.plot(offsets, scores, color=colors_g[key], alpha=0.12, linewidth=0.4, zorder=1)
        all_offsets = sorted(set(off for _, offs, _ in trajs for off in offs))
        means = np.array([np.mean([sc[of == off][0] for _, of, sc in trajs if off in of]) for off in all_offsets])
        sems = np.array([np.std([sc[of == off][0] for _, of, sc in trajs if off in of]) / max(1, np.sqrt(sum(1 for _, of, _ in trajs if off in of))) for off in all_offsets])
        ax_a.fill_between(all_offsets, means - sems, means + sems, color=fills_g[key], zorder=2)
        ax_a.plot(all_offsets, means, color=colors_g[key], linewidth=1.4, marker="o", markersize=4,
                  zorder=3, label=f"{labels_g[key]} (n={len(trajs)})")
    ax_a.axvline(0, color="black", linestyle="--", linewidth=0.8, zorder=0)
    ax_a.set_xlabel("Segment Offset from Event"); ax_a.set_ylabel("RDI")
    ax_a.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax_a.text(0.02, 1.05, "(c)", transform=ax_a.transAxes, fontsize=10, fontweight="bold", va="bottom")

    # ── Panel (d): Cumulative detection ──────────────────────────────
    sr = det_eval["strategy_results"]
    y_true = det_eval["y_true"]
    n_pos, n_total = det_eval["n_pos"], det_eval["n_total"]
    ranked = sorted(sr.items(), key=lambda kv: kv[1]["roc_auc"], reverse=True)
    top_info = ranked[0][1]
    ro = np.argsort(top_info["scores"])[::-1]
    ch = np.cumsum(y_true[ro])
    dr = ch / n_pos
    hr = ch / np.arange(1, n_total + 1)
    pct = np.arange(1, n_total + 1) / n_total * 100

    ax_b.plot([0, 100], [0, 1], color="gray", linestyle="--", linewidth=0.6, label="Random")
    ax_b.plot(pct, dr, color="#27ae60", linewidth=1.5, label="Detection Rate")
    ax_b.plot(pct, hr, color="#8e44ad", linewidth=1.2, linestyle=":", label="Precision@K")
    pct_npos = n_pos / n_total * 100
    det_ceil = math.ceil(float(dr[n_pos - 1]) * 100) / 100
    ax_b.plot(pct_npos, dr[n_pos - 1], "D", color="red", markersize=6, zorder=5,
              label=f"Top-{n_pos} ({pct_npos:.0f}%): Det={det_ceil:.2f}")
    ax_b.set_xlabel("% of Segments Inspected"); ax_b.set_ylabel("Rate")
    ax_b.set_xlim(-2, 102); ax_b.set_ylim(-0.02, 1.05)
    ax_b.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax_b.text(0.02, 1.05, "(d)", transform=ax_b.transAxes, fontsize=10, fontweight="bold", va="bottom")

    fig.tight_layout(h_pad=1.2, w_pad=1.8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.replace(".png", f".{ext}"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    records, aff_set = _load_merged_records()
    optimal = _compute_optimal_per_event(records, aff_set)
    det = _detection_eval(records, aff_set, label_offsets=(0,))
    _plot(records, aff_set, optimal, det,
          os.path.join(OUT_DIR, "fig_combined_4panel.png"))
