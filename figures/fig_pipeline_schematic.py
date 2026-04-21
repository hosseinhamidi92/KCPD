#!/usr/bin/env python3
"""
Pipeline Schematic — Synthetic illustration of the segment-outlier method.

Panels:
  (a) Multivariate time series segmented by KernelCPD
  (b) Three pairwise distance matrices (DTW, RuLSIF, Cohen's d)
  (c) Metric intuition sketches (shape, distribution, mean shift)
  (d) Per-method normalised scores
  (e) Consensus outlier score with threshold

Produces: output/figures/fig_pipeline_schematic.{png,pdf}
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee", "no-latex"])
except Exception:
    pass

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 600, "savefig.dpi": 600,
    "text.usetex": False, "font.family": "serif",
})

OUT_DIR = "output/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Synthetic data ──────────────────────────────────────────────────────
np.random.seed(42)
T = 200
t = np.arange(T)
feat_names = [r"$x_t^{(1)}$", r"$x_t^{(2)}$", r"$x_t^{(3)}$"]
n_feat = len(feat_names)
cps = [0, 35, 72, 110, 148, 180, T]
n_seg = len(cps) - 1
seg_labels = [f"$S_{{{i+1}}}$" for i in range(n_seg)]

seg_levels = [
    [0.30, 0.20, 0.50],
    [0.30, 0.20, 0.50],
    [0.72, 0.78, 0.22],
    [0.31, 0.21, 0.51],
    [0.30, 0.20, 0.50],
    [0.55, 0.58, 0.35],
]
features = np.zeros((T, n_feat))
for si in range(n_seg):
    s, e = cps[si], cps[si + 1]
    seg_len = e - s
    for fi in range(n_feat):
        base = seg_levels[si][fi]
        if si == 2:
            wave = 0.10 * np.abs(np.sin(np.linspace(0, 7 * np.pi, seg_len)))
            features[s:e, fi] = base + wave + 0.015 * np.random.randn(seg_len)
        elif si == 1:
            mode = (np.random.rand(seg_len) > 0.5).astype(float)
            features[s:e, fi] = base + mode * 0.12 - 0.06 + 0.02 * np.random.randn(seg_len)
        elif si == 4:
            ramp = np.linspace(-0.08, 0.08, seg_len)
            features[s:e, fi] = base + ramp + 0.015 * np.random.randn(seg_len)
        elif si == 5:
            features[s:e, fi] = base + 0.025 * np.random.randn(seg_len)
        else:
            features[s:e, fi] = base + 0.025 * np.random.randn(seg_len)

C_NORM, C_OUT = "#d5e8d4", "#f8cecc"
E_NORM, E_OUT = "#82b366", "#b85450"
C_FEAT = ["#1f77b4", "#9467bd", "#2ca02c"]
C_DTW, C_RUL, C_COH, C_CONS = "#1f77b4", "#ff7f0e", "#2ca02c", "#c0392b"
is_outlier = [False, False, True, False, False, True]

def _sc(i): return C_OUT if is_outlier[i] else C_NORM
def _ec(i): return E_OUT if is_outlier[i] else E_NORM


# ── Distance matrices ───────────────────────────────────────────────────
def _make_dm(seed, style):
    rng = np.random.RandomState(seed)
    dm = np.zeros((n_seg, n_seg))
    for i in range(n_seg):
        for j in range(i + 1, n_seg):
            if style == "shape":
                i_shape, j_shape = (i == 2), (j == 2)
                i_ramp, j_ramp = (i == 4), (j == 4)
                if i_shape and j_shape:
                    d = 0.10 + 0.05 * rng.rand()
                elif i_shape or j_shape:
                    d = 0.82 + 0.12 * rng.rand()
                elif i_ramp or j_ramp:
                    d = (0.05 + 0.03 * rng.rand()) if (i_ramp and j_ramp) else (0.40 + 0.12 * rng.rand())
                else:
                    d = 0.03 + 0.06 * rng.rand()
            elif style == "distribution":
                i_bim, j_bim = (i == 1), (j == 1)
                i_ev, j_ev = (i == 2), (j == 2)
                i_sh, j_sh = (i == 5), (j == 5)
                i_ramp, j_ramp = (i == 4), (j == 4)
                if i_bim or j_bim:
                    d = (0.08 + 0.04 * rng.rand()) if (i_bim and j_bim) else (0.70 + 0.10 * rng.rand()) if (i_ev or j_ev) else (0.65 + 0.12 * rng.rand())
                elif i_ev or j_ev:
                    d = (0.05 + 0.03 * rng.rand()) if (i_ev and j_ev) else (0.55 + 0.15 * rng.rand())
                elif i_sh or j_sh:
                    d = (0.05 + 0.03 * rng.rand()) if (i_sh and j_sh) else (0.35 + 0.10 * rng.rand())
                elif i_ramp or j_ramp:
                    d = 0.25 + 0.10 * rng.rand()
                else:
                    d = 0.08 + 0.08 * rng.rand()
            else:
                i_big, j_big = (i == 2), (j == 2)
                i_mod, j_mod = (i == 5), (j == 5)
                if i_big or j_big:
                    d = (0.03 + 0.02 * rng.rand()) if (i_big and j_big) else (0.50 + 0.10 * rng.rand()) if (i_mod or j_mod) else (0.92 + 0.06 * rng.rand())
                elif i_mod or j_mod:
                    d = (0.03 + 0.02 * rng.rand()) if (i_mod and j_mod) else (0.52 + 0.10 * rng.rand())
                else:
                    d = 0.02 + 0.04 * rng.rand()
            dm[i, j] = dm[j, i] = d
    return dm

dms = {"DTW": _make_dm(10, "shape"), "RuLSIF": _make_dm(20, "distribution"),
       "Cohen's d": _make_dm(30, "effect_size")}
method_info = [("DTW", C_DTW, "shape"), ("RuLSIF", C_RUL, "distribution"),
               ("Cohen's d", C_COH, "effect size")]

per_method = {}
for name, dm in dms.items():
    avg = dm.sum(axis=1) / (n_seg - 1)
    per_method[name] = avg / avg.max()
consensus = np.mean(list(per_method.values()), axis=0)

# ── Figure layout ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.16, 4.8))
gs = fig.add_gridspec(
    nrows=4, ncols=4,
    height_ratios=[0.6, 0.75, 0.5, 0.5],
    width_ratios=[0.5, 0.5, 0.5, 1.8],
    hspace=0.70, wspace=0.40,
    left=0.07, right=0.97, top=0.97, bottom=0.06,
)

# ── (a) Time Series ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
for fi in range(n_feat):
    off = (n_feat - 1 - fi) * 0.6
    ax.plot(t, features[:, fi] + off, color=C_FEAT[fi], lw=0.55, alpha=0.9, zorder=3)
    ax.text(-6, seg_levels[0][fi] + off + 0.02, feat_names[fi],
            fontsize=8, color=C_FEAT[fi], ha="right", va="center", fontweight="bold")
for si in range(n_seg):
    s, e = cps[si], cps[si + 1]
    ax.axvspan(s, e, alpha=0.20 if is_outlier[si] else 0.12, color=_sc(si), zorder=1)
for cp in cps[1:-1]:
    ax.axvline(cp, color="#666", lw=0.7, ls="--", alpha=0.6, zorder=4)
for si in range(n_seg):
    mid = (cps[si] + cps[si + 1]) / 2
    ax.text(mid, -0.28, seg_labels[si], ha="center", va="top", fontsize=9,
            fontweight="bold" if is_outlier[si] else "normal",
            color=E_OUT if is_outlier[si] else "#333",
            bbox=dict(boxstyle="round,pad=0.2", fc=_sc(si), ec=_ec(si), lw=0.9, alpha=0.92))
ax.annotate("", xy=(cps[2], -0.15), xytext=(cps[3], -0.15),
            arrowprops=dict(arrowstyle="<->", color="#666", lw=0.7))
ax.text((cps[2]+cps[3])/2, -0.10, "segment $S_3$", ha="center", va="bottom",
        fontsize=7.5, color="#666", style="italic")
ax.annotate("KernelCPD\n(RBF, $\\lambda$=1.0)", xy=(cps[1], 1.55),
            xytext=(cps[1]+25, 1.78), fontsize=8, ha="center", va="bottom",
            color="#444",
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=0.8,
                            connectionstyle="arc3,rad=-0.2"),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fffde7", ec="#f9a825", lw=0.6))
ax.set_xlim(-3, T+3); ax.set_ylim(-0.42, n_feat*0.6+0.35)
ax.set_yticks([]); ax.set_xlabel("Time (samples)", fontsize=9, labelpad=2)
ax.set_title("(a)", fontsize=10, fontweight="bold", loc="left", pad=6)

# ── (b) Distance Matrices ──────────────────────────────────────────────
cmaps = ["Blues", "Oranges", "Greens"]
ax_b_row = []
for mi, (mname, mcol, _) in enumerate(method_info):
    axd = fig.add_subplot(gs[1, mi])
    ax_b_row.append(axd)
    dm = dms[mname]
    axd.imshow(dm, cmap=cmaps[mi], vmin=0, vmax=dm.max(),
               aspect="equal", interpolation="nearest")
    axd.set_xticks(range(n_seg)); axd.set_xticklabels(seg_labels, fontsize=6.5)
    axd.set_yticks(range(n_seg)); axd.set_yticklabels(seg_labels, fontsize=6.5)
    axd.set_title(f"$D^{{\\mathrm{{{mname}}}}}$",
                  fontsize=8, fontweight="bold", color=mcol, pad=3)
    for idx in [2, 5]:
        axd.add_patch(mpatches.Rectangle((-0.5, idx-0.5), n_seg, 1,
                      lw=1.0, ec=E_OUT, fc="none", zorder=5))
        axd.add_patch(mpatches.Rectangle((idx-0.5, -0.5), 1, n_seg,
                      lw=1.0, ec=E_OUT, fc="none", zorder=5))

pos_a = ax.get_position()
pos_b = ax_b_row[0].get_position()
fig.text(pos_a.x0, pos_b.y1 + 0.04, "(b)",
         fontsize=10, fontweight="bold", ha="left", va="bottom")

# ── (c) Metric Intuition ───────────────────────────────────────────────
from matplotlib.gridspec import GridSpecFromSubplotSpec
gs_e = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 3], hspace=0.55)
np.random.seed(77)
_t_mini = np.linspace(0, 2 * np.pi, 60)

ax_dtw = fig.add_subplot(gs_e[0])
_s_a = 0.5 * np.sin(_t_mini)
_s_b = 0.5 * np.sin(_t_mini * 1.5 + 0.4)
ax_dtw.plot(_t_mini, _s_a, color=C_DTW, lw=1.2, label="$S_i$")
ax_dtw.plot(_t_mini, _s_b, color=E_OUT, lw=1.2, ls="--", label="$S_j$")
for k in [8, 20, 35, 48]:
    k2 = min(len(_t_mini)-1, int(k * 1.15))
    ax_dtw.plot([_t_mini[k], _t_mini[k2]], [_s_a[k], _s_b[k2]],
                color="#aaa", lw=0.5, alpha=0.6, zorder=1)
ax_dtw.set_ylabel("DTW", fontsize=7, color=C_DTW, fontweight="bold", labelpad=1)
ax_dtw.set_xticks([]); ax_dtw.set_yticks([])
ax_dtw.text(0.98, 0.95, "shape\nalignment", transform=ax_dtw.transAxes,
            fontsize=5.5, ha="right", va="top", color="#555", style="italic")
ax_dtw.legend(fontsize=5, loc="upper left", framealpha=0.7, handlelength=1.0,
              borderpad=0.2, handletextpad=0.3)

ax_rul = fig.add_subplot(gs_e[1])
_d1 = np.random.normal(0.0, 0.3, 300)
_d2 = np.concatenate([np.random.normal(-0.15, 0.18, 150),
                      np.random.normal(0.25, 0.18, 150)])
_bins = np.linspace(-1.0, 1.0, 30)
ax_rul.hist(_d1, bins=_bins, density=True, alpha=0.55, color=C_RUL, label="$S_i$", lw=0)
ax_rul.hist(_d2, bins=_bins, density=True, alpha=0.45, color=E_OUT, label="$S_j$", lw=0)
ax_rul.set_ylabel("RuLSIF", fontsize=7, color=C_RUL, fontweight="bold", labelpad=1)
ax_rul.set_xticks([]); ax_rul.set_yticks([])
ax_rul.text(0.98, 0.95, "density\nratio", transform=ax_rul.transAxes,
            fontsize=5.5, ha="right", va="top", color="#555", style="italic")
ax_rul.legend(fontsize=5, loc="upper left", framealpha=0.7, handlelength=1.0,
              borderpad=0.2, handletextpad=0.3)

ax_coh = fig.add_subplot(gs_e[2])
_x_g = np.linspace(-1.5, 2.5, 200)
_mu_a, _sig, _mu_b = 0.0, 0.45, 1.0
_ga = np.exp(-0.5 * ((_x_g - _mu_a) / _sig)**2) / (_sig * np.sqrt(2 * np.pi))
_gb = np.exp(-0.5 * ((_x_g - _mu_b) / _sig)**2) / (_sig * np.sqrt(2 * np.pi))
ax_coh.fill_between(_x_g, _ga, alpha=0.45, color=C_COH, lw=0)
ax_coh.fill_between(_x_g, _gb, alpha=0.35, color=E_OUT, lw=0)
ax_coh.plot(_x_g, _ga, color=C_COH, lw=1.0, label="$S_i$")
ax_coh.plot(_x_g, _gb, color=E_OUT, lw=1.0, ls="--", label="$S_j$")
_y_arr = max(_ga.max(), _gb.max()) * 0.55
ax_coh.annotate("", xy=(_mu_b, _y_arr), xytext=(_mu_a, _y_arr),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.0))
ax_coh.text((_mu_a + _mu_b) / 2, _y_arr + 0.05, "$d$", ha="center", va="bottom",
            fontsize=7, fontweight="bold", color="#333")
ax_coh.set_ylabel("Cohen", fontsize=7, color=C_COH, fontweight="bold", labelpad=1)
ax_coh.set_xticks([]); ax_coh.set_yticks([])
ax_coh.text(0.98, 0.95, "mean\nshift", transform=ax_coh.transAxes,
            fontsize=5.5, ha="right", va="top", color="#555", style="italic")
ax_coh.legend(fontsize=5, loc="upper left", framealpha=0.7, handlelength=1.0,
              borderpad=0.2, handletextpad=0.3)
ax_dtw.set_title("(c)", fontsize=10, fontweight="bold", loc="left", pad=4)

# ── (d) Per-Method Bars ────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[2, :])
x = np.arange(n_seg); bw = 0.24
for mi, (mn, mc, _) in enumerate(method_info):
    ax_bar.bar(x + (mi-1)*bw, per_method[mn], bw, color=mc, alpha=0.70,
               edgecolor="white", lw=0.4,
               label=f"$\\hat{{d}}^{{\\mathrm{{{mn}}}}}$")
ax_bar.set_xticks(x); ax_bar.set_xticklabels(seg_labels, fontsize=8)
ax_bar.set_ylabel("Norm.\nScore", fontsize=8, labelpad=2)
ax_bar.set_ylim(0, 1.18)
ax_bar.legend(loc="upper left", fontsize=7.5, ncol=3, framealpha=0.9,
              handlelength=1.0, columnspacing=0.8)
for idx in [2, 5]:
    ax_bar.axvspan(idx-0.45, idx+0.45, alpha=0.10, color=C_OUT, zorder=0)
ax_bar.set_title("(d)", fontsize=10, fontweight="bold", loc="left", pad=4)

# ── (e) Consensus Score ────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[3, :])
bcolors = [C_CONS if is_outlier[i] else "#999" for i in range(n_seg)]
ax_c.bar(x, consensus, 0.52, color=bcolors, alpha=0.85, edgecolor="white", lw=0.5)
for i in range(n_seg):
    ax_c.text(x[i], consensus[i]+0.02, f"{consensus[i]:.2f}", ha="center",
              va="bottom", fontsize=8,
              fontweight="bold" if is_outlier[i] else "normal",
              color=C_CONS if is_outlier[i] else "#555")
thr = np.percentile(consensus, 75)
ax_c.axhline(thr, color="#555", lw=0.6, ls=":", alpha=0.7)
ax_c.text(n_seg-0.35, thr+0.015, "$P_{75}$", fontsize=7.5, color="#555",
          ha="right", va="bottom", style="italic")
ax_c.set_xticks(x); ax_c.set_xticklabels(seg_labels, fontsize=8)
ax_c.set_ylabel("Consensus\n$c_i$", fontsize=8, labelpad=2)
ax_c.set_ylim(0, 1.18)
ax_c.set_title("(e)", fontsize=10, fontweight="bold", loc="left", pad=4)
for idx in [2, 5]:
    ax_c.axvspan(idx-0.42, idx+0.42, alpha=0.12, color=C_OUT, zorder=0)
    ax_c.text(idx, consensus[idx] * 0.5, "event", ha="center", va="center",
              fontsize=8, color="white", fontweight="bold", zorder=12, rotation=90)

# Adjust vertical spacing
_cd_lift, _cd_squeeze = 0.07, 0.03
pos_c = ax_bar.get_position()
pos_d = ax_c.get_position()
ax_bar.set_position([pos_c.x0, pos_c.y0 + _cd_lift - _cd_squeeze, pos_c.width, pos_c.height])
ax_c.set_position([pos_d.x0, pos_d.y0 + _cd_lift + _cd_squeeze, pos_d.width, pos_d.height])

for ext in ("png", "pdf"):
    p = os.path.join(OUT_DIR, f"fig_pipeline_schematic.{ext}")
    fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
    print(f"Saved → {p}")
plt.close(fig)
