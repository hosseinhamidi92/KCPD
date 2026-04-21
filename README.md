# KCPD — Kernel-based Change Point Detection for Physiological Signals

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Unsupervised Kernel Segmentation for Characterization of Physiological Signals Regime Disruptions in Simulated Driving**
>
> Hossein Hamidi Shishavan, Ming Nie, Sagar Kamarthi
>
> *IEEE Transactions on Affective Computing*, 2025

---

## Overview

KCPD is an unsupervised framework that detects physiological regime disruptions
from multivariate biosignals recorded during naturalistic driving.  It segments
continuous physiological time series into regimes using a Gaussian kernel in a
Reproducing Kernel Hilbert Space (RKHS), then quantifies how each regime
diverges from baseline via a consensus **Regime Divergence Index (RDI)** that
combines three complementary distance measures.

### Key Contributions

- **KernelCPD segmentation**: Gaussian RBF kernel change point detection via
  penalized dynamic programming — no labelled data required.
- **Regime Divergence Index (RDI)**: Consensus scoring from DTW distance
  (shape), RuLSIF divergence (distribution), and Cohen's d (effect size).
- **Multi-modal SQA gating**: Per-modality signal quality assessment prevents
  artifact-driven false detections.
- **Statistical validation**: Permutation tests (n = 10,000) confirm that
  event-containing segments have significantly higher RDI (p < 0.001).
- **Cross-dataset generalization**: Validated on both the driving dataset and
  the public WESAD dataset.

### Pipeline Overview (Algorithm 1)

```
Raw Signals (ECG, PPG, Resp, EDA, Temp)
    │
    ▼
Signal Quality Assessment ──► Gate low-quality segments
    │
    ▼
Sliding Window Feature Extraction (W=30s, Δ=1s) ──► 19 features × T windows
    │
    ▼
KernelCPD Segmentation (Gaussian RBF, β=1, m=10) ──► K regime boundaries
    │
    ▼
Pairwise Regime Comparison:
  ├── DTW distance        (shape similarity)
  ├── RuLSIF divergence   (distributional shift)
  └── Cohen's d           (mean shift magnitude)
    │
    ▼
Consensus RDI ──► Within-subject z-normalization ──► Outlier regime detection
    │
    ▼
Permutation Test (n=10,000) ──► Statistical validation (p-value, effect size)
```

---

## Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── kcpd/                              # Core algorithm package
│   ├── __init__.py
│   ├── pipeline.py                    # End-to-end pipeline (Algorithm 1)
│   │
│   ├── preprocessing/                 # Section III.A–C
│   │   ├── data_loader.py             # Biopac data loading & normalization
│   │   ├── signal_quality.py          # Multi-criterion SQA (EDA, PPG, Resp)
│   │   └── feature_extraction.py      # 19-feature extraction (Table II)
│   │
│   ├── detection/                     # Section III.D–E
│   │   ├── kernel_cpd.py              # KernelCPD segmentation
│   │   ├── dtw.py                     # Dynamic Time Warping distance
│   │   ├── cohens_d.py                # Multi-feature Cohen's d
│   │   ├── rulsif.py                  # RuLSIF density-ratio divergence
│   │   └── rdi.py                     # Regime Divergence Index (consensus)
│   │
│   └── validation/                    # Section III.F
│       └── permutation_test.py        # Permutation test & z-normalization
│
├── figures/                           # Figure generation scripts
│   ├── fig_pipeline_schematic.py      # Pipeline overview (Algorithm 1)
│   ├── fig_sqi.py                     # Fig. 4: SQI overlays
│   ├── fig_sqa_gate.py                # Fig. 5: SQA gating effect
│   ├── fig_questionnaire.py           # Fig. 8: Arousal-valence ratings
│   ├── fig_session_overview.py        # Fig. 9: Full-session heatmap + RDI
│   ├── fig_permutation.py             # Fig. 10: Permutation test
│   ├── fig_combined_4panel.py         # Fig. 11: ROC, AUC, trajectories
│   ├── fig_modality_ablation.py       # Fig. 12: Solo modality ablation
│   ├── fig_feature_response.py        # Fig. 13: Feature ΔZ response
│   └── fig_rdi_heatmap.py             # Fig. 14: WESAD vs driving RDI
│
├── data/                              # Data directory (not included)
│   └── README.md                      # Download instructions & DOI
│
└── docs/                              # Paper PDFs
    ├── paper.pdf                      # Main manuscript
    └── supplementary.pdf              # Supplementary material
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/hosseinhamidi92/KCPD.git
cd KCPD

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package       | Version | Purpose                              |
|---------------|---------|--------------------------------------|
| numpy         | ≥ 1.24  | Array operations                     |
| scipy         | ≥ 1.10  | Signal processing, statistics        |
| pandas        | ≥ 2.0   | Data manipulation                    |
| scikit-learn  | ≥ 1.3   | Distance metrics, evaluation         |
| neurokit2     | ≥ 0.2.4 | ECG/PPG/Resp/EDA processing          |
| heartpy       | ≥ 1.2.7 | Heart rate variability               |
| flirt         | ≥ 0.0.4 | Windowed HRV feature extraction      |
| ruptures      | ≥ 1.1.8 | Kernel change point detection        |
| matplotlib    | ≥ 3.7   | Publication-quality figures          |
| scienceplots  | ≥ 2.1   | IEEE-style formatting                |
| plotly        | ≥ 5.15  | Interactive dashboards               |

---

## Quick Start

```python
import numpy as np
from kcpd.pipeline import run_pipeline

# Load your physiological signals (example with synthetic data)
fs_ecg, fs_ppg, fs_resp, fs_eda, fs_temp = 200, 200, 100, 4, 4
duration = 600  # 10 minutes

ecg = np.random.randn(duration * fs_ecg)
ppg = np.random.randn(duration * fs_ppg)
resp = np.random.randn(duration * fs_resp)
eda = np.random.randn(duration * fs_eda) + 5
temp = np.random.randn(duration * fs_temp) * 0.1 + 33

# Run the full pipeline
results = run_pipeline(
    ecg, ppg, resp, eda, temp,
    penalty=1.0,         # KernelCPD penalty β
    min_size=10,         # Minimum segment length m
    event_times=[120, 300, 450],
    event_names=["Crash", "Barrel", "Brake"],
)

# Access results
print(f"Change points: {results['boundaries']}")
print(f"Number of segments: {len(results['segments'])}")
print(f"Consensus RDI: {results['consensus_rdi']}")
print(f"Permutation p-value: {results['permutation']['p_value']:.4f}")
print(f"Effect size (r_rb): {results['permutation']['effect_size']:.3f}")
```

---

## Reproducing Paper Figures

Each script in `figures/` generates one figure from the paper.  Scripts
load pre-computed cache files from `output/` (generated by the full
analysis pipeline).

```bash
# Generate all figures
for f in figures/fig_*.py; do
    MPLBACKEND=Agg python "$f"
done
```

| Script | Paper Figure | Content |
|--------|-------------|---------|
| `fig_sqi.py` | Fig. 4 | Signal Quality Index overlays (4 modalities) |
| `fig_sqa_gate.py` | Fig. 5 | SQA gating effect on feature heatmap |
| `fig_questionnaire.py` | Fig. 8 | Self-report arousal-valence ratings |
| `fig_session_overview.py` | Fig. 9 | Full-session feature heatmap + RDI strip |
| `fig_permutation.py` | Fig. 10 | Permutation test (4-panel) |
| `fig_combined_4panel.py` | Fig. 11 | ROC, AUC violins, trajectories, cumulative |
| `fig_modality_ablation.py` | Fig. 12 | Solo modality leave-one-out ablation |
| `fig_feature_response.py` | Fig. 13 | Feature ΔZ response profiles |
| `fig_rdi_heatmap.py` | Fig. 14 | WESAD vs. driving RDI comparison |
| `fig_pipeline_schematic.py` | — | Pipeline overview (Algorithm 1) |

---

## Dataset

The driving physiological dataset is publicly available:

> **Physiological Signals from Simulated Driving with Affective Events**
>
> DOI: [10.21227/x51t-jr44](https://dx.doi.org/10.21227/x51t-jr44)

- **15 participants**, ~30 min driving sessions
- **5 modalities**: ECG (200 Hz), PPG (200 Hz), Resp (100 Hz), EDA (4 Hz), Skin Temp (4 Hz)
- **Affective events**: Vehicle crash, barrel roll, sudden braking, fog onset, slow traffic

See [`data/README.md`](data/README.md) for download and setup instructions.

---

## Method Details

### KernelCPD (Section III.D)

Change points are detected by minimizing the penalized cost:

$$\arg\min_{t_1,\ldots,t_K} \sum_{k=1}^{K+1} C(y_{t_k:t_{k+1}}) + \beta \cdot K$$

where $C(\cdot)$ is the kernel cost in RKHS using the Gaussian RBF kernel
$k(x,x') = \exp(-\|x-x'\|^2 / 2\sigma^2)$, and $\sigma$ is set via the
median heuristic.

### RDI Scoring (Section III.E)

For each pair of segments $(S_i, S_j)$, three distances are computed:

| Metric | What it measures | Formula |
|--------|-----------------|---------|
| DTW | Shape similarity | $\text{DTW}(S_i, S_j) / \max(\|S_i\|, \|S_j\|)$ |
| RuLSIF | Distributional shift | Pearson divergence via density ratio |
| Cohen's d | Mean shift | $\sqrt{\frac{1}{N}\sum_j d_j^2}$ (RMS across features) |

The consensus RDI is the average of the three normalized scores.

### Permutation Test (Section III.F)

Statistical significance is assessed by permuting event labels 10,000
times and comparing the observed Δ (mean RDI of event segments − mean RDI
of non-event segments) against the null distribution.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{shishavan2025kcpd,
  title     = {Unsupervised Kernel Segmentation for Characterization of
               Physiological Signals Regime Disruptions in Simulated Driving},
  author    = {Shishavan, Hossein Hamidi and Nie, Ming and Kamarthi, Sagar},
  journal   = {IEEE Transactions on Affective Computing},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/TAFFC.2025.XXXXXXX}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

This work was conducted at the Toyota Research Institute of North America (TRINA),
Toyota Motor North America R&D.
