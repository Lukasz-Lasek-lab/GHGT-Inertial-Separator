# Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems

[![Conference: GHGT](https://img.shields.io/badge/Conference-GHGT-brightgreen.svg)](https://ghgt.info)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Official source code repository accompanying the research presentation at the **[GHGT (Greenhouse Gas Control Technologies)](https://ghgt.info)** conference and research publication:  
**"Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems"**.

<p align="center">
  <img src="assets/qr_code_github.png" alt="GitHub Repository QR Code" width="160"/>
  <br>
  <em>Scan QR code or visit <a href="https://github.com/Lukasz-Lasek-Lab/MsCO2limit">github.com/Lukasz-Lasek-Lab/MsCO2limit</a></em>
</p>

---

## Overview

Inertial particle separators play a critical role in **Chemical Looping Combustion (CLC)** and industrial carbon capture systems by separating solid oxygen carriers from gas streams under harsh hydrodynamic conditions. Traditional Computational Fluid Dynamics (CFD) optimizations require substantial computational time per geometric variation.

This framework introduces a high-performance **hybrid AI-driven metamodeling and evolutionary optimization methodology**:
1. **Physics-informed Feature Engineering:** Generates polynomial, interaction, and trigonometric aerodynamic descriptors (expanding geometric inputs into engineered interaction spaces).
2. **Surrogate ML Regression:** Employs multi-target `HistGradientBoostingRegressor` with physical consistency constraints ($N_1$: particle loss, $\Delta = N_2 - N_1$: net collection advantage, $N_2$: carrier capture).
3. **Evolutionary Multi-Objective Optimization (NSGA-II):** Discovers Pareto-optimal design configurations balancing zero-loss emissions against maximum carrier capture, accelerated via vectorized surrogate evaluation.
4. **Engineering Clustering & Validation:** Employs K-Means non-dominated front clustering and comparative case studies against CFD baseline geometries.
5. **Publication-Ready Visualization:** Automated generation of Q1 journal standard figures (Okabe-Ito colorblind-safe palettes, vector PDF/SVG, standalone panels and composite grids).

---

## Geometric Optimization Problem

| Parameter | Symbol | Domain Range | Physical Role |
| :--- | :---: | :---: | :--- |
| **Inlet Angle** | $\alpha$ | $42.75^\circ - 60.00^\circ$ | Gas-solid inlet trajectory and tangential momentum |
| **Guide Angle** | $\beta$ | $42.75^\circ - 60.00^\circ$ | Deflection and vortex stabilization |
| **Lower Height** | $H_1$ | $0.0080 - 0.0609\text{ m}$ | Separation zone depth |
| **Upper Height** | $H_2$ | $0.0080 - 0.0609\text{ m}$ | Vortex finder / clean gas outlet position |

**Optimization Objectives:**
- $\min N_1$ (minimize escaping particle emissions / losses)
- $\max N_2$ (maximize captured carrier particles)
- $\max \Delta = N_2 - N_1$ (maximize net separation advantage)

---

## Repository Structure

```text
MsCO2limit/
├── .gitignore               # Strict exclusion of proprietary CFD data and model binaries
├── .env.example             # Optional environment variable template
├── LICENSE                  # MIT License (Łukasz Lasek, UJD)
├── README.md                # Project documentation and reproduction guidelines
├── requirements.txt         # Minimal, stable Python dependencies
├── assets/                  # Conference poster assets and vector QR codes
│   ├── qr_code_github.svg   # Vector QR code for conference poster printing
│   ├── qr_code_github.png   # High-resolution raster QR code (300 DPI)
│   └── qr_code_github_transparent.png
├── config/
│   ├── __init__.py
│   ├── best_params.json     # HistGradientBoosting hyperparameter configuration
│   ├── path.py              # Dynamic, cross-platform path resolution
│   └── selected_features.json # Selected engineered feature set
├── data/
│   └── demo/                # Synthetic benchmark dataset for instant reproduction
│       └── inertial_separator_demo.csv
└── src/
    ├── __init__.py          # Package initialization
    ├── __main__.py          # Unified Command-Line Interface (CLI)
    ├── features.py          # Physics-based feature extraction and scaling
    ├── feature_selection.py # Multi-objective feature selection routines
    ├── models.py            # HistGradientBoosting multi-target surrogate modeling
    ├── genetic.py           # NSGA-II multi-objective genetic optimization (vectorized)
    ├── pareto_selection.py  # Non-dominated sorting, K-Means clustering, and decision support
    ├── pipeline.py          # Modular training, CV, and evaluation orchestrator
    ├── sensitivity.py       # 1D response sweeps and 2D interaction contour generation
    └── visualization/
        ├── __init__.py
        ├── figures.py       # Publication figure generators (Fig 2 - Fig 6)
        └── publication_plots.py # Q1 journal publication plotting standards (Okabe-Ito)
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/Lukasz-Lasek-Lab/MsCO2limit.git
cd MsCO2limit

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Replication & Demonstration Mode (`--demo`)

For conference attendees, reviewers, and researchers wishing to immediately replicate the computational pipeline and generate Q1 vector figures without needing access to proprietary CFD meshes, a **synthetic benchmark dataset** is provided in `data/demo/inertial_separator_demo.csv`.

> **Automatic Fallback:** The framework automatically detects if proprietary CFD data is absent and seamlessly falls back to the benchmark demonstration dataset. Adding the `--demo` flag explicitly ensures demonstration mode across all CLI commands.

```bash
# 1. Replicate all publication figures in seconds:
python -m src --demo --all-figures

# 2. Run 5-fold cross-validation in demo mode:
python -m src --cv --demo

# 3. Train the surrogate model on demo data:
python -m src --train --demo

# 4. Generate a specific figure (e.g., Figure 2 Diagnostics):
python -m src --figure 2 --demo

# 5. Run full end-to-end pipeline (Train -> CV -> NSGA-II -> All Figures):
python -m src --demo --all
```

---

## Command-Line Usage (CLI)

The framework is operated via the unified `src` module:

```bash
# View all available CLI options
python -m src --help
```

### Running Computational Pipelines
```bash
# Run 5-fold cross-validation evaluation
python -m src --cv

# Train production surrogate model
python -m src --train

# Run multi-objective NSGA-II optimization
python -m src --optimize

# Execute full end-to-end pipeline
python -m src --all
```

### Generating Publication Figures (Q1 Standard)
Figures are automatically exported in vector (`.pdf`, `.svg`) and raster (`.png`, 300+ DPI) formats to `figures/`:

```bash
# Generate Figure 2: Model Diagnostics & Residual Analysis (5-Fold CV OOF)
python -m src --figure 2

# Generate Figure 3: Geometric & Aerodynamic Feature Importance (XAI)
python -m src --figure 3

# Generate Figure 4: Sensitivity Sweeps 1D and Interaction Contours 2D
python -m src --figure 4

# Generate Figure 5: NSGA-II Multi-Objective Convergence & Pareto Front
python -m src --figure 5

# Generate Figure 6: Engineering Validation Case Study (Baseline CFD vs. Optima)
python -m src --figure 6

# Generate all publication figures simultaneously
python -m src --all-figures
```

---

## Data and Model Availability Statement

> **Notice on Proprietary Research Data:**  
> The raw Computational Fluid Dynamics (CFD) simulation meshes and industrial particle tracking datasets used in this study are proprietary and part of ongoing industrial carbon capture research.  
> They are available from the corresponding authors upon reasonable scientific request for academic verification and peer review (contact: `l.lasek@ujd.edu.pl`).  
> This repository provides the **complete, open-source computational methodology, feature extraction pipeline, surrogate architecture, evolutionary optimization algorithms, publication figure reproduction code, and synthetic benchmark dataset**.

---

## Citation

If you use this framework, parts of its methodology, or the poster presentation in your research, please cite:

```bibtex
@article{MsCO2limit2026,
  title={Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems},
  author={Lasek, {\L}ukasz and Contributors},
  journal={18th International Conference on Greenhouse Gas Control Technologies (GHGT)},
  year={2026},
  url={https://github.com/Lukasz-Lasek-Lab/MsCO2limit}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
