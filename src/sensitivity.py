"""
Comprehensive Sensitivity Analysis module for the MsCO2limit project.
Evaluates response surfaces across single parameter 1D sweeps (Alfa, Beta, H1, H2)
and 2D interaction contours for N1 (particle loss) and N2 (carrier capture)
centered around the CFD experimental reference design.
"""

from pathlib import Path
import time
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.path import sensitivity_plots_dir
from src.constants import (
    BASE_FEATURES,
    DEFAULT_PARAM_RANGES,
    DEFAULT_REFERENCE_PARAMS,
    NOMINAL_BASELINE,
    PARAM_BOUNDS,
    PARAM_RANGES,
    PARAM_UNITS,
    REFERENCE_PARAMS,
)
from src.features import create_features
from src.models import load_model, predict, train_final_model


def calculate_reference_predictions(model, ref_params: Dict[str, float]) -> Dict[str, float]:
    """
    Computes surrogate predictions at the reference baseline point.
    """
    df_ref = pd.DataFrame([ref_params])
    X_feat = create_features(df_ref)
    n1, n2, delta = predict(model, X_feat)
    return {
        "N1_ref": float(n1[0]),
        "N2_ref": float(n2[0]),
        "Delta_ref": float(delta[0]),
    }


def analyze_parameter_sensitivity(
    model,
    param_name: str,
    ref_params: Dict[str, float],
    param_range: Tuple[float, float],
    n_points: int = 100,
) -> pd.DataFrame:
    """
    Executes 1D parameter sweep while keeping all other parameters fixed at reference values.
    """
    values = np.linspace(param_range[0], param_range[1], n_points)
    rows = []
    for val in values:
        item = ref_params.copy()
        item[param_name] = float(val)
        rows.append(item)

    df_base = pd.DataFrame(rows)
    X_feat = create_features(df_base)
    n1, n2, delta = predict(model, X_feat)

    return pd.DataFrame({
        param_name: values,
        "N1_pred": n1,
        "N2_pred": n2,
        "Delta": delta,
    })


def plot_sensitivity_analysis(
    results_dict: Dict[str, pd.DataFrame],
    ref_dict: Dict[str, float],
    save_path: Optional[Path] = None,
) -> None:
    """
    Generates full 3x4 grid: N1, N2, and Delta vs. each geometric parameter.
    """
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharey="row")
    params = list(results_dict.keys())
    colors = ["#0072B2", "#D55E00", "#009E73", "#E69F00"]

    for col_idx, param in enumerate(params):
        df_p = results_dict[param]
        unit = PARAM_UNITS.get(param, "")

        # Row 1: N1
        ax1 = axes[0, col_idx]
        ax1.plot(df_p[param], df_p["N1_pred"], color=colors[col_idx], linewidth=2.5, label="$N_1$")
        ax1.axhline(ref_dict["N1_ref"], color="gray", linestyle="--", alpha=0.7, label="CFD Ref")
        ax1.axvline(ref_dict[param], color="black", linestyle=":", alpha=0.5)
        ax1.set_title(f"$N_1$ vs {param}")
        ax1.set_xlabel(f"{param} [{unit}]")
        ax1.set_ylabel("$N_1$ pred [particles/s]")
        ax1.grid(True, linestyle=":", alpha=0.6)
        if col_idx == 0:
            ax1.legend()

        # Row 2: N2
        ax2 = axes[1, col_idx]
        ax2.plot(df_p[param], df_p["N2_pred"], color=colors[col_idx], linewidth=2.5, label="$N_2$")
        ax2.axhline(ref_dict["N2_ref"], color="gray", linestyle="--", alpha=0.7, label="CFD Ref")
        ax2.axvline(ref_dict[param], color="black", linestyle=":", alpha=0.5)
        ax2.set_title(f"$N_2$ vs {param}")
        ax2.set_xlabel(f"{param} [{unit}]")
        ax2.set_ylabel("$N_2$ pred [particles/s]")
        ax2.grid(True, linestyle=":", alpha=0.6)
        if col_idx == 0:
            ax2.legend()

        # Row 3: Delta
        ax3 = axes[2, col_idx]
        ax3.plot(df_p[param], df_p["Delta"], color=colors[col_idx], linewidth=2.5, label="$\\Delta$")
        ax3.axhline(ref_dict["Delta_ref"], color="gray", linestyle="--", alpha=0.7, label="CFD Ref")
        ax3.axvline(ref_dict[param], color="black", linestyle=":", alpha=0.5)
        ax3.set_title(f"$\\Delta$ vs {param}")
        ax3.set_xlabel(f"{param} [{unit}]")
        ax3.set_ylabel("$\\Delta$ pred [particles/s]")
        ax3.grid(True, linestyle=":", alpha=0.6)
        if col_idx == 0:
            ax3.legend()

    plt.suptitle("Global Sensitivity Sweeps of Inertial Separator Geometric Parameters", fontsize=16)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_combined_view(
    results_dict: Dict[str, pd.DataFrame],
    ref_dict: Dict[str, float],
    save_path: Optional[Path] = None,
) -> None:
    """
    Generates dual-Y sensitivity plots of N1 (loss) and N2 (capture) across each parameter.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (param, df_p) in enumerate(results_dict.items()):
        ax = axes[i]
        unit = PARAM_UNITS.get(param, "")

        color_n1 = "#0072B2"
        ax.set_xlabel(f"{param} [{unit}]", fontweight="bold")
        ax.set_ylabel("Particle Loss $N_1$ [particles/s]", color=color_n1)
        ax.plot(df_p[param], df_p["N1_pred"], color=color_n1, linewidth=2.5, label="$N_1$ (Loss)")
        ax.tick_params(axis="y", labelcolor=color_n1)

        ax2 = ax.twinx()
        color_n2 = "#D55E00"
        ax2.set_ylabel("Captured Particles $N_2$ [particles/s]", color=color_n2)
        ax2.plot(df_p[param], df_p["N2_pred"], color=color_n2, linewidth=2.5, linestyle="-.", label="$N_2$ (Capture)")
        ax2.tick_params(axis="y", labelcolor=color_n2)

        ax.set_title(f"Influence of Parameter {param} on $N_1$ and $N_2$")
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Dual-Response Sensitivity Sweeps ($N_1$ Loss vs. $N_2$ Capture)", fontsize=16)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close()


def plot_heatmap_interactions(
    model,
    ref_params: Optional[Dict[str, float]] = None,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    save_path: Optional[Path] = None,
    grid_size: int = 40,
) -> None:
    """
    Generates 2D interaction contour maps for key parameter pairs:
    1. Alfa vs. Beta
    2. H1 vs. H2
    """
    if ref_params is None:
        ref_params = DEFAULT_REFERENCE_PARAMS.copy()
    if param_ranges is None:
        param_ranges = DEFAULT_PARAM_RANGES.copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    pairs = [
        ("Alfa", "Beta", param_ranges["Alfa"], param_ranges["Beta"]),
        ("H1", "H2", param_ranges["H1"], param_ranges["H2"]),
    ]

    for pair_idx, (p1, p2, r1, r2) in enumerate(pairs):
        v1 = np.linspace(r1[0], r1[1], grid_size)
        v2 = np.linspace(r2[0], r2[1], grid_size)
        G1, G2 = np.meshgrid(v1, v2)

        rows = []
        for val1, val2 in zip(G1.ravel(), G2.ravel()):
            item = ref_params.copy()
            item[p1] = float(val1)
            item[p2] = float(val2)
            rows.append(item)

        df_grid = pd.DataFrame(rows)
        X_feat = create_features(df_grid)
        n1, n2, _ = predict(model, X_feat)

        N1_grid = n1.reshape(grid_size, grid_size)
        N2_grid = n2.reshape(grid_size, grid_size)

        ref_lbl = (
            f"CFD Ref ({ref_params[p1]:.3f}, {ref_params[p2]:.3f})"
            if "H" in p2
            else f"CFD Ref ({ref_params[p1]:.1f}, {ref_params[p2]:.1f})"
        )

        # N1 Contour Map
        ax_n1 = axes[pair_idx, 0]
        c1 = ax_n1.contourf(G1, G2, N1_grid, levels=20, cmap="viridis_r")
        fig.colorbar(c1, ax=ax_n1)
        ax_n1.scatter(
            ref_params[p1],
            ref_params[p2],
            color="red",
            marker="*",
            s=220,
            edgecolors="white",
            linewidth=1.5,
            label=ref_lbl,
            zorder=5,
        )
        ax_n1.set_title(f"Particle Loss $N_1$ (Minimization) for {p1} $\\times$ {p2}")
        ax_n1.set_xlabel(f"{p1} [{PARAM_UNITS.get(p1, '')}]")
        ax_n1.set_ylabel(f"{p2} [{PARAM_UNITS.get(p2, '')}]")
        ax_n1.legend(loc="upper right")

        # N2 Contour Map
        ax_n2 = axes[pair_idx, 1]
        c2 = ax_n2.contourf(G1, G2, N2_grid, levels=20, cmap="magma")
        fig.colorbar(c2, ax=ax_n2)
        ax_n2.scatter(
            ref_params[p1],
            ref_params[p2],
            color="cyan",
            marker="*",
            s=220,
            edgecolors="black",
            linewidth=1.5,
            label=ref_lbl,
            zorder=5,
        )
        ax_n2.set_title(f"Captured Particles $N_2$ (Maximization) for {p1} $\\times$ {p2}")
        ax_n2.set_xlabel(f"{p1} [{PARAM_UNITS.get(p1, '')}]")
        ax_n2.set_ylabel(f"{p2} [{PARAM_UNITS.get(p2, '')}]")
        ax_n2.legend(loc="upper right")

    plt.suptitle("2D Parameter Interaction Contours around CFD Reference Design", fontsize=16)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.close()


def run_sensitivity_analysis(
    output_dir: Optional[Path] = None,
    ref_params: Optional[Dict[str, float]] = None,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    n_points: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    Main sensitivity analysis execution function: generates 1D response profiles, 2D heatmaps, and exports CSV.
    """
    if ref_params is None:
        ref_params = DEFAULT_REFERENCE_PARAMS.copy()
    if param_ranges is None:
        param_ranges = DEFAULT_PARAM_RANGES.copy()

    print("[INFO] Loading surrogate model for sensitivity analysis...")
    try:
        model = load_model()
    except FileNotFoundError:
        print("[INFO] Model not found. Training surrogate model on available dataset...")
        model, _ = train_final_model()

    print(f"[PARAM] Geometric sensitivity parameter ranges: {param_ranges}")
    print("[INFO] Evaluating baseline predictions at CFD reference point...")
    ref_preds = calculate_reference_predictions(model, ref_params)
    ref_full = {**ref_params, **ref_preds}
    print(f"[PARAM] Reference baseline: {ref_params}")
    print(f"[PARAM] Predictions at reference: N1={ref_preds['N1_ref']:.2f}, N2={ref_preds['N2_ref']:.2f}, Delta={ref_preds['Delta_ref']:.2f}")

    print("[INFO] Computing 1D sensitivity sweeps across all base parameters...")
    results = {}
    for param in BASE_FEATURES:
        results[param] = analyze_parameter_sensitivity(
            model, param, ref_params, param_ranges[param], n_points=n_points
        )

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    if output_dir is None:
        output_dir = sensitivity_plots_dir / f"sensitivity_analysis_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Generating sensitivity figures...")
    plot_sensitivity_analysis(results, ref_full, save_path=output_dir / "sensitivity_full.png")
    plot_combined_view(results, ref_full, save_path=output_dir / "sensitivity_combined.png")
    plot_heatmap_interactions(
        model,
        ref_params=ref_params,
        param_ranges=param_ranges,
        save_path=output_dir / "sensitivity_heatmaps.png",
    )

    print("[INFO] Exporting sensitivity sweep data to CSV...")
    for param, df_p in results.items():
        csv_file = output_dir / f"sensitivity_{param}.csv"
        df_p.to_csv(csv_file, index=False)

    print(f"[OK] Sensitivity analysis successfully completed. Results saved to: {output_dir}")
    return results
