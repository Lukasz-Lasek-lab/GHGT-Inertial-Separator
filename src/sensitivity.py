"""
Comprehensive Sensitivity Analysis module for the MsCO2limit project.
Evaluates response surfaces across single parameter 1D sweeps (Alfa, Beta, H1, H2)
and 2D interaction contours for N1 (particle loss) and N2 (carrier capture)
centered around the CFD experimental reference design.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

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


def compute_2d_grid(
    model,
    param1: str,
    param2: str,
    range1: Tuple[float, float],
    range2: Tuple[float, float],
    ref_params: Dict[str, float],
    grid_size: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes 2D meshgrid and surrogate prediction surfaces for a pair of geometric parameters.
    Keeps all other parameters fixed at reference values.

    Returns:
        Tuple of (G1, G2, N1_grid, N2_grid)
    """
    v1 = np.linspace(range1[0], range1[1], grid_size)
    v2 = np.linspace(range2[0], range2[1], grid_size)
    G1, G2 = np.meshgrid(v1, v2)

    rows = []
    for val1, val2 in zip(G1.ravel(), G2.ravel()):
        item = ref_params.copy()
        item[param1] = float(val1)
        item[param2] = float(val2)
        rows.append(item)

    df_grid = pd.DataFrame(rows)
    X_feat = create_features(df_grid)
    n1, n2, _ = predict(model, X_feat)

    N1_grid = n1.reshape(grid_size, grid_size)
    N2_grid = n2.reshape(grid_size, grid_size)
    return G1, G2, N1_grid, N2_grid


def plot_sensitivity_analysis(
    results_dict: Dict[str, pd.DataFrame],
    ref_dict: Dict[str, float],
    save_path: Optional[Path] = None,
) -> None:
    """
    Deprecated facade delegating to src.visualization.fig4_sensitivity.
    Eliminates matplotlib dependency from core sensitivity calculations.
    """
    import warnings
    warnings.warn(
        "plot_sensitivity_analysis in src.sensitivity is deprecated. "
        "Use src.visualization.fig4_sensitivity.plot_sensitivity_sweeps_1d instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.visualization.fig4_sensitivity import plot_sensitivity_sweeps_1d

    output_dir = save_path.parent if save_path is not None else None
    figure_name = save_path.stem if save_path is not None else "sensitivity_full"
    formats = (save_path.suffix.lstrip(".").lower() or "png",) if save_path is not None else ("png",)

    plot_sensitivity_sweeps_1d(
        sensitivity_data=results_dict,
        ref_params=ref_dict,
        figure_name=figure_name,
        output_dir=output_dir,
        formats=formats,
        save_individual=False,
    )


def plot_combined_view(
    results_dict: Dict[str, pd.DataFrame],
    ref_dict: Dict[str, float],
    save_path: Optional[Path] = None,
) -> None:
    """
    Deprecated facade delegating to src.visualization.fig4_sensitivity.
    Eliminates matplotlib dependency from core sensitivity calculations.
    """
    import warnings
    warnings.warn(
        "plot_combined_view in src.sensitivity is deprecated. "
        "Use src.visualization.fig4_sensitivity.plot_sensitivity_sweeps_1d instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.visualization.fig4_sensitivity import plot_sensitivity_sweeps_1d

    output_dir = save_path.parent if save_path is not None else None
    figure_name = save_path.stem if save_path is not None else "sensitivity_combined"
    formats = (save_path.suffix.lstrip(".").lower() or "png",) if save_path is not None else ("png",)

    plot_sensitivity_sweeps_1d(
        sensitivity_data=results_dict,
        ref_params=ref_dict,
        figure_name=figure_name,
        output_dir=output_dir,
        formats=formats,
        save_individual=False,
    )


def plot_heatmap_interactions(
    model,
    ref_params: Optional[Dict[str, float]] = None,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    save_path: Optional[Path] = None,
    grid_size: int = 40,
) -> None:
    """
    Deprecated facade delegating to src.visualization.fig4_sensitivity.
    Eliminates matplotlib dependency from core sensitivity calculations.
    """
    import warnings
    warnings.warn(
        "plot_heatmap_interactions in src.sensitivity is deprecated. "
        "Use src.visualization.fig4_sensitivity.plot_sensitivity_heatmaps_2d instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.visualization.fig4_sensitivity import plot_sensitivity_heatmaps_2d

    if ref_params is None:
        ref_params = DEFAULT_REFERENCE_PARAMS.copy()
    if param_ranges is None:
        param_ranges = DEFAULT_PARAM_RANGES.copy()

    def predict_wrapper(df_geom: pd.DataFrame):
        X_feat = create_features(df_geom)
        return predict(model, X_feat)

    param_pairs = [
        ("Alfa", "Beta", param_ranges["Alfa"], param_ranges["Beta"]),
        ("H1", "H2", param_ranges["H1"], param_ranges["H2"]),
    ]

    output_dir = save_path.parent if save_path is not None else None
    figure_name = save_path.stem if save_path is not None else "sensitivity_heatmaps"
    formats = (save_path.suffix.lstrip(".").lower() or "png",) if save_path is not None else ("png",)

    plot_sensitivity_heatmaps_2d(
        predict_fn=predict_wrapper,
        ref_params=ref_params,
        param_pairs=param_pairs,
        grid_size=grid_size,
        figure_name=figure_name,
        output_dir=output_dir,
        formats=formats,
        save_individual=False,
    )


def run_sensitivity_analysis(
    output_dir: Optional[Path] = None,
    ref_params: Optional[Dict[str, float]] = None,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    n_points: int = 100,
    model: Optional[Any] = None,
    data_path: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main sensitivity analysis execution function: generates 1D response profiles, 2D heatmaps, and exports CSV.
    """
    if ref_params is None:
        ref_params = DEFAULT_REFERENCE_PARAMS.copy()
    if param_ranges is None:
        param_ranges = DEFAULT_PARAM_RANGES.copy()

    if model is None:
        print("[INFO] Loading surrogate model for sensitivity analysis...")
        try:
            model = load_model()
        except FileNotFoundError:
            print("[INFO] Model not found. Training surrogate model on available dataset...")
            model, _ = train_final_model(data_path=data_path)

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


__all__ = [
    "calculate_reference_predictions",
    "analyze_parameter_sensitivity",
    "compute_2d_grid",
    "plot_sensitivity_analysis",
    "plot_combined_view",
    "plot_heatmap_interactions",
    "run_sensitivity_analysis",
]
