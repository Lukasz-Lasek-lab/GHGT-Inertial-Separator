"""
Main Orchestrator and Computational Pipeline for the MsCO2limit project:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems'

Supports end-to-end execution or modular stages:
- select_features: Automated feature engineering and selection
- tune: Bayesian hyperparameter optimization via Optuna
- train: Full-dataset surrogate model training
- cv: 5-fold cross-validation and predictive evaluation
- optimize: NSGA-II multi-objective evolutionary optimization
- select: Pareto front clustering, deduplication, and particle balance
- sensitivity: 1D parameter sweeps and 2D interaction contour maps
- all: Sequential execution of the complete computational framework
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Sequence

# Ensure project root is available in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.path import (
    config_dir,
    create_directories,
    diagnostic_plots_dir,
    processed_data_dir,
    raw_data_dir,
    demo_data_file,
)
from src.feature_selection import run_feature_selection
from src.features import create_features, reload_features
from src.genetic import run_genetic_optimization
from src.models import (
    evaluate_cv,
    evaluate_train_test,
    load_best_params,
    train_final_model,
    tune_hyperparameters,
)
from src.pareto_selection import select_optimal_configurations
from src.sensitivity import run_sensitivity_analysis


def resolve_data_paths(use_demo: bool = False):
    """Resolves data paths based on availability and demo mode flag."""
    features_json_path = config_dir / "selected_features.json"
    processed_csv_path = processed_data_dir / "df_selected.csv"

    if use_demo:
        print("[INFO] Operating in DEMONSTRATION mode with synthetic benchmark data.")
        demo_processed_path = processed_data_dir / "df_selected_demo.csv"
        return demo_data_file, demo_processed_path, features_json_path

    if processed_csv_path.exists():
        return processed_csv_path, processed_csv_path, features_json_path

    raw_candidates = [
        raw_data_dir / "inertial_separator_cfd.xlsx",
        raw_data_dir / "Dane_T5.xlsx",
    ]
    raw_path = next((p for p in raw_candidates if p.exists()), None)
    if raw_path is not None:
        return raw_path, processed_csv_path, features_json_path

    # Graceful fallback to demo dataset
    if demo_data_file.exists():
        print("[NOTICE] Proprietary CFD dataset not detected. Utilizing demonstration dataset.")
        print("[NOTICE] (See README.md Data and Model Availability Statement to request research data).")
        demo_processed_path = processed_data_dir / "df_selected_demo.csv"
        return demo_data_file, demo_processed_path, features_json_path

    raise FileNotFoundError(
        "No dataset found. Please provide data in data/raw/ or run with '--demo'."
    )


def run_cv_pipeline(
    data_path: Optional[Path] = None,
    use_demo: bool = False,
    save_plot: bool = True,
):
    """Executes 5-fold cross-validation evaluation pipeline."""
    if data_path is None:
        raw_or_active, processed_csv, _ = resolve_data_paths(use_demo=use_demo)
        data_path = processed_csv if processed_csv.exists() else raw_or_active
    plot_cv_path = diagnostic_plots_dir / "actual_vs_predicted_cv.png" if save_plot else None
    df_folds, df_summary, oof_df = evaluate_cv(data_path=data_path, save_plot_path=plot_cv_path)
    print("\n--- 5-Fold Cross-Validation Metrics Summary ---")
    print(df_summary.to_string(index=False))
    return df_folds, df_summary, oof_df


def run_training_pipeline(data_path: Optional[Path] = None, use_demo: bool = False):
    """Trains final production surrogate model."""
    if data_path is None:
        raw_or_active, processed_csv, _ = resolve_data_paths(use_demo=use_demo)
        data_path = processed_csv if processed_csv.exists() else raw_or_active
    return train_final_model(data_path=data_path)


def run_optimization_pipeline(data_path: Optional[Path] = None, use_demo: bool = False, pop_size: int = 100, n_gen: int = 25):
    """Executes multi-objective NSGA-II optimization algorithm."""
    if data_path is None:
        raw_or_active, processed_csv, _ = resolve_data_paths(use_demo=use_demo)
        data_path = processed_csv if processed_csv.exists() else raw_or_active
    return run_genetic_optimization(data_path=data_path, pop_size=pop_size, n_gen=n_gen)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Legacy pipeline entry point.
    Delegates argument parsing and execution to src.cli.main.
    """
    from src.cli import main as cli_main
    return cli_main(argv=argv, default_step="all")


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "resolve_data_paths",
    "run_cv_pipeline",
    "run_training_pipeline",
    "run_optimization_pipeline",
    "main",
]

