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
from typing import Optional, Tuple, Dict, Any, List

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


def run_cv_pipeline(data_path: Optional[Path] = None, use_demo: bool = False):
    """Executes 5-fold cross-validation evaluation pipeline."""
    if data_path is None:
        raw_or_active, processed_csv, _ = resolve_data_paths(use_demo=use_demo)
        data_path = processed_csv if processed_csv.exists() else raw_or_active
    plot_cv_path = diagnostic_plots_dir / "actual_vs_predicted_cv.png"
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



def main():
    parser = argparse.ArgumentParser(
        description="MsCO2limit ML & Evolutionary Optimization Pipeline"
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=[
            "select_features",
            "tune",
            "train",
            "cv",
            "eval",
            "optimize",
            "select",
            "sensitivity",
            "all",
        ],
        default="all",
        help="Computational stage to execute (default: all)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run pipeline in demonstration mode using synthetic benchmark dataset",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Force Bayesian hyperparameter optimization prior to training",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of Optuna evaluation trials (default: 200)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=100,
        help="NSGA-II population size (default: 100)",
    )
    parser.add_argument(
        "--n-gen",
        type=int,
        default=25,
        help="Number of NSGA-II evolutionary generations (default: 25)",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=100,
        help="Number of evaluation points in 1D sensitivity sweeps (default: 100)",
    )

    args = parser.parse_args()
    create_directories()

    print("=" * 70)
    print("MsCO2limit ML & NSGA-II Optimization Pipeline".center(70))
    print(f"Selected Stage: {args.step}".center(70))
    print("=" * 70)

    raw_or_active_path, processed_csv_path, features_json_path = resolve_data_paths(use_demo=args.demo)

    # 1. Feature selection (if requested or when selected features JSON is absent)
    if args.step == "select_features" or (args.step == "all" and not features_json_path.exists()):
        print("\n--- [STAGE: Automated Feature Engineering and Selection] ---")
        run_feature_selection(
            raw_path=raw_or_active_path,
            output_csv_path=processed_csv_path,
            json_path=features_json_path,
        )
        reload_features()

    active_data_path = processed_csv_path if processed_csv_path.exists() else raw_or_active_path

    # 2. Hyperparameter optimization
    best_params_json = config_dir / "best_params.json"
    if args.step == "tune" or args.tune or (args.step in ["train", "all"] and not best_params_json.exists()):
        print(f"\n--- [STAGE: Hyperparameter Tuning via Optuna ({args.n_trials} trials)] ---")
        tune_hyperparameters(
            data_path=active_data_path,
            n_trials=args.n_trials,
            save_json_path=best_params_json,
        )
    else:
        if args.step in ["train", "cv", "eval", "all"]:
            params = load_best_params()
            print(f"[INFO] Using cached surrogate hyperparameters from: {best_params_json.name}")

    # 3. Final model training
    if args.step in ["train", "all"]:
        print("\n--- [STAGE: Surrogate Model Training on Full Dataset] ---")
        train_final_model(data_path=active_data_path)

    # 4. Cross-validation and model evaluation
    if args.step in ["cv", "eval", "all"]:
        print("\n--- [STAGE: 5-Fold Cross-Validation and Predictive Evaluation] ---")
        plot_cv_path = diagnostic_plots_dir / "actual_vs_predicted_cv.png"
        df_folds, df_summary, oof_df = evaluate_cv(
            data_path=active_data_path, save_plot_path=plot_cv_path
        )
        print("\n5-Fold Cross-Validation Summary Metrics:")
        print(df_summary.to_string(index=False))

        plot_test_path = diagnostic_plots_dir / "actual_vs_predicted_test.png"
        _, metrics_test, _, _ = evaluate_train_test(
            data_path=active_data_path, save_plot_path=plot_test_path
        )
        print("\nTest Partition Evaluation (80/20 train/test):")
        for k, v in metrics_test.items():
            print(f"  {k:12s}: {v:.4f}")

    # 5. Multi-objective NSGA-II optimization
    pareto_df = None
    if args.step in ["optimize", "all"]:
        print("\n--- [STAGE: Multi-Objective NSGA-II Optimization] ---")
        pop_df, hof_df, pareto_df, log_df = run_genetic_optimization(
            data_path=active_data_path,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
        )

    # 6. Pareto front selection and physical balance verification
    if args.step in ["select", "all"]:
        print("\n--- [STAGE: Pareto Front Selection & Decision Support] ---")
        result_df = select_optimal_configurations(pareto_df=pareto_df)
        print("\nTop 5 Recommended Separator Configurations:")
        print(result_df.head(5).to_string(index=False))

    # 7. Comprehensive sensitivity analysis
    if args.step in ["sensitivity", "all"]:
        print("\n--- [STAGE: Sensitivity Analysis & 2D Response Contours] ---")
        run_sensitivity_analysis(n_points=args.n_points)

    print("\n" + "=" * 70)
    print("[OK] Pipeline execution completed successfully.".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
