"""
Unified Command-Line Interface (CLI) and Orchestrator for MsCO2limit:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems'

Combines computational pipeline execution, hyperparameter optimization, surrogate modeling,
NSGA-II multi-objective optimization, and publication-standard figure generation into a
cohesive, modular, and robust CLI.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Ensure project root is available in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.path import (
    config_dir,
    create_directories,
    demo_data_file,
    diagnostic_plots_dir,
    figures_dir as default_figures_dir,
    optimization_results_dir,
    processed_data_dir,
)
from src.feature_selection import run_feature_selection
from src.features import reload_features
from src.genetic import run_genetic_optimization
from src.models import (
    evaluate_cv,
    evaluate_train_test,
    load_best_params,
    load_model,
    train_final_model,
    tune_hyperparameters,
)
from src.pareto_selection import select_optimal_configurations
from src.pipeline import resolve_data_paths
from src.sensitivity import run_sensitivity_analysis
from src.visualization.orchestrator import (
    generate_all_figures,
    generate_figure_2,
    generate_figure_3,
    generate_figure_4,
    generate_figure_5,
    generate_figure_6,
)

STEP_ALIASES: Dict[str, str] = {
    "eval": "cv",
    "select": "pareto",
}

VALID_STEPS: List[str] = [
    "select_features",
    "tune",
    "train",
    "cv",
    "optimize",
    "pareto",
    "sensitivity",
    "figures",
    "all",
]


@dataclass
class PipelineContext:
    """
    Encapsulates execution state, environment flags, paths, surrogate models,
    and computational hyperparameters across all pipeline stages.
    """

    demo: bool = False
    output_dir: Optional[Path] = None
    formats: Sequence[str] = ("pdf", "svg", "png")
    data_path: Optional[Path] = None
    raw_path: Optional[Path] = None
    processed_path: Optional[Path] = None
    features_json_path: Optional[Path] = None
    model: Optional[Any] = None
    pop_size: int = 100
    n_gen: int = 25
    n_trials: int = 200
    n_points: int = 100
    step: Optional[str] = "all"
    figure: Optional[Union[int, str]] = None
    tune: bool = False
    results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_dir is not None and not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.formats, (list, tuple)):
            self.formats = tuple(f.lower().strip().lstrip(".") for f in self.formats)
        if self.step is not None:
            norm_step = str(self.step).lower().strip()
            self.step = STEP_ALIASES.get(norm_step, norm_step)
        if self.figure is not None:
            fig_str = str(self.figure).lower().strip()
            for prefix in ("figure_", "figure", "fig_", "fig"):
                if fig_str.startswith(prefix):
                    fig_str = fig_str[len(prefix):].strip()
                    break
            self.figure = fig_str

    def ensure_directories(self) -> None:
        """Ensures that all project output and config directories exist."""
        create_directories()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_paths(self) -> Tuple[Path, Path, Path]:
        """
        Resolves raw, processed, and feature configuration paths based on demo mode.
        Guarantees that raw demo data is never overwritten.
        """
        raw_p, proc_p, feat_json = resolve_data_paths(use_demo=self.demo)
        self.raw_path = raw_p
        self.processed_path = proc_p
        self.features_json_path = feat_json

        if self.data_path is None:
            self.data_path = self.processed_path if self.processed_path.exists() else self.raw_path
        return self.raw_path, self.processed_path, self.features_json_path

    def get_active_data_path(self) -> Path:
        """Returns the currently active dataset file path, resolving paths if needed."""
        if self.data_path is None or self.raw_path is None:
            self.resolve_paths()
        return self.data_path  # type: ignore

    def ensure_model(self, force_retrain: bool = False) -> Any:
        """Returns active surrogate model, loading from cache or training if necessary."""
        if self.model is not None and not force_retrain:
            return self.model

        if not force_retrain:
            try:
                self.model = load_model()
                return self.model
            except FileNotFoundError:
                pass

        data_file = self.get_active_data_path()
        model, _ = train_final_model(data_path=data_file)
        self.model = model
        return self.model

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        default_step: Optional[str] = None,
    ) -> PipelineContext:
        """Constructs and returns a PipelineContext from parsed command-line arguments."""
        raw_step = getattr(args, "step", None)
        has_all = getattr(args, "all", False)
        has_all_figures = getattr(args, "all_figures", False)
        has_cv = getattr(args, "cv", False)
        has_train = getattr(args, "train", False)
        has_optimize = getattr(args, "optimize", False)
        has_tune = getattr(args, "tune", False)
        figure_arg = getattr(args, "figure", None)

        if raw_step is not None:
            effective_step = raw_step
        elif has_all:
            effective_step = "all"
        elif has_all_figures:
            effective_step = "figures"
            figure_arg = "all"
        elif has_cv:
            effective_step = "cv"
        elif has_train:
            effective_step = "train"
        elif has_optimize:
            effective_step = "optimize"
        elif figure_arg is not None:
            effective_step = "figures"
        elif has_tune:
            effective_step = "tune"
        else:
            effective_step = default_step

        out_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else None
        formats = tuple(args.formats) if getattr(args, "formats", None) else ("pdf", "svg", "png")

        return cls(
            demo=bool(getattr(args, "demo", False)),
            output_dir=out_dir,
            formats=formats,
            step=effective_step,
            figure=figure_arg,
            tune=bool(has_tune),
            pop_size=int(getattr(args, "pop_size", 100)),
            n_gen=int(getattr(args, "n_gen", 25)),
            n_trials=int(getattr(args, "n_trials", 200)),
            n_points=int(getattr(args, "n_points", 100)),
        )


def run_step_select_features(context: PipelineContext) -> None:
    """Executes automated feature engineering and selection stage."""
    print("\n--- [STAGE: Automated Feature Engineering and Selection] ---")
    raw_path, processed_path, json_path = context.resolve_paths()
    run_feature_selection(
        raw_path=raw_path,
        output_csv_path=processed_path,
        json_path=json_path,
    )
    reload_features()
    context.data_path = processed_path
    context.results["features_json"] = json_path
    context.results["processed_csv"] = processed_path


def run_step_tune(context: PipelineContext) -> Dict[str, Any]:
    """Executes Optuna hyperparameter optimization stage."""
    active_data = context.get_active_data_path()
    best_params_json = config_dir / "best_params.json"
    print(f"\n--- [STAGE: Hyperparameter Tuning via Optuna ({context.n_trials} trials)] ---")
    best_params = tune_hyperparameters(
        data_path=active_data,
        n_trials=context.n_trials,
        save_json_path=best_params_json,
    )
    context.results["best_params"] = best_params
    return best_params


def run_step_train(context: PipelineContext) -> Any:
    """Executes surrogate model training on the full active dataset."""
    active_data = context.get_active_data_path()
    print("\n--- [STAGE: Surrogate Model Training on Full Dataset] ---")
    model, model_path = train_final_model(data_path=active_data)
    context.model = model
    context.results["model"] = model
    context.results["model_path"] = model_path
    return model


def run_step_cv(
    context: PipelineContext,
    save_plot: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    Executes 5-fold cross-validation and predictive evaluation.
    Caches OOF predictions to disk so publication Figure 2 does not recalculate CV.
    If save_plot is False (e.g. during full pipeline), skips duplicate rendering of Figure 2.
    """
    active_data = context.get_active_data_path()
    print("\n--- [STAGE: 5-Fold Cross-Validation and Predictive Evaluation] ---")
    plot_cv_path = diagnostic_plots_dir / "actual_vs_predicted_cv.png" if save_plot else None

    df_folds, df_summary, oof_df = evaluate_cv(
        data_path=active_data,
        save_plot_path=plot_cv_path,
    )

    # Cache OOF predictions for Figure 2 diagnostics
    oof_filename = "oof_predictions_demo.csv" if context.demo else "oof_predictions.csv"
    oof_cache = processed_data_dir / oof_filename
    oof_cache.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_cache, index=False)

    print("\n5-Fold Cross-Validation Summary Metrics:")
    print(df_summary.to_string(index=False))

    plot_test_path = diagnostic_plots_dir / "actual_vs_predicted_test.png" if save_plot else None
    _, metrics_test, _, _ = evaluate_train_test(
        data_path=active_data,
        save_plot_path=plot_test_path,
    )
    print("\nTest Partition Evaluation (80/20 train/test):")
    for k, v in metrics_test.items():
        print(f"  {k:12s}: {v:.4f}")

    context.results["df_folds"] = df_folds
    context.results["df_summary"] = df_summary
    context.results["oof_df"] = oof_df
    context.results["metrics_test"] = metrics_test
    return df_folds, df_summary, oof_df


def run_step_optimize(context: PipelineContext) -> Tuple[Any, Any, Any, Any]:
    """Executes multi-objective NSGA-II genetic optimization algorithm."""
    active_data = context.get_active_data_path()
    print(
        f"\n--- [STAGE: Multi-Objective NSGA-II Optimization "
        f"(pop={context.pop_size}, gen={context.n_gen})] ---"
    )
    pop_df, hof_df, pareto_df, log_df = run_genetic_optimization(
        data_path=active_data,
        pop_size=context.pop_size,
        n_gen=context.n_gen,
    )
    context.results["pop_df"] = pop_df
    context.results["hof_df"] = hof_df
    context.results["pareto_df"] = pareto_df
    context.results["log_df"] = log_df
    return pop_df, hof_df, pareto_df, log_df


def run_step_pareto(context: PipelineContext) -> Any:
    """Executes Pareto front selection, clustering, and decision support analysis."""
    print("\n--- [STAGE: Pareto Front Selection & Decision Support] ---")
    pareto_df = context.results.get("pareto_df")
    if pareto_df is None:
        pareto_candidates = [
            optimization_results_dir / "pareto_optimal_designs.csv",
            optimization_results_dir / "pareto_front.csv",
            optimization_results_dir / "Test5_Results_src.csv",
        ]
        pareto_candidates.extend(list(optimization_results_dir.glob("pareto_front*.csv")))
        pareto_candidates.extend(list(optimization_results_dir.glob("genetic_results_pareto_*.csv")))
        has_existing = any(p.exists() and p.stat().st_size > 0 for p in pareto_candidates)
        if not has_existing:
            print("[NOTICE] No prior Pareto front results found. Executing NSGA-II optimization first...")
            _, _, pareto_df, _ = run_step_optimize(context)

    out_excel = (
        context.output_dir / "pareto_optimal_designs.xlsx"
        if context.output_dir is not None
        else None
    )
    result_df = select_optimal_configurations(
        pareto_df=pareto_df,
        output_excel_path=out_excel,
    )
    context.results["optimal_designs"] = result_df
    print("\nTop 5 Recommended Separator Configurations:")
    print(result_df.head(5).to_string(index=False))
    return result_df


def run_step_sensitivity(context: PipelineContext) -> Dict[str, Any]:
    """Executes 1D parameter sweeps and 2D interaction response analysis."""
    print(f"\n--- [STAGE: Sensitivity Analysis & 2D Response Contours (n_points={context.n_points})] ---")
    model = context.ensure_model()
    res = run_sensitivity_analysis(
        model=model,
        n_points=context.n_points,
        output_dir=context.output_dir,
    )
    context.results["sensitivity"] = res
    return res


def run_step_figures(context: PipelineContext) -> Dict[str, Any]:
    """
    Executes publication figure generation via orchestrator.
    Renders requested figure(s) exactly once with publication styling.
    """
    out_dir = context.output_dir or default_figures_dir
    formats = context.formats
    use_demo = context.demo
    fig_req = str(context.figure).lower() if context.figure is not None else "all"

    print("\n--- [STAGE: Publication Figures Generation] ---")
    print(
        f"[INFO] Destination: {out_dir} | "
        f"Formats: {list(formats)} | "
        f"Mode: {'DEMO' if use_demo else 'FULL'}"
    )

    fig_results: Dict[str, Any] = {}
    if fig_req in ("all", "none"):
        fig_results = generate_all_figures(
            output_dir=out_dir,
            formats=formats,
            use_demo=use_demo,
            n_points_1d=context.n_points,
        )
    elif fig_req == "2":
        fig_results["Fig2"] = generate_figure_2(output_dir=out_dir, formats=formats, use_demo=use_demo)
    elif fig_req == "3":
        fig_results["Fig3"] = generate_figure_3(output_dir=out_dir, formats=formats, use_demo=use_demo)
    elif fig_req == "4":
        fig_results["Fig4"] = generate_figure_4(
            output_dir=out_dir,
            formats=formats,
            n_points_1d=context.n_points,
            use_demo=use_demo,
        )
    elif fig_req == "5":
        fig_results["Fig5"] = generate_figure_5(output_dir=out_dir, formats=formats, use_demo=use_demo)
    elif fig_req == "6":
        fig_results["Fig6"] = generate_figure_6(output_dir=out_dir, formats=formats, use_demo=use_demo)
    else:
        raise ValueError(
            f"Unknown figure selection '{context.figure}'. Choose from [2, 3, 4, 5, 6, all]."
        )

    context.results["figures"] = fig_results
    return fig_results


def run_step_all(context: PipelineContext) -> Dict[str, Any]:
    """
    Sequentially executes the complete computational framework:
    1. Automated Feature Engineering and Selection
    2. Hyperparameter Tuning
    3. Surrogate Model Training
    4. 5-Fold Cross-Validation (with save_plot=False to avoid duplicate Figure 2 rendering)
    5. Multi-Objective NSGA-II Genetic Optimization
    6. Pareto Front Selection and Physical Clustering
    7. Sensitivity Analysis sweeps
    8. Publication Figures Generation (Figure 2 - 6 rendered once in publication quality)
    """
    _, proc_path, feat_json = context.resolve_paths()

    # 1. Feature selection
    needs_features = not feat_json.exists() or not proc_path.exists()
    if needs_features:
        run_step_select_features(context)
    else:
        print(f"\n[INFO] Reusing existing feature configuration from: {feat_json.name}")

    # 2. Hyperparameter optimization
    best_params_json = config_dir / "best_params.json"
    if context.tune or not best_params_json.exists():
        run_step_tune(context)
    else:
        load_best_params()
        print(f"\n[INFO] Reusing cached surrogate hyperparameters from: {best_params_json.name}")

    # 3. Model training
    run_step_train(context)

    # 4. Cross-validation (save_plot=False ensures Figure 2 is NOT rendered here)
    run_step_cv(context, save_plot=False)

    # 5. NSGA-II Optimization
    run_step_optimize(context)

    # 6. Pareto selection
    run_step_pareto(context)

    # 7. Sensitivity analysis
    run_step_sensitivity(context)

    # 8. Publication figures
    run_step_figures(context)

    return context.results


def execute_pipeline(context: PipelineContext) -> Dict[str, Any]:
    """
    Dispatches and executes the pipeline according to context.step.
    Ensures directories and resolved paths are configured prior to execution.
    """
    context.ensure_directories()
    context.resolve_paths()

    step = STEP_ALIASES.get(str(context.step), str(context.step))

    mode_label = "DEMONSTRATION (Synthetic Data)" if context.demo else "STANDARD / FULL (CFD Data)"
    print("=" * 72)
    print("MsCO2limit ML & NSGA-II Optimization Framework".center(72))
    print(f"Executing Stage: {step.upper()} | Mode: {mode_label}".center(72))
    print("=" * 72)

    if step == "select_features":
        run_step_select_features(context)
    elif step == "tune":
        run_step_tune(context)
    elif step == "train":
        run_step_train(context)
    elif step == "cv":
        run_step_cv(context, save_plot=True)
    elif step == "optimize":
        run_step_optimize(context)
    elif step == "pareto":
        run_step_pareto(context)
    elif step == "sensitivity":
        run_step_sensitivity(context)
    elif step == "figures":
        run_step_figures(context)
    elif step == "all":
        run_step_all(context)
    else:
        raise ValueError(
            f"Unknown execution stage: '{step}'. "
            f"Supported stages: {', '.join(VALID_STEPS)}"
        )

    print("\n" + "=" * 72)
    print(f"[OK] Pipeline stage '{step}' completed successfully.".center(72))
    print("=" * 72)
    return context.results


def _normalize_figure_choice(val: str) -> str:
    s = str(val).lower().strip()
    for prefix in ("figure_", "figure", "fig_", "fig"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    if s not in ("2", "3", "4", "5", "6", "all"):
        raise argparse.ArgumentTypeError(
            f"Invalid figure selection '{val}'. Choose from 2, 3, 4, 5, 6, all."
        )
    return s


def build_parser() -> argparse.ArgumentParser:
    """Builds the comprehensive command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "MsCO2limit ML & Evolutionary Optimization Pipeline: "
            "Hybrid AI-driven Approach for Optimization and Geometric Analysis of "
            "Inertial Separators in Chemical Looping Systems"
        ),
    )

    # Step selection
    parser.add_argument(
        "--step",
        type=str.lower,
        choices=[
            "select_features",
            "tune",
            "train",
            "cv",
            "eval",
            "optimize",
            "pareto",
            "select",
            "sensitivity",
            "figures",
            "all",
        ],
        default=None,
        help="Computational stage to execute: select_features, tune, train, cv, optimize, pareto, sensitivity, figures, all (default: all)",
    )

    # Figure selection
    parser.add_argument(
        "--figure",
        type=_normalize_figure_choice,
        choices=["2", "3", "4", "5", "6", "all"],
        default=None,
        help="Generate a specific publication figure (2, 3, 4, 5, 6, or all)",
    )

    # Global execution mode
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run pipeline in demonstration mode using synthetic benchmark dataset",
    )

    # Convenience action shortcuts
    group_actions = parser.add_argument_group("Action shortcuts (alternative to --step)")
    group_actions.add_argument(
        "--all",
        action="store_true",
        help="Execute complete end-to-end framework",
    )
    group_actions.add_argument(
        "--cv",
        action="store_true",
        help="Run 5-fold cross-validation pipeline",
    )
    group_actions.add_argument(
        "--train",
        action="store_true",
        help="Train final production surrogate model",
    )
    group_actions.add_argument(
        "--optimize",
        action="store_true",
        help="Execute NSGA-II multi-objective genetic optimization",
    )
    group_actions.add_argument(
        "--all-figures",
        action="store_true",
        help="Generate all publication figures (Fig 2 - Fig 6)",
    )
    group_actions.add_argument(
        "--tune",
        action="store_true",
        help="Force Optuna Bayesian hyperparameter optimization prior to training",
    )

    # Computational hyperparameters
    group_params = parser.add_argument_group("Computational hyperparameters")
    group_params.add_argument(
        "--n-trials",
        type=int,
        default=200,
        help="Number of Optuna evaluation trials (default: 200)",
    )
    group_params.add_argument(
        "--pop-size",
        type=int,
        default=100,
        help="NSGA-II population size (default: 100)",
    )
    group_params.add_argument(
        "--n-gen",
        type=int,
        default=25,
        help="Number of NSGA-II evolutionary generations (default: 25)",
    )
    group_params.add_argument(
        "--n-points",
        type=int,
        default=100,
        help="Number of evaluation points in 1D sensitivity sweeps (default: 100)",
    )

    # Output configuration
    group_out = parser.add_argument_group("Output configuration")
    group_out.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom destination directory for generated figures/artifacts",
    )
    group_out.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        help="List of graphic formats to export (default: pdf svg png)",
    )

    return parser


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments using the unified parser."""
    parser = build_parser()
    return parser.parse_args(args)


def main(
    argv: Optional[Sequence[str]] = None,
    default_step: Optional[str] = None,
) -> int:
    """
    Universal Command-Line Interface entry point.
    Parses arguments, builds PipelineContext, and executes the designated stages.
    """
    parser = build_parser()
    parsed_args = parser.parse_args(argv)

    context = PipelineContext.from_args(parsed_args, default_step=default_step)

    if context.step is None:
        parser.print_help()
        return 0

    execute_pipeline(context)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "PipelineContext",
    "build_parser",
    "parse_args",
    "execute_pipeline",
    "main",
    "run_step_select_features",
    "run_step_tune",
    "run_step_train",
    "run_step_cv",
    "run_step_optimize",
    "run_step_pareto",
    "run_step_sensitivity",
    "run_step_figures",
    "run_step_all",
    "STEP_ALIASES",
    "VALID_STEPS",
]
