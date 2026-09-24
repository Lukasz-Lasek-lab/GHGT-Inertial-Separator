"""
Unified Command-Line Interface (CLI) for MsCO2limit:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems'

Usage:
    python -m src --help
    python -m src --figure 2
    python -m src --all-figures
    python -m src --cv
    python -m src --train
    python -m src --optimize
    python -m src --all
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "MsCO2limit ML & NSGA-II Optimization Framework: "
            "Hybrid AI-driven Approach for Optimization and Geometric Analysis of "
            "Inertial Separators in Chemical Looping Systems"
        ),
    )

    group_action = parser.add_argument_group("Execution actions")
    group_action.add_argument(
        "--figure",
        type=int,
        choices=[2, 3, 4, 5, 6],
        help="Generate a specific publication figure (2: Diagnostics, 3: Importance, 4: Sensitivity, 5: NSGA-II/Pareto, 6: Case Study).",
    )
    group_action.add_argument(
        "--all-figures",
        action="store_true",
        help="Generate all publication figures (Fig 2 - Fig 6) in PDF, SVG, and PNG formats.",
    )
    group_action.add_argument(
        "--cv",
        action="store_true",
        help="Run 5-fold cross-validation pipeline on selected features.",
    )
    group_action.add_argument(
        "--train",
        action="store_true",
        help="Train the final production surrogate model (HistGradientBoostingRegressor).",
    )
    group_action.add_argument(
        "--optimize",
        action="store_true",
        help="Execute multi-objective NSGA-II genetic optimization algorithm.",
    )
    group_action.add_argument(
        "--all",
        action="store_true",
        help="Execute full pipeline: Cross-validation, Model Training, NSGA-II Optimization, and Figures.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom destination directory for generated figures/results.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        help="List of graphic formats to export (default: pdf svg png).",
    )

    args = parser.parse_args()

    # If no arguments provided, print help
    if not (args.figure or args.all_figures or args.cv or args.train or args.optimize or args.all):
        parser.print_help()
        sys.exit(0)

    from config.path import figures_dir, processed_data_dir
    import src.visualization.figures as fig_module

    out_dir = Path(args.output_dir) if args.output_dir else figures_dir
    formats = tuple(args.formats)

    if args.cv or args.all:
        from src.pipeline import run_cv_pipeline
        print("\n>>> Executing 5-Fold Cross-Validation Pipeline...")
        run_cv_pipeline()

    if args.train or args.all:
        from src.pipeline import run_training_pipeline
        print("\n>>> Training Final Production Surrogate Model...")
        run_training_pipeline()

    if args.optimize or args.all:
        from src.genetic import run_genetic_optimization
        print("\n>>> Running NSGA-II Multi-Objective Genetic Optimization...")
        run_genetic_optimization()

    if args.figure == 2 or (args.all and not args.figure):
        fig_module.generate_figure_2(output_dir=out_dir, formats=formats)
    elif args.figure == 3:
        fig_module.generate_figure_3(output_dir=out_dir, formats=formats)
    elif args.figure == 4:
        fig_module.generate_figure_4(output_dir=out_dir, formats=formats)
    elif args.figure == 5:
        fig_module.generate_figure_5(output_dir=out_dir, formats=formats)
    elif args.figure == 6:
        fig_module.generate_figure_6(output_dir=out_dir, formats=formats)

    if args.all_figures or args.all:
        fig_module.generate_all_figures(output_dir=out_dir, formats=formats)


if __name__ == "__main__":
    main()
