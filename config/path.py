"""
Path and Directory Management for MsCO2limit project:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems'

Resolves paths dynamically relative to the project root with optional environment variable overrides.
Independent of operating system or machine-specific absolute paths.
"""

from pathlib import Path
import os
from typing import List

# Dynamic project root resolution (independent of OS or installation path)
project_root = Path(os.getenv("MSCO2_PROJECT_ROOT", Path(__file__).resolve().parent.parent))

# Primary project directories
data_dir = Path(os.getenv("MSCO2_DATA_DIR", project_root / "data"))
models_dir = Path(os.getenv("MSCO2_MODELS_DIR", project_root / "models"))
results_dir = project_root / "results"
plots_dir = project_root / "plots"
figures_dir = project_root / "figures"
config_dir = project_root / "config"
assets_dir = project_root / "assets"

# Subdirectories
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"
demo_data_dir = data_dir / "demo"
demo_data_file = demo_data_dir / "inertial_separator_demo.csv"

surrogate_models_dir = models_dir / "surrogate"
evaluation_results_dir = results_dir / "evaluation"
optimization_results_dir = results_dir / "optimization"
diagnostic_plots_dir = plots_dir / "diagnostics"
optimization_plots_dir = plots_dir / "optimization"
sensitivity_plots_dir = plots_dir / "sensitivity"

# Backward compatibility aliases for legacy scripts
models_test_5_dir = surrogate_models_dir
results_test_5_dir = evaluation_results_dir
results_genetic_dir = optimization_results_dir
plots_test_5_dir = diagnostic_plots_dir
plots_genetic_dir = optimization_plots_dir

ALL_DIRS: List[Path] = [
    data_dir,
    raw_data_dir,
    processed_data_dir,
    demo_data_dir,
    models_dir,
    surrogate_models_dir,
    results_dir,
    evaluation_results_dir,
    optimization_results_dir,
    plots_dir,
    diagnostic_plots_dir,
    optimization_plots_dir,
    sensitivity_plots_dir,
    figures_dir,
    config_dir,
    assets_dir,
]


def create_directories() -> None:
    """Creates required project directory hierarchy if it does not already exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "project_root",
    "data_dir",
    "raw_data_dir",
    "processed_data_dir",
    "demo_data_dir",
    "demo_data_file",
    "models_dir",
    "surrogate_models_dir",
    "results_dir",
    "evaluation_results_dir",
    "optimization_results_dir",
    "plots_dir",
    "diagnostic_plots_dir",
    "optimization_plots_dir",
    "sensitivity_plots_dir",
    "figures_dir",
    "config_dir",
    "assets_dir",
    # Legacy aliases
    "models_test_5_dir",
    "results_test_5_dir",
    "results_genetic_dir",
    "plots_test_5_dir",
    "plots_genetic_dir",
    "create_directories",
]

# Ensure directory structure exists
create_directories()