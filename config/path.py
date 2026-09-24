"""
Path and Directory Management for MsCO2limit project:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems'

Resolves paths dynamically relative to the project root with optional environment variable overrides.
No hardcoded operating system paths.
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

# Subdirectories
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"
results_test_5_dir = results_dir / "test_5"
models_test_5_dir = models_dir / "test_5"
results_genetic_dir = results_dir / "genetic"
plots_test_5_dir = plots_dir / "test_5"
plots_genetic_dir = plots_dir / "genetic"

ALL_DIRS: List[Path] = [
    data_dir,
    raw_data_dir,
    processed_data_dir,
    models_dir,
    models_test_5_dir,
    results_dir,
    results_test_5_dir,
    results_genetic_dir,
    plots_dir,
    plots_test_5_dir,
    plots_genetic_dir,
    figures_dir,
    config_dir,
]


def create_directories() -> None:
    """Creates required project directory hierarchy if it does not already exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "project_root",
    "results_dir",
    "plots_dir",
    "models_dir",
    "data_dir",
    "config_dir",
    "figures_dir",
    "raw_data_dir",
    "processed_data_dir",
    "results_test_5_dir",
    "models_test_5_dir",
    "results_genetic_dir",
    "plots_test_5_dir",
    "plots_genetic_dir",
    "create_directories",
]

# Ensure directory structure exists
create_directories()