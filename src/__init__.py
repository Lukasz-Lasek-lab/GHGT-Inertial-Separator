"""
MsCO2limit - Surrogate Modeling and Multi-Objective Optimization Framework for Inertial Particle Separators.
"""

from src.constants import (
    TOTAL_PARTICLES,
    BASE_FEATURES,
    TARGET_COLUMNS,
    TARGET_NAMES,
    PARAM_BOUNDS,
    NOMINAL_BASELINE,
    PARAM_UNITS,
)
from src.features import FEATURE_NAMES, create_features

try:
    from src.genetic import NSGA2Optimizer, run_genetic_optimization
except ImportError:
    NSGA2Optimizer = None
    run_genetic_optimization = None

try:
    from src.models import (
        build_regressor,
        evaluate_cv,
        evaluate_train_test,
        get_best_params,
        load_model,
        plot_actual_vs_predicted,
        predict,
        train_final_model,
    )
except ImportError:
    pass

try:
    from src.pareto_selection import partition_pareto_regimes, select_optimal_configurations
except ImportError:
    partition_pareto_regimes = None
    select_optimal_configurations = None

try:
    from src.sensitivity import compute_2d_grid, run_sensitivity_analysis
except ImportError:
    compute_2d_grid = None
    run_sensitivity_analysis = None

__all__ = [
    "TOTAL_PARTICLES",
    "BASE_FEATURES",
    "TARGET_COLUMNS",
    "TARGET_NAMES",
    "PARAM_BOUNDS",
    "NOMINAL_BASELINE",
    "PARAM_UNITS",
    "FEATURE_NAMES",
    "create_features",
    "build_regressor",
    "get_best_params",
    "load_model",
    "predict",
    "train_final_model",
    "evaluate_cv",
    "evaluate_train_test",
    "plot_actual_vs_predicted",
    "NSGA2Optimizer",
    "run_genetic_optimization",
    "partition_pareto_regimes",
    "select_optimal_configurations",
    "compute_2d_grid",
    "run_sensitivity_analysis",
]
