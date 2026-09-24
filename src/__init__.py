"""
MsCO2limit - Pakiet modelowania ML i optymalizacji wielokryterialnej separatora cząstek.
"""

from src.features import BASE_FEATURES, FEATURE_NAMES, TARGET_NAMES, create_features

try:
    from src.genetic import run_genetic_optimization
except ImportError:
    run_genetic_optimization = None

try:
    from src.models import (
        build_regressor,
        evaluate_cv,
        evaluate_train_test,
        load_model,
        plot_actual_vs_predicted,
        predict,
        train_final_model,
    )
except ImportError:
    pass

try:
    from src.pareto_selection import select_optimal_configurations
except ImportError:
    select_optimal_configurations = None

try:
    from src.sensitivity import run_sensitivity_analysis
except ImportError:
    run_sensitivity_analysis = None

__all__ = [
    "BASE_FEATURES",
    "FEATURE_NAMES",
    "TARGET_NAMES",
    "create_features",
    "build_regressor",
    "load_model",
    "predict",
    "train_final_model",
    "evaluate_cv",
    "evaluate_train_test",
    "plot_actual_vs_predicted",
    "run_genetic_optimization",
    "select_optimal_configurations",
    "run_sensitivity_analysis",
]
