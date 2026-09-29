"""
Scientific Visualization module for the MsCO2limit project.
Provides publication-standard figure generators aligned with Q1 journal guidelines (Elsevier / Springer / ACS).
"""

from src.visualization.style import (
    CVD_PALETTES,
    FEATURE_LABEL_MAP,
    LINE_STYLES,
    MARKERS,
    OKABE_ITO,
    PARAM_AXIS_LABEL_MAP,
    PARAM_BOUNDS,
    PARAM_NOMINAL_LABEL_MAP,
    PUBLICATION_RC_PARAMS,
    add_panel_label,
    figures_dir,
    publication_style,
    save_publication_figure,
    set_publication_style,
)
from src.visualization.fig2_diagnostics import (
    draw_parity_panel,
    draw_residuals_panel,
    plot_model_diagnostics,
    plot_parity_single,
    plot_residuals_single,
)
from src.visualization.fig3_importance import (
    draw_importance_bars,
    draw_importance_comparison,
    plot_feature_importance,
    plot_feature_importance_comparison,
    plot_feature_importance_single,
)
from src.visualization.fig4_sensitivity import (
    draw_contour_2d_panel,
    draw_sweep_1d_panel,
    plot_sensitivity_1d,
    plot_sensitivity_1d_single,
    plot_sensitivity_heatmap_single,
    plot_sensitivity_heatmaps_2d,
    plot_sensitivity_sweep_single,
    plot_sensitivity_sweeps_1d,
)
from src.visualization.fig5_pareto import (
    draw_convergence,
    draw_inset_zoom,
    draw_pareto_front,
    plot_convergence,
    plot_convergence_single,
    plot_optimization_figure_5,
    plot_pareto_front,
    plot_pareto_front_single,
)
from src.visualization.fig6_case_study import (
    draw_advantage_bars,
    draw_comparison_bars,
    draw_efficiency_bars,
    plot_case_study_advantage,
    plot_case_study_comparison,
    plot_case_study_efficiency,
    resolve_case_study_data,
)
def __getattr__(name: str):
    _orch_symbols = {
        "generate_figure_2",
        "generate_figure_3",
        "generate_figure_4",
        "generate_figure_5",
        "generate_figure_6",
        "generate_all_figures",
        "DEFAULT_PARAM_RANGES",
        "DEFAULT_REFERENCE_PARAMS",
        "TOTAL_PARTICLES",
    }
    if name in _orch_symbols:
        from src.visualization import orchestrator
        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Style & configuration
    "set_publication_style",
    "publication_style",
    "save_publication_figure",
    "add_panel_label",
    "OKABE_ITO",
    "CVD_PALETTES",
    "MARKERS",
    "LINE_STYLES",
    "FEATURE_LABEL_MAP",
    "PARAM_AXIS_LABEL_MAP",
    "PARAM_NOMINAL_LABEL_MAP",
    "PUBLICATION_RC_PARAMS",
    "PARAM_BOUNDS",
    "figures_dir",
    # Fig 2 Diagnostics
    "draw_parity_panel",
    "draw_residuals_panel",
    "plot_parity_single",
    "plot_residuals_single",
    "plot_model_diagnostics",
    # Fig 3 Feature Importance
    "draw_importance_bars",
    "draw_importance_comparison",
    "plot_feature_importance_single",
    "plot_feature_importance_comparison",
    "plot_feature_importance",
    # Fig 4 Sensitivity Sweeps & Contours
    "draw_sweep_1d_panel",
    "draw_contour_2d_panel",
    "plot_sensitivity_sweep_single",
    "plot_sensitivity_1d_single",
    "plot_sensitivity_sweeps_1d",
    "plot_sensitivity_1d",
    "plot_sensitivity_heatmap_single",
    "plot_sensitivity_heatmaps_2d",
    # Fig 5 Pareto Optimization
    "draw_convergence",
    "draw_pareto_front",
    "draw_inset_zoom",
    "plot_convergence",
    "plot_convergence_single",
    "plot_pareto_front",
    "plot_pareto_front_single",
    "plot_optimization_figure_5",
    # Fig 6 Case Study
    "resolve_case_study_data",
    "draw_efficiency_bars",
    "draw_advantage_bars",
    "draw_comparison_bars",
    "plot_case_study_efficiency",
    "plot_case_study_advantage",
    "plot_case_study_comparison",
    # Orchestrator
    "generate_figure_2",
    "generate_figure_3",
    "generate_figure_4",
    "generate_figure_5",
    "generate_figure_6",
    "generate_all_figures",
    "DEFAULT_PARAM_RANGES",
    "DEFAULT_REFERENCE_PARAMS",
    "TOTAL_PARTICLES",
]
