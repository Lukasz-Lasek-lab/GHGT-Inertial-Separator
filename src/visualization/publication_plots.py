"""
Scientific Visualization module for publication-quality figures (Q1 Journal Standard).
Backward-compatible facade re-exporting decomposed visualization modules.
"""

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

__all__ = [
    # Style & configuration
    "OKABE_ITO",
    "CVD_PALETTES",
    "LINE_STYLES",
    "MARKERS",
    "FEATURE_LABEL_MAP",
    "PARAM_AXIS_LABEL_MAP",
    "PARAM_NOMINAL_LABEL_MAP",
    "PUBLICATION_RC_PARAMS",
    "PARAM_BOUNDS",
    "figures_dir",
    "set_publication_style",
    "publication_style",
    "add_panel_label",
    "save_publication_figure",
    # Figure 2 Diagnostics
    "draw_parity_panel",
    "draw_residuals_panel",
    "plot_parity_single",
    "plot_residuals_single",
    "plot_model_diagnostics",
    # Figure 3 Feature Importance
    "draw_importance_bars",
    "draw_importance_comparison",
    "plot_feature_importance_single",
    "plot_feature_importance_comparison",
    "plot_feature_importance",
    # Figure 4 Sensitivity Sweeps & Contours
    "draw_sweep_1d_panel",
    "draw_contour_2d_panel",
    "plot_sensitivity_sweep_single",
    "plot_sensitivity_1d_single",
    "plot_sensitivity_sweeps_1d",
    "plot_sensitivity_1d",
    "plot_sensitivity_heatmap_single",
    "plot_sensitivity_heatmaps_2d",
    # Figure 5 Pareto Optimization
    "draw_convergence",
    "draw_pareto_front",
    "draw_inset_zoom",
    "plot_convergence",
    "plot_convergence_single",
    "plot_pareto_front",
    "plot_pareto_front_single",
    "plot_optimization_figure_5",
    # Figure 6 Case Study
    "resolve_case_study_data",
    "draw_efficiency_bars",
    "draw_advantage_bars",
    "draw_comparison_bars",
    "plot_case_study_efficiency",
    "plot_case_study_advantage",
    "plot_case_study_comparison",
]
