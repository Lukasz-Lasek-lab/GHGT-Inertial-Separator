"""
Scientific Visualization module for the MsCO2limit project.
Provides publication-standard figure generators aligned with Q1 journal guidelines (Elsevier / Springer / ACS).
"""

from src.visualization.publication_plots import (
    CVD_PALETTES,
    LINE_STYLES,
    MARKERS,
    OKABE_ITO,
    add_panel_label,
    plot_case_study_comparison,
    plot_convergence,
    plot_feature_importance,
    plot_feature_importance_comparison,
    plot_feature_importance_single,
    plot_model_diagnostics,
    plot_pareto_front,
    plot_parity_single,
    plot_residuals_single,
    plot_sensitivity_heatmaps_2d,
    plot_sensitivity_sweeps_1d,
    publication_style,
    save_publication_figure,
    set_publication_style,
)

from src.visualization.figures import (
    generate_figure_2,
    generate_figure_3,
    generate_figure_4,
    generate_figure_5,
    generate_figure_6,
    generate_all_figures,
)

__all__ = [
    "set_publication_style",
    "publication_style",
    "save_publication_figure",
    "add_panel_label",
    "OKABE_ITO",
    "CVD_PALETTES",
    "MARKERS",
    "LINE_STYLES",
    "plot_parity_single",
    "plot_residuals_single",
    "plot_model_diagnostics",
    "plot_feature_importance",
    "plot_sensitivity_sweeps_1d",
    "plot_sensitivity_heatmaps_2d",
    "plot_convergence",
    "plot_pareto_front",
    "plot_case_study_comparison",
    "generate_figure_2",
    "generate_figure_3",
    "generate_figure_4",
    "generate_figure_5",
    "generate_figure_6",
    "generate_all_figures",
]
