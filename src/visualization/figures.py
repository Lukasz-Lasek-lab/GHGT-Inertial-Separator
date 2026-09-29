"""
Legacy wrapper for src.visualization.orchestrator.
Maintains 100% backward compatibility for scripts, tests, and CLI entry points.
"""

from src.visualization.orchestrator import (
    DEFAULT_PARAM_RANGES,
    DEFAULT_REFERENCE_PARAMS,
    TOTAL_PARTICLES,
    generate_all_figures,
    generate_figure_2,
    generate_figure_3,
    generate_figure_4,
    generate_figure_5,
    generate_figure_6,
    main,
)

__all__ = [
    "generate_figure_2",
    "generate_figure_3",
    "generate_figure_4",
    "generate_figure_5",
    "generate_figure_6",
    "generate_all_figures",
    "DEFAULT_REFERENCE_PARAMS",
    "DEFAULT_PARAM_RANGES",
    "TOTAL_PARTICLES",
]

if __name__ == "__main__":
    main()
