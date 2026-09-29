"""
Publication Styling and Color Vision Deficiency (CVD) Palettes.
Complies with Q1 chemical engineering and scientific publishing standards
(Nature Methods / Elsevier / ACS / Springer).
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt

from config.path import figures_dir
from src.constants import PARAM_BOUNDS

# ==============================================================================
# 1. COLOR VISION DEFICIENCY (CVD) FRIENDLY COLOR PALETTES & PRINT STYLES
# ==============================================================================

# Universal Okabe-Ito colorblind-safe palette (Wong, 2011, Nature Methods)
OKABE_ITO: List[str] = [
    "#0072B2",  # Blue
    "#D55E00",  # Vermillion / Red-Orange
    "#009E73",  # Bluish Green
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#CC79A7",  # Reddish Purple
    "#F0E442",  # Yellow
    "#000000",  # Black
]

CVD_PALETTES: Dict[str, List[str]] = {
    "okabe_ito": OKABE_ITO,
    "two_tone": ["#0072B2", "#D55E00"],
    "three_tone": ["#0072B2", "#D55E00", "#009E73"],
    "four_tone": ["#0072B2", "#D55E00", "#009E73", "#E69F00"],
}

# Line styles and markers for black-and-white / grayscale print clarity
LINE_STYLES: List[str] = ["-", "--", "-.", ":"]
MARKERS: List[str] = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]

# Mapping feature names to LaTeX mathematical expressions and units
FEATURE_LABEL_MAP: Dict[str, str] = {
    "Alfa": r"$\alpha$ [deg]",
    "Beta": r"$\beta$ [deg]",
    "H1": r"$H_1$ [m]",
    "H2": r"$H_2$ [m]",
    "H1_div_H2": r"$H_1 / H_2$ [-]",
    "H1_plus_H2": r"$H_1 + H_2$ [m]",
    "Alfa_plus_Beta": r"$\alpha + \beta$ [deg]",
    "Alfa_Beta": r"$\alpha \cdot \beta$ [$\mathrm{deg}^2$]",
    "Alfa_div_Beta": r"$\alpha / \beta$ [-]",
    "Alfa_minus_Beta": r"$|\alpha - \beta|$ [deg]",
    "H1_minus_H2": r"$|H_1 - H_2|$ [m]",
    "log_H1": r"$\ln(H_1)$ [-]",
    "log_H2": r"$\ln(H_2)$ [-]",
    "sin_Alfa": r"$\sin(\alpha)$ [-]",
    "cos_Alfa": r"$\cos(\alpha)$ [-]",
    "sin_Beta": r"$\sin(\beta)$ [-]",
    "cos_Beta": r"$\cos(\beta)$ [-]",
    "Alfa_squared": r"$\alpha^2$ [$\mathrm{deg}^2$]",
    "Alfa_cubed": r"$\alpha^3$ [$\mathrm{deg}^3$]",
    "Beta_squared": r"$\beta^2$ [$\mathrm{deg}^2$]",
    "Beta_cubed": r"$\beta^3$ [$\mathrm{deg}^3$]",
    "H1_squared": r"$H_1^2$ [$\mathrm{m}^2$]",
    "H1_cubed": r"$H_1^3$ [$\mathrm{m}^3$]",
    "H2_squared": r"$H_2^2$ [$\mathrm{m}^2$]",
    "H2_cubed": r"$H_2^3$ [$\mathrm{m}^3$]",
}

# Mapping design parameters to complete axis descriptions (Q1 Journal Standard)
PARAM_AXIS_LABEL_MAP: Dict[str, str] = {
    "Alfa": r"Baffle angle $\alpha$ [deg]",
    "Beta": r"Deflector angle $\beta$ [deg]",
    "H1": r"Baffle height $H_1$ [m]",
    "H2": r"Baffle height $H_2$ [m]",
}

# Mapping parameters to reference point labels in legends
PARAM_NOMINAL_LABEL_MAP: Dict[str, str] = {
    "Alfa": r"Nominal $\alpha = 60^\circ$",
    "Beta": r"Nominal $\beta = 60^\circ$",
    "H1": r"Nominal $H_1 = 0.038\text{ m}$",
    "H2": r"Nominal $H_2 = 0.038\text{ m}$",
}


# ==============================================================================
# 2. CENTRAL PUBLICATION STYLING CONFIGURATION (rcParams)
# ==============================================================================

PUBLICATION_RC_PARAMS: Dict[str, Any] = {
    # Typography: exclusively clean sans-serif
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "Liberation Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": 9.0,
    # Axis labels and ticks
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "legend.title_fontsize": 8.5,
    # Line width and spines
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#262626",
    "axes.labelcolor": "#1A1A1A",
    "xtick.color": "#262626",
    "ytick.color": "#262626",
    "lines.linewidth": 1.4,
    "lines.markersize": 5.5,
    "patch.linewidth": 0.8,
    # Subtle background grid
    "axes.grid": True,
    "grid.alpha": 0.45,
    "grid.color": "#D0D0D0",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    # Inward ticks (engineering standard)
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.top": True,
    "ytick.right": True,
    # Legend styling
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#CCCCCC",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    # Rendering and export
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}


def set_publication_style() -> None:
    """Activates global publication styling in matplotlib.rcParams."""
    mpl.rcParams.update(PUBLICATION_RC_PARAMS)


@contextmanager
def publication_style():
    """Context manager temporarily applying publication rcParams."""
    original_params = mpl.rcParams.copy()
    mpl.rcParams.update(PUBLICATION_RC_PARAMS)
    try:
        yield
    finally:
        mpl.rcParams.update(original_params)


# ==============================================================================
# 3. EXPORT AND SUBPANEL LABELING (NO TITLES, PANEL LABELS)
# ==============================================================================

def add_panel_label(
    ax: plt.Axes,
    label: str,
    loc: str = "top_left",
    x_offset: float = 0.03,
    y_offset: float = 0.94,
    fontsize: float = 10.0,
    fontweight: str = "bold",
    bbox: bool = True,
) -> None:
    """
    Adds unified subfigure indicator (e.g. '(a)', '(b)') inside a panel.
    Replaces non-standard titles (plt.title) with Q1 journal compliant panel labels.
    """
    bbox_props = (
        dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85)
        if bbox
        else None
    )

    if loc == "top_left":
        x, y = x_offset, y_offset
        ha, va = "left", "top"
    elif loc == "outside_top_left":
        x, y = 0.0, 1.025
        ha, va = "left", "bottom"
        bbox_props = None
    elif loc == "top_right":
        x, y = 1.0 - x_offset, y_offset
        ha, va = "right", "top"
    elif loc == "bottom_left":
        x, y = x_offset, 1.0 - y_offset
        ha, va = "left", "bottom"
    else:
        x, y = x_offset, y_offset
        ha, va = "left", "top"

    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        ha=ha,
        va=va,
        bbox=bbox_props,
        zorder=20,
    )


def save_publication_figure(
    fig: plt.Figure,
    figure_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
    close_fig: bool = True,
) -> Dict[str, Path]:
    """
    Saves figure in vector formats (PDF, SVG) and high-resolution raster (PNG, 300 DPI)
    into the dedicated figures/ directory.

    Returns:
        Dictionary mapping format strings to saved file Paths.
    """
    if output_dir is None:
        target_dir = figures_dir
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    clean_name = Path(figure_name).name if isinstance(figure_name, Path) else str(figure_name)
    for ext in [".png", ".pdf", ".svg", ".jpg", ".jpeg"]:
        clean_name = clean_name.replace(ext, "")

    saved_paths = {}
    for fmt in formats:
        fmt_clean = fmt.lstrip(".").lower()
        file_path = target_dir / f"{clean_name}.{fmt_clean}"
        fig.savefig(
            file_path,
            format=fmt_clean,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.08,
        )
        saved_paths[fmt_clean] = file_path

    if close_fig:
        plt.close(fig)

    print(f"[FIGURE SAVED] {clean_name} -> {target_dir} ({', '.join(saved_paths.keys())})")
    return saved_paths


__all__ = [
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
]
