"""
Scientific Visualization module for publication-quality figures (Q1 Journal Standard).
Tailored for chemical engineering, carbon capture, and machine learning research.

Key features:
- Centralized sans-serif typography configuration (DejaVu Sans / Arial / Helvetica)
- Strict adherence to publication standards (no plt.title headers; subpanel labels (a), (b)...)
- Color Vision Deficiency (CVD) accessible palettes (Okabe-Ito, viridis, cividis)
- Dual coding with distinct line styles and markers for grayscale print accessibility
- Multi-format vector (.pdf, .svg) and high-resolution raster (.png, 300+ DPI) export
- Automated output directory resolution into figures/
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd

from config.path import figures_dir
from src.constants import PARAM_BOUNDS

# ==============================================================================
# 1. COLOR PALETTES AND CVD SYMBOLS (Color Vision Deficiency Friendly & Grayscale)
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

# Line styles and markers for black-and-white print clarity
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
# 2. CENTRAL PUBLICATION STYLING CONFIGURATION
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
    output_dir: Optional[Path] = None,
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
    clean_name = figure_name.replace(".png", "").replace(".pdf", "").replace(".svg", "")

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


# ==============================================================================
# 4. PUBLICATION FIGURE GENERATION ROUTINES
# ==============================================================================

def plot_parity_single(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    target_name: str = "$N_1$",
    target_label: str = "Particle loss",
    unit: str = "[particles/s]",
    color: str = OKABE_ITO[0],
    marker: str = "o",
    panel_label: Optional[str] = None,
    figure_name: str = "Fig_parity",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (3.9, 3.8),
) -> Dict[str, Path]:
    """
    Generates an autonomous parity correlation plot for a single target.
    Optimized for single-column publication figures and conference posters.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    r2 = r2_score(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))

    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            y_t,
            y_p,
            color=color,
            marker=marker,
            s=42,
            alpha=0.75,
            edgecolors="#222222",
            linewidth=0.6,
            label="Samples",
            zorder=4,
        )

        # Linia idealnego dopasowania y = x
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        span = max_val - min_val
        margin = 0.05 * span if span > 0 else 1.0
        line_vals = np.linspace(min_val - margin, max_val + margin, 100)
        ax.plot(
            line_vals,
            line_vals,
            color="#000000",
            linestyle="--",
            linewidth=1.2,
            label="Identity line ($y = x$)",
            zorder=3,
        )

        ax.set_xlabel(f"Actual {target_label} {target_name} {unit}")
        ax.set_ylabel(f"Predicted {target_label} {target_name} {unit}")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Ramka statystyk
        stats_str = f"$R^2 = {r2:.3f}$\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
        ax.text(
            0.05,
            0.72,
            stats_str,
            transform=ax.transAxes,
            fontsize=8.0,
            verticalalignment="top",
            bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
            zorder=10,
        )

        if panel_label:
            add_panel_label(ax, panel_label, loc="top_left")

        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout(pad=1.0)

        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_residuals_single(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    targets: Optional[List[Tuple[str, str, str]]] = None,
    panel_label: Optional[str] = None,
    figure_name: str = "Fig_residuals_distribution",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.2, 3.5),
) -> Dict[str, Path]:
    """
    Generates an autonomous residual error distribution figure.
    Contains prediction error histograms for target variables with zero error reference line.
    """
    if targets is None:
        targets = [
            ("N1", "$N_1$", OKABE_ITO[0]),
            ("N2", "$N_2$", OKABE_ITO[1]),
        ]

    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        for col, label, color in targets:
            res = y_true_df[col].values - y_pred_df[col].values
            ax.hist(
                res,
                bins=20,
                alpha=0.62,
                color=color,
                edgecolor="#222222",
                linewidth=0.6,
                label=f"Residuals {label}",
            )

        ax.axvline(0, color="#000000", linestyle="--", linewidth=1.2, label="Zero error line")

        ax.set_xlabel("Prediction error ($y_{\\mathrm{actual}} - y_{\\mathrm{predicted}}$) [particles/s]")
        ax.set_ylabel("Frequency [samples]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        if panel_label:
            add_panel_label(ax, panel_label, loc="top_left")

        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()

        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_model_diagnostics(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    targets: Optional[List[Tuple[str, str, str]]] = None,
    figure_name: str = "Fig2_model_diagnostics",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig2",
) -> Dict[str, Any]:
    """
    Generates Figure 2: Model Diagnostics and Surrogate Predictive Performance.
    By default generates BOTH a composite 2x2 grid AND standalone panel figures
    (for direct inclusion in conference posters and modular manuscript sections).

    Returns:
        Dictionary of file paths to composite and individual panel figures.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if targets is None:
        # (Nazwa kolumny, Etykieta osi, Oznaczenie jednostki)
        targets = [
            ("N1", "Particle loss $N_1$", "[particles/s]"),
            ("N2", "Captured particles $N_2$", "[particles/s]"),
            ("Delta", "Net capture advantage $\\Delta$", "[particles/s]"),
        ]

    saved_all: Dict[str, Any] = {}

    # 1. Generate standalone panel figures (ideal for posters and modular layouts)
    if save_individual:
        single_configs = [
            (targets[0][0], "$N_1$", "Particle loss", targets[0][2], OKABE_ITO[0], MARKERS[0], "(a)", f"{individual_prefix}a_parity_N1"),
            (targets[1][0], "$N_2$", "Captured particles", targets[1][2], OKABE_ITO[1], MARKERS[1], "(b)", f"{individual_prefix}b_parity_N2"),
            (targets[2][0], "$\\Delta$", "Net capture advantage", targets[2][2], OKABE_ITO[2], MARKERS[2], "(c)", f"{individual_prefix}c_parity_Delta"),
        ]

        for col, s_name, s_label, unit, color, marker, p_lbl, f_name in single_configs:
            res_single = plot_parity_single(
                y_true=y_true_df[col],
                y_pred=y_pred_df[col],
                target_name=s_name,
                target_label=s_label,
                unit=unit,
                color=color,
                marker=marker,
                panel_label=p_lbl,
                figure_name=f_name,
                output_dir=output_dir,
                formats=formats,
            )
            saved_all[f_name] = res_single

        # Standalone residual distribution figure
        res_dist = plot_residuals_single(
            y_true_df=y_true_df,
            y_pred_df=y_pred_df,
            panel_label="(d)",
            figure_name=f"{individual_prefix}d_residuals_distribution",
            output_dir=output_dir,
            formats=formats,
        )
        saved_all[f"{individual_prefix}d_residuals_distribution"] = res_dist

    # 2. Generate composite 2x2 grid
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.8))
        axes_flat = axes.flatten()

        panel_labels = ["(a)", "(b)", "(c)", "(d)"]

        # Wykresy korelacji y_true vs y_pred
        for i, (col, label_name, unit) in enumerate(targets):
            ax = axes_flat[i]
            y_t = y_true_df[col].values
            y_p = y_pred_df[col].values

            r2 = r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))

            color = OKABE_ITO[i % len(OKABE_ITO)]
            marker = MARKERS[i % len(MARKERS)]

            ax.scatter(
                y_t,
                y_p,
                color=color,
                marker=marker,
                s=32,
                alpha=0.75,
                edgecolors="#222222",
                linewidth=0.6,
                label="Samples",
                zorder=4,
            )

            # Linia idealnego dopasowania y = x
            min_val = min(y_t.min(), y_p.min())
            max_val = max(y_t.max(), y_p.max())
            span = max_val - min_val
            margin = 0.05 * span if span > 0 else 1.0
            line_vals = np.linspace(min_val - margin, max_val + margin, 100)
            ax.plot(
                line_vals,
                line_vals,
                color="#000000",
                linestyle="--",
                linewidth=1.2,
                label="Identity line ($y = x$)",
                zorder=3,
            )

            ax.set_xlabel(f"Actual {label_name} {unit}")
            ax.set_ylabel(f"Predicted {label_name} {unit}")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

            # Ramka statystyk
            stats_str = f"$R^2 = {r2:.3f}$\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
            ax.text(
                0.05,
                0.72,
                stats_str,
                transform=ax.transAxes,
                fontsize=7.8,
                verticalalignment="top",
                bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
                zorder=10,
            )

            add_panel_label(ax, panel_labels[i], loc="top_left")
            ax.legend(loc="lower right", framealpha=0.88)

        # Residual error histogram panel
        ax_res = axes_flat[3]
        res_n1 = y_true_df["N1"].values - y_pred_df["N1"].values
        res_n2 = y_true_df["N2"].values - y_pred_df["N2"].values

        ax_res.hist(
            res_n1,
            bins=18,
            alpha=0.65,
            color=OKABE_ITO[0],
            edgecolor="#222222",
            linewidth=0.6,
            label="Residuals $N_1$",
        )
        ax_res.hist(
            res_n2,
            bins=18,
            alpha=0.60,
            color=OKABE_ITO[1],
            edgecolor="#222222",
            linewidth=0.6,
            label="Residuals $N_2$",
        )
        ax_res.axvline(0, color="#000000", linestyle="--", linewidth=1.2, label="Zero error line")

        ax_res.set_xlabel("Prediction error ($y_{\\mathrm{actual}} - y_{\\mathrm{predicted}}$) [particles/s]")
        ax_res.set_ylabel("Frequency [samples]")
        ax_res.xaxis.set_minor_locator(AutoMinorLocator())
        ax_res.yaxis.set_minor_locator(AutoMinorLocator())
        add_panel_label(ax_res, panel_labels[3], loc="top_left")
        ax_res.legend(loc="upper right", framealpha=0.88)

        fig.tight_layout(pad=1.2)
        res_grid = save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )
        saved_all["combined_grid"] = res_grid

    return saved_all


def plot_feature_importance_single(
    importance_df: pd.DataFrame,
    target_name: str = "$N_1$",
    target_label: str = "Particle loss",
    feature_col: str = "feature",
    importance_col: str = "importance_pct",
    std_col: Optional[str] = "std_pct",
    base_features: Optional[Sequence[str]] = None,
    panel_label: Optional[str] = "(a)",
    figure_name: str = "Fig3a_importance_N1",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.0, 3.8),
) -> Dict[str, Path]:
    """
    Generates an autonomous feature importance figure for a single target.
    Horizontal bar chart with permutation variance, distinguishing base and engineered features
    with LaTeX mathematical formatting on the vertical axis.
    """
    if base_features is None:
        base_features = ["Alfa", "Beta", "H1", "H2"]

    # Filter features with negligible importance (< 0.1%)
    df_plot = importance_df[importance_df[importance_col] >= 0.1].copy()
    df_sorted = df_plot.sort_values(by=importance_col, ascending=True).copy()

    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        colors = [
            OKABE_ITO[0] if feat in base_features else OKABE_ITO[1]
            for feat in df_sorted[feature_col]
        ]
        hatches = [
            "" if feat in base_features else "//"
            for feat in df_sorted[feature_col]
        ]
        y_labels = [
            FEATURE_LABEL_MAP.get(f, f)
            for f in df_sorted[feature_col]
        ]

        xerr = df_sorted[std_col].values if (std_col and std_col in df_sorted.columns) else None

        bars = ax.barh(
            y_labels,
            df_sorted[importance_col],
            xerr=xerr,
            color=colors,
            edgecolor="#222222",
            linewidth=0.6,
            height=0.65,
            capsize=2.5,
            error_kw=dict(elinewidth=0.8, ecolor="#222222"),
            zorder=3,
        )

        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        # Percentage annotations next to bars
        max_val = df_sorted[importance_col].max()
        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            val = row[importance_col]
            err = row[std_col] if (std_col and std_col in row) else 0.0
            if val > 55.0:
                ax.text(
                    val - 2.5,
                    idx,
                    f"{val:.1f}%",
                    va="center",
                    ha="right",
                    fontsize=7.5,
                    color="#FFFFFF",
                    fontweight="bold",
                    zorder=5,
                )
            else:
                x_pos = val + (err if pd.notnull(err) else 0.0) + 1.2
                ax.text(
                    x_pos,
                    idx,
                    f"{val:.1f}%",
                    va="center",
                    ha="left",
                    fontsize=7.5,
                    color="#222222",
                    zorder=5,
                )

        ax.set_xlabel(f"Relative importance for {target_label} {target_name} [%]")
        ax.set_ylabel("Geometric parameter")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0, max(max_val * 1.20, max_val + 10.0) if max_val < 55 else 100)

        # Legend distinguishing base and engineered features
        custom_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=OKABE_ITO[0], edgecolor="#222222", linewidth=0.6),
            plt.Rectangle((0, 0), 1, 1, facecolor=OKABE_ITO[1], edgecolor="#222222", linewidth=0.6, hatch="//"),
        ]
        ax.legend(
            custom_handles,
            ["Base geometry", "Engineered interaction"],
            loc="lower right",
            framealpha=0.92,
        )

        if panel_label:
            add_panel_label(ax, panel_label, loc="outside_top_left")

        fig.tight_layout(pad=1.0)
        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_feature_importance_comparison(
    importance_dict: Dict[str, pd.DataFrame],
    feature_col: str = "feature",
    importance_col: str = "importance_pct",
    panel_label: Optional[str] = "(d)",
    figure_name: str = "Fig3d_importance_comparison",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.4, 3.8),
) -> Dict[str, Path]:
    """
    Generates Figure 3d: Comparative feature importance across objectives (N1 vs. N2 vs. Delta).
    Grouped horizontal bar chart demonstrating shift in dominance from H1 (losses) to Alfa (collection).
    """
    # Determine common active feature list ordered by mean importance
    all_feats = set()
    for df in importance_dict.values():
        all_feats.update(df[df[importance_col] >= 0.1][feature_col].tolist())

    feat_avg = {}
    for f in all_feats:
        vals = []
        for df in importance_dict.values():
            m = df[df[feature_col] == f]
            vals.append(m[importance_col].values[0] if len(m) > 0 else 0.0)
        feat_avg[f] = np.mean(vals)

    sorted_feats = sorted(feat_avg.keys(), key=lambda x: feat_avg[x], reverse=False)
    y_labels = [FEATURE_LABEL_MAP.get(f, f) for f in sorted_feats]

    targets_info = [
        ("N1", "Loss $N_1$", OKABE_ITO[0], ""),
        ("N2", "Capture $N_2$", OKABE_ITO[1], "//"),
        ("Delta", "Advantage $\\Delta$", OKABE_ITO[2], "\\\\"),
    ]

    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        n_feats = len(sorted_feats)
        y_indices = np.arange(n_feats)
        bar_height = 0.25

        for t_idx, (t_key, t_label, t_color, t_hatch) in enumerate(targets_info):
            if t_key not in importance_dict:
                continue
            df_t = importance_dict[t_key]
            vals = []
            for f in sorted_feats:
                m = df_t[df_t[feature_col] == f]
                vals.append(m[importance_col].values[0] if len(m) > 0 else 0.0)

            offset = (t_idx - 1) * bar_height
            bars = ax.barh(
                y_indices + offset,
                vals,
                height=bar_height,
                color=t_color,
                edgecolor="#222222",
                linewidth=0.6,
                label=t_label,
                zorder=3,
            )
            for bar in bars:
                bar.set_hatch(t_hatch)

        ax.set_yticks(y_indices)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Relative feature importance [%]")
        ax.set_ylabel("Geometric parameter")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xlim(0, 95)
        ax.legend(loc="lower right", framealpha=0.92)

        if panel_label:
            add_panel_label(ax, panel_label, loc="outside_top_left")

        fig.tight_layout(pad=1.0)
        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_feature_importance(
    importance_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    figure_name: str = "Fig3_feature_importance",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig3",
) -> Dict[str, Any]:
    """
    Generates Figure 3: Geometric Feature Importance (Explainable AI / Permutation Importance).
    By default generates BOTH a composite 2x2 grid AND standalone panel figures
    (N1, N2, Delta, and cross-target comparison).

    Returns:
        Dictionary of file paths to composite and individual panel figures.
    """
    saved_all: Dict[str, Any] = {}

    if isinstance(importance_data, pd.DataFrame):
        # Backward compatibility if a single DataFrame is passed
        importance_dict = {"N1": importance_data}
    else:
        importance_dict = importance_data

    target_configs = [
        ("N1", "$N_1$", "Particle loss", "(a)", f"{individual_prefix}a_importance_N1"),
        ("N2", "$N_2$", "Captured particles", "(b)", f"{individual_prefix}b_importance_N2"),
        ("Delta", "$\\Delta$", "Net capture advantage", "(c)", f"{individual_prefix}c_importance_Delta"),
    ]

    # 1. Generate standalone panel figures (ideal for posters and modular layouts)
    if save_individual:
        for t_key, t_sym, t_name, p_lbl, f_name in target_configs:
            if t_key in importance_dict:
                res_single = plot_feature_importance_single(
                    importance_df=importance_dict[t_key],
                    target_name=t_sym,
                    target_label=t_name,
                    panel_label=p_lbl,
                    figure_name=f_name,
                    output_dir=output_dir,
                    formats=formats,
                )
                saved_all[f_name] = res_single

        # Standalone comparison figure (Fig 3d)
        if len(importance_dict) >= 2:
            res_comp = plot_feature_importance_comparison(
                importance_dict=importance_dict,
                panel_label="(d)",
                figure_name=f"{individual_prefix}d_importance_comparison",
                output_dir=output_dir,
                formats=formats,
            )
            saved_all[f"{individual_prefix}d_importance_comparison"] = res_comp

    # 2. Generate composite 2x2 grid
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.0))
        axes_flat = axes.flatten()

        # Panele (a), (b), (c)
        base_features = ["Alfa", "Beta", "H1", "H2"]
        for i, (t_key, t_sym, t_name, p_lbl, _) in enumerate(target_configs):
            ax = axes_flat[i]
            if t_key not in importance_dict:
                continue
            df_t = importance_dict[t_key]
            df_plot = df_t[df_t["importance_pct"] >= 0.1].copy()
            df_sorted = df_plot.sort_values(by="importance_pct", ascending=True).copy()

            colors = [
                OKABE_ITO[0] if feat in base_features else OKABE_ITO[1]
                for feat in df_sorted["feature"]
            ]
            hatches = [
                "" if feat in base_features else "//"
                for feat in df_sorted["feature"]
            ]
            y_labels = [FEATURE_LABEL_MAP.get(f, f) for f in df_sorted["feature"]]
            xerr = df_sorted["std_pct"].values if "std_pct" in df_sorted.columns else None

            bars = ax.barh(
                y_labels,
                df_sorted["importance_pct"],
                xerr=xerr,
                color=colors,
                edgecolor="#222222",
                linewidth=0.6,
                height=0.65,
                capsize=2.2,
                error_kw=dict(elinewidth=0.8, ecolor="#222222"),
                zorder=3,
            )
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

            max_val = df_sorted["importance_pct"].max()
            for idx, (_, row) in enumerate(df_sorted.iterrows()):
                val = row["importance_pct"]
                err = row["std_pct"] if "std_pct" in row else 0.0
                if val > 55.0:
                    ax.text(
                        val - 2.5,
                        idx,
                        f"{val:.1f}%",
                        va="center",
                        ha="right",
                        fontsize=7.2,
                        color="#FFFFFF",
                        fontweight="bold",
                        zorder=5,
                    )
                else:
                    x_pos = val + (err if pd.notnull(err) else 0.0) + 1.2
                    ax.text(
                        x_pos,
                        idx,
                        f"{val:.1f}%",
                        va="center",
                        ha="left",
                        fontsize=7.2,
                        color="#222222",
                        zorder=5,
                    )

            ax.set_xlabel(f"Relative importance {t_name} {t_sym} [%]")
            ax.set_ylabel("Geometric parameter")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.set_xlim(0, max(max_val * 1.20, max_val + 10.0) if max_val < 55 else 100)

            custom_handles = [
                plt.Rectangle((0, 0), 1, 1, facecolor=OKABE_ITO[0], edgecolor="#222222", linewidth=0.6),
                plt.Rectangle((0, 0), 1, 1, facecolor=OKABE_ITO[1], edgecolor="#222222", linewidth=0.6, hatch="//"),
            ]
            ax.legend(
                custom_handles,
                ["Base geometry", "Engineered interaction"],
                loc="lower right",
                framealpha=0.92,
            )
            add_panel_label(ax, p_lbl, loc="outside_top_left")

        # Panel (d): Cross-target comparison
        ax_d = axes_flat[3]
        all_feats = set()
        for df in importance_dict.values():
            all_feats.update(df[df["importance_pct"] >= 0.1]["feature"].tolist())

        feat_avg = {}
        for f in all_feats:
            vals = []
            for df in importance_dict.values():
                m = df[df["feature"] == f]
                vals.append(m["importance_pct"].values[0] if len(m) > 0 else 0.0)
            feat_avg[f] = np.mean(vals)

        sorted_feats = sorted(feat_avg.keys(), key=lambda x: feat_avg[x], reverse=False)
        y_labels = [FEATURE_LABEL_MAP.get(f, f) for f in sorted_feats]

        targets_info = [
            ("N1", "Loss $N_1$", OKABE_ITO[0], ""),
            ("N2", "Capture $N_2$", OKABE_ITO[1], "//"),
            ("Delta", "Advantage $\\Delta$", OKABE_ITO[2], "\\\\"),
        ]

        n_feats = len(sorted_feats)
        y_indices = np.arange(n_feats)
        bar_height = 0.25

        for t_idx, (t_key, t_label, t_color, t_hatch) in enumerate(targets_info):
            if t_key not in importance_dict:
                continue
            df_t = importance_dict[t_key]
            vals = []
            for f in sorted_feats:
                m = df_t[df_t["feature"] == f]
                vals.append(m["importance_pct"].values[0] if len(m) > 0 else 0.0)

            offset = (t_idx - 1) * bar_height
            bars = ax_d.barh(
                y_indices + offset,
                vals,
                height=bar_height,
                color=t_color,
                edgecolor="#222222",
                linewidth=0.6,
                label=t_label,
                zorder=3,
            )
            for bar in bars:
                bar.set_hatch(t_hatch)

        ax_d.set_yticks(y_indices)
        ax_d.set_yticklabels(y_labels)
        ax_d.set_xlabel("Relative feature importance [%]")
        ax_d.set_ylabel("Geometric parameter")
        ax_d.xaxis.set_minor_locator(AutoMinorLocator())
        ax_d.set_xlim(0, 95)
        ax_d.legend(loc="lower right", framealpha=0.92)
        add_panel_label(ax_d, "(d)", loc="outside_top_left")

        fig.tight_layout(pad=1.4)
        res_grid = save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )
        saved_all["combined_grid"] = res_grid

    return saved_all


def plot_sensitivity_sweep_single(
    df_param: pd.DataFrame,
    param: str,
    ref_val: Optional[float] = None,
    panel_label: Optional[str] = "(a)",
    figure_name: str = "Fig4a_sweep_Alfa",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.0, 3.5),
) -> Dict[str, Path]:
    """
    Generates an autonomous 1D dual-response profile (Dual-Y axis).
    Left axis: N1 (particle loss), Right axis: N2 (carrier capture).
    Includes CFD nominal reference line and LaTeX annotations.
    """
    with publication_style():
        fig, ax1 = plt.subplots(figsize=figsize)
        color_n1 = OKABE_ITO[0]
        color_n2 = OKABE_ITO[1]

        # Left axis: N1 (loss)
        line1 = ax1.plot(
            df_param[param],
            df_param["N1_pred"],
            color=color_n1,
            linestyle=LINE_STYLES[0],
            linewidth=1.8,
            label="Particle loss $N_1$",
            zorder=4,
        )
        xlabel = PARAM_AXIS_LABEL_MAP.get(param, f"Parameter {param}")
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("Particle loss $N_1$ [particles/s]", color=color_n1)
        ax1.tick_params(axis="y", labelcolor=color_n1)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

        lines = list(line1)
        # Linia referencyjna
        if ref_val is not None:
            ref_lbl = PARAM_NOMINAL_LABEL_MAP.get(
                param, f"Nominal {param} ({ref_val:.3g})"
            )
            line_ref = ax1.axvline(
                ref_val,
                color="#333333",
                linestyle=":",
                linewidth=1.2,
                label=ref_lbl,
                zorder=3,
            )
            lines.append(line_ref)

        # Right axis: N2 (capture)
        ax2 = ax1.twinx()
        line2 = ax2.plot(
            df_param[param],
            df_param["N2_pred"],
            color=color_n2,
            linestyle=LINE_STYLES[1],
            linewidth=1.8,
            label="Captured particles $N_2$",
            zorder=4,
        )
        ax2.set_ylabel("Captured particles $N_2$ [particles/s]", color=color_n2)
        ax2.tick_params(axis="y", labelcolor=color_n2)
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid(False)

        # Horizontal axis margin for clean reference line appearance
        span = df_param[param].max() - df_param[param].min()
        margin = 0.025 * span if span > 0 else 0.5
        ax1.set_xlim(df_param[param].min() - margin, df_param[param].max() + margin)

        lines.extend(line2)
        labels = [l.get_label() for l in lines]
        # Adaptive legend placement for optimal readability
        legend_loc_map = {
            "Alfa": "center left",
            "Beta": "center left",
            "H1": "upper left",
            "H2": "lower left",
        }
        legend_loc = legend_loc_map.get(param, "best")
        ax1.legend(lines, labels, loc=legend_loc, framealpha=0.92)

        if panel_label:
            add_panel_label(ax1, panel_label, loc="outside_top_left")

        fig.tight_layout(pad=0.8)
        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_sensitivity_sweeps_1d(
    sensitivity_data: Dict[str, pd.DataFrame],
    ref_params: Dict[str, float],
    units: Optional[Dict[str, str]] = None,
    figure_name: str = "Fig4_sensitivity_sweeps_1d",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig4",
) -> Dict[str, Any]:
    """
    Generates Figure 4: 1D Engineering Sensitivity Sweeps around Reference Point.
    By default generates BOTH a composite 2x2 grid AND standalone figures
    (Alfa, Beta, H1, H2) z dwiema osiami pionowymi (Dual-Y).
    """
    saved_all: Dict[str, Any] = {}
    params = list(sensitivity_data.keys())
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    # 1. Generowanie osobnych figur
    if save_individual:
        for i, param in enumerate(params):
            p_lbl = panel_labels[i] if i < len(panel_labels) else f"({chr(97+i)})"
            f_letter = chr(97 + i)  # 'a', 'b', 'c', 'd'
            f_name = f"{individual_prefix}{f_letter}_sweep_{param}"
            df_p = sensitivity_data[param]
            ref_val = ref_params.get(param, None)

            res_single = plot_sensitivity_sweep_single(
                df_param=df_p,
                param=param,
                ref_val=ref_val,
                panel_label=p_lbl,
                figure_name=f_name,
                output_dir=output_dir,
                formats=formats,
            )
            saved_all[f_name] = res_single

    # 2. Generate composite 2x2 grid
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
        axes_flat = axes.flatten()

        for i, param in enumerate(params):
            ax1 = axes_flat[i]
            df_p = sensitivity_data[param]

            color_n1 = OKABE_ITO[0]
            color_n2 = OKABE_ITO[1]

            # Left axis: N1 (loss)
            line1 = ax1.plot(
                df_p[param],
                df_p["N1_pred"],
                color=color_n1,
                linestyle=LINE_STYLES[0],
                linewidth=1.6,
                label="Particle loss $N_1$",
                zorder=4,
            )
            xlabel = PARAM_AXIS_LABEL_MAP.get(param, f"Parameter {param}")
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel("Particle loss $N_1$ [particles/s]", color=color_n1)
            ax1.tick_params(axis="y", labelcolor=color_n1)
            ax1.xaxis.set_minor_locator(AutoMinorLocator())
            ax1.yaxis.set_minor_locator(AutoMinorLocator())

            lines = list(line1)
            # Linia referencyjna
            if param in ref_params:
                ref_val = ref_params[param]
                ref_lbl = PARAM_NOMINAL_LABEL_MAP.get(
                    param, f"Nominal {param} ({ref_val:.3g})"
                )
                line_ref = ax1.axvline(
                    ref_val,
                    color="#333333",
                    linestyle=":",
                    linewidth=1.1,
                    label=ref_lbl,
                    zorder=3,
                )
                lines.append(line_ref)

            # Right axis: N2 (capture)
            ax2 = ax1.twinx()
            line2 = ax2.plot(
                df_p[param],
                df_p["N2_pred"],
                color=color_n2,
                linestyle=LINE_STYLES[1],
                linewidth=1.6,
                label="Captured particles $N_2$",
                zorder=4,
            )
            ax2.set_ylabel("Captured particles $N_2$ [particles/s]", color=color_n2)
            ax2.tick_params(axis="y", labelcolor=color_n2)
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.grid(False)

            # Margines osi poziomej
            span = df_p[param].max() - df_p[param].min()
            margin = 0.025 * span if span > 0 else 0.5
            ax1.set_xlim(df_p[param].min() - margin, df_p[param].max() + margin)

            lines.extend(line2)
            labels = [l.get_label() for l in lines]
            legend_loc_map = {
                "Alfa": "center left",
                "Beta": "center left",
                "H1": "upper left",
                "H2": "lower left",
            }
            legend_loc = legend_loc_map.get(param, "best")
            ax1.legend(lines, labels, loc=legend_loc, framealpha=0.90)

            add_panel_label(ax1, panel_labels[i], loc="outside_top_left")

        fig.tight_layout(pad=1.4)
        res_grid = save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )
        saved_all["combined_grid"] = res_grid

    return saved_all


def plot_sensitivity_heatmap_single(
    G1: np.ndarray,
    G2: np.ndarray,
    Z: np.ndarray,
    p1: str,
    p2: str,
    target: str = "N1",
    ref_val1: Optional[float] = None,
    ref_val2: Optional[float] = None,
    panel_label: Optional[str] = "(a)",
    figure_name: str = "Fig4e_contour_Alfa_Beta_N1",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.3, 3.8),
    contour_levels: int = 16,
) -> Dict[str, Path]:
    """
    Generates an autonomous 2D interaction contour map (heatmap) for selected parameter pair.
    For N1 (loss): colormap viridis_r (yellow = minimal loss / optimal).
    For N2 (capture): colormap viridis (yellow = maximal capture / optimal).
    Czysta powierzchnia odpowiedzi bez linii izoliniowych i bez markera punktu CFD (wg wytycznych).
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        if target == "N1":
            cmap = "viridis_r"
            cbar_label = "Particle loss $N_1$ [particles/s]"
        else:
            cmap = "viridis"
            cbar_label = "Captured particles $N_2$ [particles/s]"

        cf = ax.contourf(G1, G2, Z, levels=contour_levels, cmap=cmap)

        cbar = fig.colorbar(cf, ax=ax, pad=0.03, aspect=22)
        cbar.set_label(cbar_label, fontsize=8.5)
        cbar.ax.tick_params(labelsize=8.0)
        cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.set_xlabel(PARAM_AXIS_LABEL_MAP.get(p1, p1))
        ax.set_ylabel(PARAM_AXIS_LABEL_MAP.get(p2, p2))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Czyste granice osi bez marginesu
        ax.set_xlim(G1.min(), G1.max())
        ax.set_ylim(G2.min(), G2.max())

        if panel_label:
            add_panel_label(ax, panel_label, loc="outside_top_left")

        fig.tight_layout(pad=0.8)
        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_sensitivity_heatmaps_2d(
    predict_fn: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ref_params: Dict[str, float],
    param_pairs: Optional[List[Tuple[str, str, Tuple[float, float], Tuple[float, float]]]] = None,
    units: Optional[Dict[str, str]] = None,
    grid_size: int = 40,
    figure_name: str = "Fig4_sensitivity_heatmaps_2d",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig4",
) -> Dict[str, Any]:
    """
    Generuje Fig 4b: Mapy konturowe 2D interakcji geometrycznych dla kluczowych par.
    By default generates BOTH composite 2x2 grid AND standalone figures
    (Alfa-Beta N1/N2 oraz H1-H2 N1/N2).
    """
    saved_all: Dict[str, Any] = {}

    if param_pairs is None:
        param_pairs = [
            ("Alfa", "Beta", PARAM_BOUNDS["Alfa"], PARAM_BOUNDS["Beta"]),
            ("H1", "H2", PARAM_BOUNDS["H1"], PARAM_BOUNDS["H2"]),
        ]

    panel_letters = [("e", "f"), ("g", "h")]
    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"]]

    grids = []
    for row_idx, (p1, p2, r1, r2) in enumerate(param_pairs):
        v1 = np.linspace(r1[0], r1[1], grid_size)
        v2 = np.linspace(r2[0], r2[1], grid_size)
        G1, G2 = np.meshgrid(v1, v2)

        rows = []
        for val1, val2 in zip(G1.ravel(), G2.ravel()):
            item = ref_params.copy()
            item[p1] = float(val1)
            item[p2] = float(val2)
            rows.append(item)

        df_grid = pd.DataFrame(rows)
        n1, n2, _ = predict_fn(df_grid)
        N1_grid = n1.reshape(grid_size, grid_size)
        N2_grid = n2.reshape(grid_size, grid_size)

        grids.append((p1, p2, G1, G2, N1_grid, N2_grid))

    # 1. Zapis osobnych figur
    if save_individual:
        for row_idx, (p1, p2, G1, G2, N1_grid, N2_grid) in enumerate(grids):
            let_n1, let_n2 = panel_letters[row_idx]
            ref_val1 = ref_params.get(p1, None)
            ref_val2 = ref_params.get(p2, None)

            # N1 single
            fname_n1 = f"{individual_prefix}{let_n1}_contour_{p1}_{p2}_N1"
            res_n1 = plot_sensitivity_heatmap_single(
                G1=G1,
                G2=G2,
                Z=N1_grid,
                p1=p1,
                p2=p2,
                target="N1",
                ref_val1=ref_val1,
                ref_val2=ref_val2,
                panel_label=panel_labels[row_idx][0],
                figure_name=fname_n1,
                output_dir=output_dir,
                formats=formats,
            )
            saved_all[fname_n1] = res_n1

            # N2 single
            fname_n2 = f"{individual_prefix}{let_n2}_contour_{p1}_{p2}_N2"
            res_n2 = plot_sensitivity_heatmap_single(
                G1=G1,
                G2=G2,
                Z=N2_grid,
                p1=p1,
                p2=p2,
                target="N2",
                ref_val1=ref_val1,
                ref_val2=ref_val2,
                panel_label=panel_labels[row_idx][1],
                figure_name=fname_n2,
                output_dir=output_dir,
                formats=formats,
            )
            saved_all[fname_n2] = res_n2

    # 2. Save composite 2x2 grid
    with publication_style():
        fig, axes = plt.subplots(len(param_pairs), 2, figsize=(7.6, 3.6 * len(param_pairs)))

        for row_idx, (p1, p2, G1, G2, N1_grid, N2_grid) in enumerate(grids):
            ref_val1 = ref_params.get(p1, None)
            ref_val2 = ref_params.get(p2, None)

            # Lewy wykres: N1
            ax_n1 = axes[row_idx, 0] if len(param_pairs) > 1 else axes[0]
            cf1 = ax_n1.contourf(G1, G2, N1_grid, levels=16, cmap="viridis_r")

            cbar1 = fig.colorbar(cf1, ax=ax_n1, pad=0.02, aspect=20)
            cbar1.set_label("Particle loss $N_1$ [particles/s]", fontsize=8.0)
            cbar1.ax.tick_params(labelsize=7.5)
            cbar1.ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax_n1.set_xlabel(PARAM_AXIS_LABEL_MAP.get(p1, p1))
            ax_n1.set_ylabel(PARAM_AXIS_LABEL_MAP.get(p2, p2))
            ax_n1.xaxis.set_minor_locator(AutoMinorLocator())
            ax_n1.yaxis.set_minor_locator(AutoMinorLocator())

            ax_n1.set_xlim(G1.min(), G1.max())
            ax_n1.set_ylim(G2.min(), G2.max())
            add_panel_label(ax_n1, panel_labels[row_idx][0], loc="outside_top_left")

            # Prawy wykres: N2
            ax_n2 = axes[row_idx, 1] if len(param_pairs) > 1 else axes[1]
            cf2 = ax_n2.contourf(G1, G2, N2_grid, levels=16, cmap="viridis")

            cbar2 = fig.colorbar(cf2, ax=ax_n2, pad=0.02, aspect=20)
            cbar2.set_label("Captured particles $N_2$ [particles/s]", fontsize=8.0)
            cbar2.ax.tick_params(labelsize=7.5)
            cbar2.ax.yaxis.set_minor_locator(AutoMinorLocator())

            ax_n2.set_xlabel(PARAM_AXIS_LABEL_MAP.get(p1, p1))
            ax_n2.set_ylabel(PARAM_AXIS_LABEL_MAP.get(p2, p2))
            ax_n2.xaxis.set_minor_locator(AutoMinorLocator())
            ax_n2.yaxis.set_minor_locator(AutoMinorLocator())

            ax_n2.set_xlim(G1.min(), G1.max())
            ax_n2.set_ylim(G2.min(), G2.max())
            add_panel_label(ax_n2, panel_labels[row_idx][1], loc="outside_top_left")

        fig.tight_layout(pad=1.4)
        res_grid = save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )
        saved_all["combined_grid"] = res_grid

    return saved_all


def plot_convergence(
    log_df: pd.DataFrame,
    panel_labels: Tuple[str, str] = ("(a)", "(b)"),
    figure_name: str = "Fig5a_nsga2_convergence",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.5, 4.8),
) -> Dict[str, Path]:
    """
    Generates Figure 5a: Multi-Objective NSGA-II Convergence across Generations.
    Dual panels: N1 minimization (top) and Delta maximization (bottom).
    """
    if "n1_min" in log_df.columns:
        generations = log_df["gen"].values if "gen" in log_df.columns else log_df.index.values
        min_n1 = log_df["n1_min"].values
        avg_n1 = log_df["n1_avg"].values
        max_n1 = log_df["n1_max"].values
        min_d = log_df["delta_min"].values
        avg_d = log_df["delta_avg"].values
        max_d = log_df["delta_max"].values
    else:
        generations = log_df.index.values
        avg_n1 = log_df["avg"].apply(lambda x: x[0]).values
        min_n1 = log_df["min"].apply(lambda x: x[0]).values
        max_n1 = log_df["max"].apply(lambda x: x[0]).values
        avg_d = log_df["avg"].apply(lambda x: x[1]).values
        min_d = log_df["min"].apply(lambda x: x[1]).values
        max_d = log_df["max"].apply(lambda x: x[1]).values

    with publication_style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Panel (a): N1 (minimalizacja)
        ax1.plot(generations, min_n1, color=OKABE_ITO[2], linestyle="-", linewidth=1.5, label="Population minimum")
        ax1.plot(generations, avg_n1, color=OKABE_ITO[0], linestyle="--", linewidth=1.5, label="Population mean")
        ax1.plot(generations, max_n1, color=OKABE_ITO[1], linestyle=":", linewidth=1.5, label="Population maximum")
        ax1.fill_between(generations, min_n1, max_n1, color=OKABE_ITO[0], alpha=0.12)
        ax1.set_ylabel("Loss objective $N_1$ [particles/s]")
        ax1.set_ylim(-15, 525)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.legend(loc="upper right", framealpha=0.90)
        if panel_labels and len(panel_labels) > 0 and panel_labels[0]:
            add_panel_label(ax1, panel_labels[0], loc="outside_top_left")

        # Panel (b): Delta (maksymalizacja)
        ax2.plot(generations, max_d, color=OKABE_ITO[2], linestyle="-", linewidth=1.5, label="Population maximum")
        ax2.plot(generations, avg_d, color=OKABE_ITO[0], linestyle="--", linewidth=1.5, label="Population mean")
        ax2.plot(generations, min_d, color=OKABE_ITO[1], linestyle=":", linewidth=1.5, label="Population minimum")
        ax2.fill_between(generations, min_d, max_d, color=OKABE_ITO[0], alpha=0.12)
        ax2.set_xlabel("Evolution generation $g$ [-]")
        ax2.set_ylabel("Advantage objective $\\Delta$ [particles/s]")
        ax2.set_ylim(3300, 6050)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.legend(loc="lower right", framealpha=0.90)
        if panel_labels and len(panel_labels) > 1 and panel_labels[1]:
            add_panel_label(ax2, panel_labels[1], loc="outside_top_left")

        fig.tight_layout(pad=0.8)
        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_pareto_front(
    all_population_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    baseline_point: Optional[Dict[str, float]] = None,
    cluster_col: Optional[str] = "cluster",
    cluster_labels_map: Optional[Dict[int, str]] = None,
    panel_label: Optional[str] = "(b)",
    figure_name: str = "Fig5b_pareto_front",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (5.2, 4.4),
    show_inset: bool = True,
    inset_xlim: Tuple[float, float] = (-0.05, 1.0),
    inset_ylim: Tuple[float, float] = (5840.0, 5866.0),
) -> Dict[str, Path]:
    """
    Generates Figure 5b: Objective Space and Pareto Optimal Front.
    X axis: Particle loss N1 [particles/s], Y axis: Net capture advantage Delta [particles/s].
    Highlights design space exploration, Pareto front, K-Means clusters, and baseline CFD.
    Includes inset zoom detailing micro-scale cluster separation.
    """
    x_col = "N1_pred" if "N1_pred" in pareto_df.columns else "N1"
    y_col = "Delta_pred" if "Delta_pred" in pareto_df.columns else "Delta"
    x_pop_col = "N1_pred" if "N1_pred" in all_population_df.columns else "N1"
    y_pop_col = "Delta_pred" if "Delta_pred" in all_population_df.columns else "Delta"

    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)

        # 1. Explored design space points
        ax.scatter(
            all_population_df[x_pop_col],
            all_population_df[y_pop_col],
            color="#A8A8A8",
            s=16,
            alpha=0.35,
            edgecolors="none",
            label="Explored candidates",
            zorder=2,
        )

        # 2. Non-dominated Pareto front solutions
        if cluster_col and cluster_col in pareto_df.columns:
            clusters = np.sort(pareto_df[cluster_col].unique())
            cluster_colors = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[5]]
            for c_idx, cluster_id in enumerate(clusters):
                sub_df = pareto_df[pareto_df[cluster_col] == cluster_id]
                c_color = cluster_colors[c_idx % len(cluster_colors)]
                c_marker = MARKERS[c_idx % len(MARKERS)]
                if cluster_labels_map and cluster_id in cluster_labels_map:
                    c_lbl = cluster_labels_map[cluster_id]
                else:
                    c_lbl = f"Pareto cluster {cluster_id + 1}"
                ax.scatter(
                    sub_df[x_col],
                    sub_df[y_col],
                    color=c_color,
                    marker=c_marker,
                    s=46,
                    edgecolors="#111111",
                    linewidth=0.7,
                    label=c_lbl,
                    zorder=4,
                )
        else:
            ax.scatter(
                pareto_df[x_col],
                pareto_df[y_col],
                color=OKABE_ITO[0],
                marker="o",
                s=40,
                edgecolors="#111111",
                linewidth=0.7,
                label="Non-dominated Pareto front",
                zorder=4,
            )

        # Connect Pareto front points with trade-off spline
        pareto_sorted = pareto_df.sort_values(by=x_col)
        ax.plot(
            pareto_sorted[x_col],
            pareto_sorted[y_col],
            color="#222222",
            linestyle="--",
            linewidth=1.1,
            alpha=0.80,
            zorder=3,
        )

        # 3. Punkt bazowy (Baseline CFD)
        if baseline_point is not None:
            ax.scatter(
                baseline_point["N1"],
                baseline_point["Delta"],
                color="#D55E00",
                marker="*",
                s=160,
                edgecolors="#000000",
                linewidth=1.1,
                label="Baseline CFD geometry",
                zorder=6,
            )

        ax.set_xlabel(r"Particle loss $N_1$ [particles/s] (Minimization $\rightarrow$)")
        ax.set_ylabel(r"Capture advantage $\Delta$ [particles/s] (Maximization $\rightarrow$)")
        ax.set_xlim(-15, 525)
        ax.set_ylim(3400, 6100)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc="lower left", framealpha=0.92)

        if panel_label:
            add_panel_label(ax, panel_label, loc="outside_top_left")

        fig.tight_layout(pad=0.8)

        # 4. Inset zoom view highlighting separation of clusters 1 and 2
        if show_inset:
            ax_ins = inset_axes(
                ax,
                width="45%",
                height="38%",
                loc="upper right",
                bbox_to_anchor=(0.0, -0.02, 0.98, 0.98),
                bbox_transform=ax.transAxes,
                borderpad=1.0,
            )
            ax_ins.set_facecolor("#FFFFFF")

            if cluster_col and cluster_col in pareto_df.columns:
                for c_idx, cluster_id in enumerate(clusters):
                    sub_df = pareto_df[pareto_df[cluster_col] == cluster_id]
                    c_color = cluster_colors[c_idx % len(cluster_colors)]
                    c_marker = MARKERS[c_idx % len(MARKERS)]
                    ax_ins.scatter(
                        sub_df[x_col],
                        sub_df[y_col],
                        color=c_color,
                        marker=c_marker,
                        s=38,
                        edgecolors="#111111",
                        linewidth=0.6,
                        zorder=4,
                    )
            else:
                ax_ins.scatter(
                    pareto_df[x_col],
                    pareto_df[y_col],
                    color=OKABE_ITO[0],
                    marker="o",
                    s=36,
                    edgecolors="#111111",
                    linewidth=0.6,
                    zorder=4,
                )

            ax_ins.plot(
                pareto_sorted[x_col],
                pareto_sorted[y_col],
                color="#222222",
                linestyle="--",
                linewidth=0.9,
                alpha=0.75,
                zorder=3,
            )

            ax_ins.set_xlim(inset_xlim)
            ax_ins.set_ylim(inset_ylim)
            ax_ins.tick_params(labelsize=6.8)
            ax_ins.set_xlabel(
                r"$N_1$ [particles/s]",
                fontsize=7.2,
                labelpad=2,
                bbox=dict(boxstyle="square,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.92),
            )
            ax_ins.set_ylabel(
                r"$\Delta$ [particles/s]",
                fontsize=7.2,
                labelpad=3.2,
                bbox=dict(boxstyle="square,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.92),
            )
            ax_ins.xaxis.set_minor_locator(AutoMinorLocator())
            ax_ins.yaxis.set_minor_locator(AutoMinorLocator())

            mark_inset(ax, ax_ins, loc1=2, loc2=3, fc="none", ec="#666666", ls=":", lw=0.9, alpha=0.8)

        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_optimization_figure_5(
    log_df: pd.DataFrame,
    all_population_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    baseline_point: Optional[Dict[str, float]] = None,
    cluster_col: Optional[str] = "cluster",
    cluster_labels_map: Optional[Dict[int, str]] = None,
    figure_name: str = "Fig5_optimization_pareto",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig5",
    show_inset: bool = True,
) -> Dict[str, Any]:
    """
    Generuje komplet publikacyjny Figury 5:
    Panel (a): NSGA-II generational convergence history (N1 and Delta)
    Panel (b): Multi-objective trade-off space and Pareto front with CFD baseline and cluster inset.
    By default generates BOTH two-column composite AND standalone panels.
    """
    saved_all: Dict[str, Any] = {}

    if save_individual:
        fname_a = f"{individual_prefix}a_nsga2_convergence"
        res_a = plot_convergence(
            log_df=log_df,
            panel_labels=("(a)", "(b)"),
            figure_name=fname_a,
            output_dir=output_dir,
            formats=formats,
        )
        saved_all[fname_a] = res_a

        fname_b = f"{individual_prefix}b_pareto_front"
        res_b = plot_pareto_front(
            all_population_df=all_population_df,
            pareto_df=pareto_df,
            baseline_point=baseline_point,
            cluster_col=cluster_col,
            cluster_labels_map=cluster_labels_map,
            panel_label="(b)",
            figure_name=fname_b,
            output_dir=output_dir,
            formats=formats,
            show_inset=show_inset,
        )
        saved_all[fname_b] = res_b

    # Two-column composite: left column convergence, right column Pareto front
    with publication_style():
        fig = plt.figure(figsize=(7.6, 4.0))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.25, wspace=0.32)

        # 1. Generational convergence (left panel)
        ax_conv1 = fig.add_subplot(gs[0, 0])
        ax_conv2 = fig.add_subplot(gs[1, 0], sharex=ax_conv1)

        if "n1_min" in log_df.columns:
            generations = log_df["gen"].values if "gen" in log_df.columns else log_df.index.values
            min_n1 = log_df["n1_min"].values
            avg_n1 = log_df["n1_avg"].values
            max_n1 = log_df["n1_max"].values
            min_d = log_df["delta_min"].values
            avg_d = log_df["delta_avg"].values
            max_d = log_df["delta_max"].values
        else:
            generations = log_df.index.values
            avg_n1 = log_df["avg"].apply(lambda x: x[0]).values
            min_n1 = log_df["min"].apply(lambda x: x[0]).values
            max_n1 = log_df["max"].apply(lambda x: x[0]).values
            avg_d = log_df["avg"].apply(lambda x: x[1]).values
            min_d = log_df["min"].apply(lambda x: x[1]).values
            max_d = log_df["max"].apply(lambda x: x[1]).values

        ax_conv1.plot(generations, min_n1, color=OKABE_ITO[2], linestyle="-", linewidth=1.3, label="Min")
        ax_conv1.plot(generations, avg_n1, color=OKABE_ITO[0], linestyle="--", linewidth=1.3, label="Mean")
        ax_conv1.plot(generations, max_n1, color=OKABE_ITO[1], linestyle=":", linewidth=1.3, label="Max")
        ax_conv1.fill_between(generations, min_n1, max_n1, color=OKABE_ITO[0], alpha=0.12)
        ax_conv1.set_ylabel("Loss $N_1$ [particles/s]", fontsize=8.2)
        ax_conv1.set_ylim(-15, 525)
        ax_conv1.xaxis.set_minor_locator(AutoMinorLocator())
        ax_conv1.yaxis.set_minor_locator(AutoMinorLocator())
        ax_conv1.legend(loc="upper right", fontsize=7.2, framealpha=0.88)
        add_panel_label(ax_conv1, "(a)", loc="outside_top_left")

        ax_conv2.plot(generations, max_d, color=OKABE_ITO[2], linestyle="-", linewidth=1.3, label="Max")
        ax_conv2.plot(generations, avg_d, color=OKABE_ITO[0], linestyle="--", linewidth=1.3, label="Mean")
        ax_conv2.plot(generations, min_d, color=OKABE_ITO[1], linestyle=":", linewidth=1.3, label="Min")
        ax_conv2.fill_between(generations, min_d, max_d, color=OKABE_ITO[0], alpha=0.12)
        ax_conv2.set_xlabel("Evolution generation $g$ [-]", fontsize=8.2)
        ax_conv2.set_ylabel(r"Advantage $\Delta$ [particles/s]", fontsize=8.2)
        ax_conv2.set_ylim(3300, 6050)
        ax_conv2.xaxis.set_minor_locator(AutoMinorLocator())
        ax_conv2.yaxis.set_minor_locator(AutoMinorLocator())
        ax_conv2.legend(loc="lower right", fontsize=7.2, framealpha=0.88)

        # 2. Front Pareto (prawa strona)
        ax_pareto = fig.add_subplot(gs[:, 1])

        x_col = "N1_pred" if "N1_pred" in pareto_df.columns else "N1"
        y_col = "Delta_pred" if "Delta_pred" in pareto_df.columns else "Delta"
        x_pop_col = "N1_pred" if "N1_pred" in all_population_df.columns else "N1"
        y_pop_col = "Delta_pred" if "Delta_pred" in all_population_df.columns else "Delta"

        ax_pareto.scatter(
            all_population_df[x_pop_col],
            all_population_df[y_pop_col],
            color="#A8A8A8",
            s=14,
            alpha=0.35,
            edgecolors="none",
            label="Explored candidates",
            zorder=2,
        )

        if cluster_col and cluster_col in pareto_df.columns:
            clusters = np.sort(pareto_df[cluster_col].unique())
            cluster_colors = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[5]]
            for c_idx, cluster_id in enumerate(clusters):
                sub_df = pareto_df[pareto_df[cluster_col] == cluster_id]
                c_color = cluster_colors[c_idx % len(cluster_colors)]
                c_marker = MARKERS[c_idx % len(MARKERS)]
                if cluster_labels_map and cluster_id in cluster_labels_map:
                    c_lbl = cluster_labels_map[cluster_id]
                else:
                    c_lbl = f"Pareto cluster {cluster_id + 1}"
                ax_pareto.scatter(
                    sub_df[x_col],
                    sub_df[y_col],
                    color=c_color,
                    marker=c_marker,
                    s=44,
                    edgecolors="#111111",
                    linewidth=0.7,
                    label=c_lbl,
                    zorder=4,
                )
        else:
            ax_pareto.scatter(
                pareto_df[x_col],
                pareto_df[y_col],
                color=OKABE_ITO[0],
                marker="o",
                s=38,
                edgecolors="#111111",
                linewidth=0.7,
                label="Non-dominated Pareto front",
                zorder=4,
            )

        pareto_sorted = pareto_df.sort_values(by=x_col)
        ax_pareto.plot(
            pareto_sorted[x_col],
            pareto_sorted[y_col],
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            zorder=3,
        )

        if baseline_point is not None:
            ax_pareto.scatter(
                baseline_point["N1"],
                baseline_point["Delta"],
                color="#D55E00",
                marker="*",
                s=150,
                edgecolors="#000000",
                linewidth=1.1,
                label="Baseline CFD geometry",
                zorder=6,
            )

        ax_pareto.set_xlabel(r"Particle loss $N_1$ [particles/s] (Minimization $\rightarrow$)", fontsize=8.2)
        ax_pareto.set_ylabel(r"Capture advantage $\Delta$ [particles/s] (Maximization $\rightarrow$)", fontsize=8.2)
        ax_pareto.set_xlim(-15, 525)
        ax_pareto.set_ylim(3400, 6100)
        ax_pareto.xaxis.set_minor_locator(AutoMinorLocator())
        ax_pareto.yaxis.set_minor_locator(AutoMinorLocator())
        ax_pareto.legend(loc="lower left", fontsize=7.5, framealpha=0.90)

        # Inset zoom in composite panel (b)
        if show_inset:
            ax_ins = inset_axes(
                ax_pareto,
                width="44%",
                height="37%",
                loc="upper right",
                bbox_to_anchor=(0.0, -0.02, 0.98, 0.98),
                bbox_transform=ax_pareto.transAxes,
                borderpad=1.0,
            )
            ax_ins.set_facecolor("#FFFFFF")

            if cluster_col and cluster_col in pareto_df.columns:
                for c_idx, cluster_id in enumerate(clusters):
                    sub_df = pareto_df[pareto_df[cluster_col] == cluster_id]
                    c_color = cluster_colors[c_idx % len(cluster_colors)]
                    c_marker = MARKERS[c_idx % len(MARKERS)]
                    ax_ins.scatter(
                        sub_df[x_col],
                        sub_df[y_col],
                        color=c_color,
                        marker=c_marker,
                        s=34,
                        edgecolors="#111111",
                        linewidth=0.6,
                        zorder=4,
                    )
            else:
                ax_ins.scatter(
                    pareto_df[x_col],
                    pareto_df[y_col],
                    color=OKABE_ITO[0],
                    marker="o",
                    s=32,
                    edgecolors="#111111",
                    linewidth=0.6,
                    zorder=4,
                )

            ax_ins.plot(
                pareto_sorted[x_col],
                pareto_sorted[y_col],
                color="#222222",
                linestyle="--",
                linewidth=0.85,
                alpha=0.75,
                zorder=3,
            )

            ax_ins.set_xlim(-0.05, 1.0)
            ax_ins.set_ylim(5840.0, 5866.0)
            ax_ins.tick_params(labelsize=6.5)
            ax_ins.set_xlabel(
                r"$N_1$ [particles/s]",
                fontsize=7.0,
                labelpad=2,
                bbox=dict(boxstyle="square,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.92),
            )
            ax_ins.set_ylabel(
                r"$\Delta$ [particles/s]",
                fontsize=7.0,
                labelpad=3.8,
                bbox=dict(boxstyle="square,pad=0.12", fc="#FFFFFF", ec="none", alpha=0.92),
            )
            ax_ins.xaxis.set_minor_locator(AutoMinorLocator())
            ax_ins.yaxis.set_minor_locator(AutoMinorLocator())

            mark_inset(ax_pareto, ax_ins, loc1=2, loc2=3, fc="none", ec="#666666", ls=":", lw=0.9, alpha=0.8)

        add_panel_label(ax_pareto, "(b)", loc="outside_top_left")

        res_grid = save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )
        saved_all["combined_grid"] = res_grid

    return saved_all


def plot_case_study_efficiency(
    comparison_df: pd.DataFrame,
    labels_col: str = "variant",
    loss_col: str = "eta_1_pct",
    capture_col: str = "eta_2_pct",
    panel_label: Optional[str] = "(a)",
    figure_name: str = "Fig6a_efficiency_comparison",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.5, 3.8),
) -> Dict[str, Path]:
    """
    Generates Figure 6a: Engineering Validation - Separation Efficiencies eta_1 (loss) and eta_2 (capture) [%].
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(comparison_df))
        width = 0.30

        rects1 = ax.bar(
            x - width / 2,
            comparison_df[loss_col],
            width,
            label=r"Particle loss $\eta_1$",
            color=OKABE_ITO[1],
            edgecolor="#111111",
            linewidth=0.7,
        )
        rects2 = ax.bar(
            x + width / 2,
            comparison_df[capture_col],
            width,
            label=r"Capture efficiency $\eta_2$",
            color=OKABE_ITO[0],
            hatch="//",
            edgecolor="#111111",
            linewidth=0.7,
        )

        for rect in list(rects1) + list(rects2):
            h = rect.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
            )

        ax.set_ylabel("Separation efficiency [%]", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df[labels_col], fontsize=7.8)
        ax.set_xlim(-0.55, len(comparison_df) - 0.45)
        ax.set_ylim(0, 118)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92, ncol=2)

        if panel_label:
            add_panel_label(ax, panel_label, loc="outside_top_left")

        fig.tight_layout(pad=1.0)
        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_case_study_advantage(
    comparison_df: pd.DataFrame,
    labels_col: str = "variant",
    delta_col: str = "Delta",
    panel_label: Optional[str] = "(b)",
    figure_name: str = "Fig6b_advantage_comparison",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.5, 3.8),
) -> Dict[str, Path]:
    """
    Generates Figure 6b: Engineering Validation - Net Collection Advantage Delta [particles/s].
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(comparison_df))
        width = 0.42
        base_delta = comparison_df[delta_col].iloc[0]

        colors = [OKABE_ITO[3], OKABE_ITO[2], OKABE_ITO[2]]
        hatches = ["", "\\\\", "\\\\"]

        rects = ax.bar(
            x,
            comparison_df[delta_col],
            width,
            color=colors,
            edgecolor="#111111",
            linewidth=0.7,
        )
        for rect, htc in zip(rects, hatches):
            rect.set_hatch(htc)

        for idx, rect in enumerate(rects):
            h = rect.get_height()
            val_str = f"{int(round(h))} N/s"
            if idx == 0:
                gain_str = "(Baseline)"
            else:
                gain = ((h - base_delta) / base_delta) * 100.0
                gain_str = f"(+{gain:.1f}%)"
            ax.annotate(
                f"{val_str}\n{gain_str}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                multialignment="center",
            )

        ax.set_ylabel(r"Net capture advantage $\Delta$ [particles/s]", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df[labels_col], fontsize=7.8)
        ax.set_xlim(-0.55, len(comparison_df) - 0.45)
        ax.set_ylim(0, 7800)
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        from matplotlib.patches import Patch
        custom_handles = [
            Patch(facecolor=OKABE_ITO[3], edgecolor="#111111", linewidth=0.7, label="Baseline design"),
            Patch(facecolor=OKABE_ITO[2], hatch="\\\\", edgecolor="#111111", linewidth=0.7, label="Pareto optimal designs"),
        ]
        ax.legend(handles=custom_handles, loc="upper left", fontsize=7.5, framealpha=0.92, ncol=2)

        if panel_label:
            add_panel_label(ax, panel_label, loc="outside_top_left")

        fig.tight_layout(pad=1.0)
        return save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )


def plot_case_study_comparison(
    comparison_df: pd.DataFrame,
    labels_col: str = "variant",
    loss_col: str = "eta_1_pct",
    capture_col: str = "eta_2_pct",
    delta_col: str = "Delta",
    figure_name: str = "Fig6_case_study_comparison",
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig6",
    figsize: Tuple[float, float] = (7.6, 3.8),
) -> Dict[str, Any]:
    """
    Generuje komplet publikacyjny Figury 6:
    Panel (a): Separation efficiencies: loss efficiency eta_1 and capture efficiency eta_2 [%].
    Panel (b): Net particle collection advantage Delta = N2 - N1 [particles/s] with percentage gain over CFD baseline.
    By default generates BOTH two-column composite AND standalone panels.
    """
    saved_all: Dict[str, Any] = {}

    if save_individual:
        fname_a = f"{individual_prefix}a_efficiency_comparison"
        res_a = plot_case_study_efficiency(
            comparison_df=comparison_df,
            labels_col=labels_col,
            loss_col=loss_col,
            capture_col=capture_col,
            panel_label="(a)",
            figure_name=fname_a,
            output_dir=output_dir,
            formats=formats,
        )
        saved_all[fname_a] = res_a

        fname_b = f"{individual_prefix}b_advantage_comparison"
        res_b = plot_case_study_advantage(
            comparison_df=comparison_df,
            labels_col=labels_col,
            delta_col=delta_col,
            panel_label="(b)",
            figure_name=fname_b,
            output_dir=output_dir,
            formats=formats,
        )
        saved_all[fname_b] = res_b

    # Two-column composite
    with publication_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Panel (a): Efficiencies
        x = np.arange(len(comparison_df))
        w1 = 0.30
        rects1 = ax1.bar(
            x - w1 / 2,
            comparison_df[loss_col],
            w1,
            label=r"Particle loss $\eta_1$",
            color=OKABE_ITO[1],
            edgecolor="#111111",
            linewidth=0.7,
        )
        rects2 = ax1.bar(
            x + w1 / 2,
            comparison_df[capture_col],
            w1,
            label=r"Capture efficiency $\eta_2$",
            color=OKABE_ITO[0],
            hatch="//",
            edgecolor="#111111",
            linewidth=0.7,
        )

        for rect in list(rects1) + list(rects2):
            h = rect.get_height()
            ax1.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 2.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
            )

        ax1.set_ylabel("Separation efficiency [%]", fontsize=8.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison_df[labels_col], fontsize=7.8)
        ax1.set_xlim(-0.55, len(comparison_df) - 0.45)
        ax1.set_ylim(0, 118)
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.92, ncol=2)
        add_panel_label(ax1, "(a)", loc="outside_top_left")

        # Panel (b): Net advantage Delta
        w2 = 0.42
        base_delta = comparison_df[delta_col].iloc[0]
        colors = [OKABE_ITO[3], OKABE_ITO[2], OKABE_ITO[2]]
        hatches = ["", "\\\\", "\\\\"]
        rects_d = ax2.bar(
            x,
            comparison_df[delta_col],
            w2,
            color=colors,
            edgecolor="#111111",
            linewidth=0.7,
        )
        for rect, htc in zip(rects_d, hatches):
            rect.set_hatch(htc)

        for idx, rect in enumerate(rects_d):
            h = rect.get_height()
            val_str = f"{int(round(h))} N/s"
            if idx == 0:
                gain_str = "(Baseline)"
            else:
                gain = ((h - base_delta) / base_delta) * 100.0
                gain_str = f"(+{gain:.1f}%)"
            ax2.annotate(
                f"{val_str}\n{gain_str}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                multialignment="center",
            )

        ax2.set_ylabel(r"Net capture advantage $\Delta$ [particles/s]", fontsize=8.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison_df[labels_col], fontsize=7.8)
        ax2.set_xlim(-0.55, len(comparison_df) - 0.45)
        ax2.set_ylim(0, 7800)
        ax2.yaxis.set_minor_locator(AutoMinorLocator())

        from matplotlib.patches import Patch
        custom_handles = [
            Patch(facecolor=OKABE_ITO[3], edgecolor="#111111", linewidth=0.7, label="Baseline design"),
            Patch(facecolor=OKABE_ITO[2], hatch="\\\\", edgecolor="#111111", linewidth=0.7, label="Pareto optimal designs"),
        ]
        ax2.legend(handles=custom_handles, loc="upper left", fontsize=7.5, framealpha=0.92, ncol=2)
        add_panel_label(ax2, "(b)", loc="outside_top_left")

        fig.tight_layout(pad=1.2)
        res_grid = save_publication_figure(
            fig, figure_name, output_dir=output_dir, formats=formats
        )
        saved_all["combined_grid"] = res_grid

    return saved_all
