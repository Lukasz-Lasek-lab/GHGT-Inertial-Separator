"""
Figure 4: Sensitivity Sweep Profiles (1D Dual-Y Sweeps and 2D Interaction Contours).
Provides panel renderers and publication figure generators for 1D sensitivity curves and 2D response contours.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

from src.visualization.style import (
    LINE_STYLES,
    OKABE_ITO,
    PARAM_AXIS_LABEL_MAP,
    PARAM_BOUNDS,
    PARAM_NOMINAL_LABEL_MAP,
    add_panel_label,
    publication_style,
    save_publication_figure,
)


def draw_sweep_1d_panel(
    ax1: plt.Axes,
    df_param: pd.DataFrame,
    param: str,
    ref_val: Optional[float] = None,
    panel_label: Optional[str] = None,
    show_legend: bool = True,
    line_width: float = 1.8,
    n1_color: str = OKABE_ITO[0],
    n2_color: str = OKABE_ITO[1],
) -> plt.Axes:
    """
    Renders a dual-Y 1D sensitivity sweep profile on ax1 (and twin ax2).
    Left axis: N1 (particle loss), Right axis: N2 (carrier capture).
    """
    # Determine column names flexibly
    col_n1 = "N1_pred" if "N1_pred" in df_param.columns else ("N1" if "N1" in df_param.columns else df_param.columns[1])
    col_n2 = "N2_pred" if "N2_pred" in df_param.columns else ("N2" if "N2" in df_param.columns else df_param.columns[2])

    # Left axis: N1 (loss)
    line1 = ax1.plot(
        df_param[param],
        df_param[col_n1],
        color=n1_color,
        linestyle=LINE_STYLES[0],
        linewidth=line_width,
        label="Particle loss $N_1$",
        zorder=4,
    )
    xlabel = PARAM_AXIS_LABEL_MAP.get(param, f"Parameter {param}")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Particle loss $N_1$ [particles/s]", color=n1_color)
    ax1.tick_params(axis="y", labelcolor=n1_color)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    lines = list(line1)
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
        df_param[col_n2],
        color=n2_color,
        linestyle=LINE_STYLES[1],
        linewidth=line_width,
        label="Captured particles $N_2$",
        zorder=4,
    )
    ax2.set_ylabel("Captured particles $N_2$ [particles/s]", color=n2_color)
    ax2.tick_params(axis="y", labelcolor=n2_color)
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.grid(False)

    # Horizontal margins
    span = float(df_param[param].max() - df_param[param].min())
    margin = 0.025 * span if span > 0 else 0.5
    ax1.set_xlim(float(df_param[param].min()) - margin, float(df_param[param].max()) + margin)

    lines.extend(line2)
    labels = [line.get_label() for line in lines]
    legend_loc_map = {
        "Alfa": "center left",
        "Beta": "center left",
        "H1": "upper left",
        "H2": "lower left",
    }
    legend_loc = legend_loc_map.get(param, "best")
    if show_legend:
        ax1.legend(lines, labels, loc=legend_loc, framealpha=0.92)

    if panel_label:
        add_panel_label(ax1, panel_label, loc="outside_top_left")

    return ax2


def draw_contour_2d_panel(
    ax: plt.Axes,
    G1: np.ndarray,
    G2: np.ndarray,
    Z: np.ndarray,
    p1: str,
    p2: str,
    target: str = "N1",
    panel_label: Optional[str] = None,
    contour_levels: int = 16,
    fig: Optional[plt.Figure] = None,
    add_colorbar: bool = True,
    cbar_pad: float = 0.03,
    cbar_aspect: float = 22.0,
) -> Any:
    """
    Renders 2D interaction contour map (heatmap) on Axes.
    """
    if target == "N1":
        cmap = "viridis_r"
        cbar_label = "Particle loss $N_1$ [particles/s]"
    else:
        cmap = "viridis"
        cbar_label = "Captured particles $N_2$ [particles/s]"

    cf = ax.contourf(G1, G2, Z, levels=contour_levels, cmap=cmap)

    if add_colorbar and fig is not None:
        cbar = fig.colorbar(cf, ax=ax, pad=cbar_pad, aspect=cbar_aspect)
        cbar.set_label(cbar_label, fontsize=8.0)
        cbar.ax.tick_params(labelsize=7.5)
        cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_xlabel(PARAM_AXIS_LABEL_MAP.get(p1, p1))
    ax.set_ylabel(PARAM_AXIS_LABEL_MAP.get(p2, p2))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(float(G1.min()), float(G1.max()))
    ax.set_ylim(float(G2.min()), float(G2.max()))

    if panel_label:
        add_panel_label(ax, panel_label, loc="outside_top_left")

    return cf


def plot_sensitivity_sweep_single(
    df_param: pd.DataFrame,
    param: str,
    ref_val: Optional[float] = None,
    panel_label: Optional[str] = "(a)",
    figure_name: str = "Fig4a_sweep_Alfa",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.0, 3.5),
) -> Dict[str, Path]:
    """
    Generates an autonomous 1D dual-response profile (Dual-Y axis) for a single parameter.
    """
    with publication_style():
        fig, ax1 = plt.subplots(figsize=figsize)
        draw_sweep_1d_panel(
            ax1=ax1,
            df_param=df_param,
            param=param,
            ref_val=ref_val,
            panel_label=panel_label,
            show_legend=True,
            line_width=1.8,
        )
        fig.tight_layout(pad=0.8)
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


# Alias for backward compatibility and test consistency
plot_sensitivity_1d_single = plot_sensitivity_sweep_single


def plot_sensitivity_sweeps_1d(
    sensitivity_data: Dict[str, pd.DataFrame],
    ref_params: Dict[str, float],
    units: Optional[Dict[str, str]] = None,
    figure_name: str = "Fig4_sensitivity_sweeps_1d",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig4",
) -> Dict[str, Any]:
    """
    Generates Figure 4: 1D Engineering Sensitivity Sweeps around Reference Point (composite 2x2 grid).
    """
    saved_all: Dict[str, Any] = {}
    params = list(sensitivity_data.keys())
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    if save_individual:
        for i, param in enumerate(params):
            p_lbl = panel_labels[i] if i < len(panel_labels) else f"({chr(97+i)})"
            f_letter = chr(97 + i)
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

    # Composite 2x2 grid
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
        axes_flat = axes.flatten()

        for i, param in enumerate(params):
            ax1 = axes_flat[i]
            df_p = sensitivity_data[param]
            ref_val = ref_params.get(param, None)
            p_lbl = panel_labels[i] if i < len(panel_labels) else f"({chr(97+i)})"

            draw_sweep_1d_panel(
                ax1=ax1,
                df_param=df_p,
                param=param,
                ref_val=ref_val,
                panel_label=p_lbl,
                show_legend=True,
                line_width=1.6,
            )

        fig.tight_layout(pad=1.4)
        res_grid = save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)
        saved_all["combined_grid"] = res_grid

    return saved_all


# Alias for backward compatibility
plot_sensitivity_1d = plot_sensitivity_sweeps_1d


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
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.3, 3.8),
    contour_levels: int = 16,
) -> Dict[str, Path]:
    """
    Generates an autonomous 2D interaction contour map (heatmap) for selected parameter pair.
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_contour_2d_panel(
            ax=ax,
            G1=G1,
            G2=G2,
            Z=Z,
            p1=p1,
            p2=p2,
            target=target,
            panel_label=panel_label,
            contour_levels=contour_levels,
            fig=fig,
            add_colorbar=True,
            cbar_pad=0.03,
            cbar_aspect=22.0,
        )
        fig.tight_layout(pad=0.8)
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


def plot_sensitivity_heatmaps_2d(
    predict_fn: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ref_params: Dict[str, float],
    param_pairs: Optional[List[Tuple[str, str, Tuple[float, float], Tuple[float, float]]]] = None,
    units: Optional[Dict[str, str]] = None,
    grid_size: int = 40,
    figure_name: str = "Fig4_sensitivity_heatmaps_2d",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig4",
) -> Dict[str, Any]:
    """
    Generates 2D interaction contour maps (Alfa-Beta and H1-H2) for both N1 and N2.
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

    if save_individual:
        for row_idx, (p1, p2, G1, G2, N1_grid, N2_grid) in enumerate(grids):
            let_n1, let_n2 = panel_letters[row_idx]
            ref_val1 = ref_params.get(p1, None)
            ref_val2 = ref_params.get(p2, None)

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

    # Composite 2x2 grid
    with publication_style():
        fig, axes = plt.subplots(len(param_pairs), 2, figsize=(7.6, 3.6 * len(param_pairs)))

        for row_idx, (p1, p2, G1, G2, N1_grid, N2_grid) in enumerate(grids):
            ax_n1 = axes[row_idx, 0] if len(param_pairs) > 1 else axes[0]
            draw_contour_2d_panel(
                ax=ax_n1,
                G1=G1,
                G2=G2,
                Z=N1_grid,
                p1=p1,
                p2=p2,
                target="N1",
                panel_label=panel_labels[row_idx][0],
                fig=fig,
                add_colorbar=True,
                cbar_pad=0.02,
                cbar_aspect=20.0,
            )

            ax_n2 = axes[row_idx, 1] if len(param_pairs) > 1 else axes[1]
            draw_contour_2d_panel(
                ax=ax_n2,
                G1=G1,
                G2=G2,
                Z=N2_grid,
                p1=p1,
                p2=p2,
                target="N2",
                panel_label=panel_labels[row_idx][1],
                fig=fig,
                add_colorbar=True,
                cbar_pad=0.02,
                cbar_aspect=20.0,
            )

        fig.tight_layout(pad=1.4)
        res_grid = save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)
        saved_all["combined_grid"] = res_grid

    return saved_all


__all__ = [
    "draw_sweep_1d_panel",
    "draw_contour_2d_panel",
    "plot_sensitivity_sweep_single",
    "plot_sensitivity_1d_single",
    "plot_sensitivity_sweeps_1d",
    "plot_sensitivity_1d",
    "plot_sensitivity_heatmap_single",
    "plot_sensitivity_heatmaps_2d",
]
