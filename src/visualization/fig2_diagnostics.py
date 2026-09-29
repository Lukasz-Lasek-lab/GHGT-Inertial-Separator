"""
Figure 2: Model Diagnostics & Surrogate Residual Analysis.
Provides clean panel renderers and publication generators for parity plots and residual error distributions.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.visualization.style import (
    MARKERS,
    OKABE_ITO,
    add_panel_label,
    publication_style,
    save_publication_figure,
)


def draw_parity_panel(
    ax: plt.Axes,
    y_true: Union[pd.Series, np.ndarray, Sequence[float]],
    y_pred: Union[pd.Series, np.ndarray, Sequence[float]],
    target_name: str = "$N_1$",
    target_label: str = "Particle loss",
    unit: str = "[particles/s]",
    color: str = OKABE_ITO[0],
    marker: str = "o",
    panel_label: Optional[str] = None,
    show_legend: bool = True,
    point_size: float = 36.0,
    text_fontsize: float = 7.8,
) -> Dict[str, float]:
    """
    Renders a single parity correlation panel on the provided matplotlib Axes.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    r2 = float(r2_score(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    ax.scatter(
        y_t,
        y_p,
        color=color,
        marker=marker,
        s=point_size,
        alpha=0.75,
        edgecolors="#222222",
        linewidth=0.6,
        label="Samples",
        zorder=4,
    )

    # Identity reference line y = x
    min_val = min(float(y_t.min()), float(y_p.min()))
    max_val = max(float(y_t.max()), float(y_p.max()))
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

    tgt_desc = f"{target_label} {target_name}".strip()
    ax.set_xlabel(f"Actual {tgt_desc} {unit}")
    ax.set_ylabel(f"Predicted {tgt_desc} {unit}")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    stats_str = f"$R^2 = {r2:.3f}$\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
    ax.text(
        0.05,
        0.72,
        stats_str,
        transform=ax.transAxes,
        fontsize=text_fontsize,
        verticalalignment="top",
        bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
        zorder=10,
    )

    if panel_label:
        add_panel_label(ax, panel_label, loc="top_left")

    if show_legend:
        ax.legend(loc="lower right", framealpha=0.88)

    return {"r2": r2, "mae": mae, "rmse": rmse}


def draw_residuals_panel(
    ax: plt.Axes,
    y_true_df: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float], Dict[str, np.ndarray]]] = None,
    y_pred_df: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]]] = None,
    targets: Optional[Sequence[Tuple[str, str, str]]] = None,
    residuals_dict: Optional[Dict[str, np.ndarray]] = None,
    panel_label: Optional[str] = None,
    bins: int = 18,
    show_legend: bool = True,
) -> None:
    """
    Renders prediction residual error distribution histogram on the provided Axes.
    Supports either precomputed residuals (array/dict) or true vs predicted values.
    """
    resolved_residuals: Dict[str, np.ndarray] = {}

    if residuals_dict is not None:
        resolved_residuals = residuals_dict
    elif isinstance(y_true_df, dict):
        resolved_residuals = y_true_df
    elif y_true_df is not None and y_pred_df is None:
        # Caller provided a single argument of residuals: draw_residuals_panel(ax, residuals)
        if isinstance(y_true_df, pd.DataFrame):
            resolved_residuals = {col: y_true_df[col].values for col in y_true_df.columns}
        else:
            resolved_residuals = {"": np.asarray(y_true_df, dtype=float)}
    elif y_true_df is not None and y_pred_df is not None:
        if isinstance(y_true_df, pd.DataFrame) and isinstance(y_pred_df, pd.DataFrame):
            if targets is None:
                targets = [
                    ("N1", "$N_1$", OKABE_ITO[0]),
                    ("N2", "$N_2$", OKABE_ITO[1]),
                ]
            for col, label, _ in targets:
                if col in y_true_df.columns and col in y_pred_df.columns:
                    resolved_residuals[label] = y_true_df[col].values - y_pred_df[col].values
        else:
            res = np.asarray(y_true_df, dtype=float) - np.asarray(y_pred_df, dtype=float)
            resolved_residuals = {"": res}
    else:
        raise ValueError("Either residuals_dict, residuals, or both y_true_df and y_pred_df must be provided.")

    colors = [OKABE_ITO[0], OKABE_ITO[1], OKABE_ITO[2]]
    for idx, (label, res) in enumerate(resolved_residuals.items()):
        col_c = colors[idx % len(colors)]
        lbl_str = f"Residuals {label}".strip() if label else "Residuals"
        ax.hist(
            res,
            bins=bins,
            alpha=0.65 - (idx * 0.05),
            color=col_c,
            edgecolor="#222222",
            linewidth=0.6,
            label=lbl_str,
        )

    ax.axvline(0, color="#000000", linestyle="--", linewidth=1.2, label="Zero error line")
    ax.set_xlabel("Prediction error ($y_{\\mathrm{actual}} - y_{\\mathrm{predicted}}$) [particles/s]")
    ax.set_ylabel("Frequency [samples]")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if panel_label:
        add_panel_label(ax, panel_label, loc="top_left")

    if show_legend:
        ax.legend(loc="upper right", framealpha=0.88)


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
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (3.9, 3.8),
) -> Dict[str, Path]:
    """
    Generates an autonomous parity correlation plot for a single target.
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_parity_panel(
            ax=ax,
            y_true=y_true,
            y_pred=y_pred,
            target_name=target_name,
            target_label=target_label,
            unit=unit,
            color=color,
            marker=marker,
            panel_label=panel_label,
            show_legend=True,
            point_size=42.0,
            text_fontsize=8.0,
        )
        fig.tight_layout(pad=1.0)
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


def plot_residuals_single(
    y_true_df: Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float], Dict[str, np.ndarray]],
    y_pred_df: Optional[Union[pd.DataFrame, pd.Series, np.ndarray, Sequence[float]]] = None,
    targets: Optional[List[Tuple[str, str, str]]] = None,
    panel_label: Optional[str] = None,
    figure_name: str = "Fig_residuals_distribution",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.2, 3.5),
) -> Dict[str, Path]:
    """
    Generates an autonomous residual error distribution figure.
    Supports either precomputed residuals or actual and predicted outputs.
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_residuals_panel(
            ax=ax,
            y_true_df=y_true_df,
            y_pred_df=y_pred_df,
            targets=targets,
            panel_label=panel_label,
            bins=20,
            show_legend=True,
        )
        fig.tight_layout()
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


def plot_model_diagnostics(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    targets: Optional[List[Tuple[str, str, str]]] = None,
    figure_name: str = "Fig2_model_diagnostics",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig2",
) -> Dict[str, Any]:
    """
    Generates Figure 2: Model Diagnostics and Surrogate Predictive Performance.
    Combines 3 parity correlation panels and 1 residual distribution panel into a 2x2 grid,
    with optional export of individual standalone panels.
    """
    if targets is None:
        targets = [
            ("N1", "Particle loss $N_1$", "[particles/s]"),
            ("N2", "Captured particles $N_2$", "[particles/s]"),
            ("Delta", "Net capture advantage $\\Delta$", "[particles/s]"),
        ]

    saved_all: Dict[str, Any] = {}

    # 1. Standalone panel figures
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

        res_dist = plot_residuals_single(
            y_true_df=y_true_df,
            y_pred_df=y_pred_df,
            panel_label="(d)",
            figure_name=f"{individual_prefix}d_residuals_distribution",
            output_dir=output_dir,
            formats=formats,
        )
        saved_all[f"{individual_prefix}d_residuals_distribution"] = res_dist

    # 2. Composite 2x2 grid
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.8))
        axes_flat = axes.flatten()
        panel_labels = ["(a)", "(b)", "(c)", "(d)"]

        for i, (col, label_name, unit) in enumerate(targets):
            color = OKABE_ITO[i % len(OKABE_ITO)]
            marker = MARKERS[i % len(MARKERS)]
            draw_parity_panel(
                ax=axes_flat[i],
                y_true=y_true_df[col],
                y_pred=y_pred_df[col],
                target_name="",
                target_label=label_name,
                unit=unit,
                color=color,
                marker=marker,
                panel_label=panel_labels[i],
                show_legend=True,
                point_size=32.0,
                text_fontsize=7.8,
            )

        # Residuals in panel (d)
        draw_residuals_panel(
            ax=axes_flat[3],
            y_true_df=y_true_df,
            y_pred_df=y_pred_df,
            targets=[
                ("N1", "$N_1$", OKABE_ITO[0]),
                ("N2", "$N_2$", OKABE_ITO[1]),
            ],
            panel_label=panel_labels[3],
            bins=18,
            show_legend=True,
        )

        fig.tight_layout(pad=1.2)
        res_grid = save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)
        saved_all["combined_grid"] = res_grid

    return saved_all


__all__ = [
    "draw_parity_panel",
    "draw_residuals_panel",
    "plot_parity_single",
    "plot_residuals_single",
    "plot_model_diagnostics",
]
