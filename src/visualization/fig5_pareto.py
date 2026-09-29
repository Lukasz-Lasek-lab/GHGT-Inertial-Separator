"""
Figure 5: Multi-Objective NSGA-II Convergence & Pareto Optimal Trade-off Space.
Provides panel renderers and publication generators for generational convergence histories,
Pareto front scatter plots with physics-consistent cluster identification, and micro-scale inset zooms.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd

from src.visualization.style import (
    MARKERS,
    OKABE_ITO,
    add_panel_label,
    publication_style,
    save_publication_figure,
)


def _resolve_cluster_labels(
    pareto_df: pd.DataFrame,
    cluster_col: Optional[str] = "cluster",
    cluster_labels_map: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    """
    Resolves cluster labels deterministically and physically based on baffle height H2.
    Prevents mislabeling high-gap (H2 ~ 59 mm) vs low-gap (H2 ~ 39 mm) Pareto regimes.
    """
    if cluster_col is None or cluster_col not in pareto_df.columns:
        return {}

    unique_clusters = np.sort(pareto_df[cluster_col].unique())
    resolved_map = {}

    if cluster_labels_map:
        resolved_map.update(cluster_labels_map)

    # Verify and auto-correct or assign physical labels if H2 is present
    if "H2" in pareto_df.columns:
        for cid in unique_clusters:
            sub = pareto_df[pareto_df[cluster_col] == cid]
            mean_h2 = float(sub["H2"].mean()) if len(sub) > 0 else 0.05
            h2_mm = mean_h2 * 1000.0
            physical_label = rf"Pareto Cluster {cid + 1} ($H_2 \approx {h2_mm:.0f}\text{{ mm}}$)"
            if cid not in resolved_map:
                resolved_map[cid] = physical_label
            else:
                lbl = resolved_map[cid]
                # If label is generic placeholder (e.g. "Pareto cluster 3" without H2)
                if "H_2" not in lbl and "H2" not in lbl:
                    resolved_map[cid] = physical_label
                # Check for legacy mismatch where a ~60mm cluster was incorrectly labeled ~39mm
                elif mean_h2 >= 0.048 and "39" in lbl and "59" not in lbl:
                    resolved_map[cid] = rf"Pareto Cluster {cid + 1} ($H_2 \approx 59\text{{ mm}}$)"
                elif mean_h2 < 0.048 and "59" in lbl and "39" not in lbl:
                    resolved_map[cid] = rf"Pareto Cluster {cid + 1} ($H_2 \approx 39\text{{ mm}}$)"
    else:
        for cid in unique_clusters:
            if cid not in resolved_map:
                resolved_map[cid] = f"Pareto cluster {cid + 1}"

    return resolved_map


def draw_convergence(
    ax1: plt.Axes,
    ax2: plt.Axes,
    log_df: pd.DataFrame,
    panel_labels: Tuple[str, str] = ("(a)", "(b)"),
    show_legend: bool = True,
    text_fontsize: float = 7.5,
    line_width: float = 1.3,
) -> None:
    """
    Renders NSGA-II generational convergence history across ax1 (N1) and ax2 (Delta).
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

    # Panel 1: N1 (minimization)
    ax1.plot(generations, min_n1, color=OKABE_ITO[2], linestyle="-", linewidth=line_width, label="Min")
    ax1.plot(generations, avg_n1, color=OKABE_ITO[0], linestyle="--", linewidth=line_width, label="Mean")
    ax1.plot(generations, max_n1, color=OKABE_ITO[1], linestyle=":", linewidth=line_width, label="Max")
    ax1.fill_between(generations, min_n1, max_n1, color=OKABE_ITO[0], alpha=0.12)
    ax1.set_ylabel("Loss objective $N_1$ [particles/s]", fontsize=8.2)
    ax1.set_ylim(-15, 525)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    if show_legend:
        ax1.legend(loc="upper right", fontsize=text_fontsize, framealpha=0.88)
    if panel_labels and len(panel_labels) > 0 and panel_labels[0]:
        add_panel_label(ax1, panel_labels[0], loc="outside_top_left")

    # Panel 2: Delta (maximization)
    ax2.plot(generations, max_d, color=OKABE_ITO[2], linestyle="-", linewidth=line_width, label="Max")
    ax2.plot(generations, avg_d, color=OKABE_ITO[0], linestyle="--", linewidth=line_width, label="Mean")
    ax2.plot(generations, min_d, color=OKABE_ITO[1], linestyle=":", linewidth=line_width, label="Min")
    ax2.fill_between(generations, min_d, max_d, color=OKABE_ITO[0], alpha=0.12)
    ax2.set_xlabel("Evolution generation $g$ [-]", fontsize=8.2)
    ax2.set_ylabel(r"Advantage objective $\Delta$ [particles/s]", fontsize=8.2)
    ax2.set_ylim(3300, 6050)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    if show_legend:
        ax2.legend(loc="lower right", fontsize=text_fontsize, framealpha=0.88)
    if panel_labels and len(panel_labels) > 1 and panel_labels[1]:
        add_panel_label(ax2, panel_labels[1], loc="outside_top_left")


def draw_pareto_front(
    ax: plt.Axes,
    all_population_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    baseline_point: Optional[Dict[str, float]] = None,
    cluster_col: Optional[str] = "cluster",
    cluster_labels_map: Optional[Dict[int, str]] = None,
    panel_label: Optional[str] = None,
    show_legend: bool = True,
    point_size_pareto: float = 44.0,
    point_size_exp: float = 14.0,
    text_fontsize: float = 7.5,
) -> None:
    """
    Renders design space exploration candidates and non-dominated Pareto front on Axes.
    """
    x_col = "N1_pred" if "N1_pred" in pareto_df.columns else "N1"
    y_col = "Delta_pred" if "Delta_pred" in pareto_df.columns else "Delta"
    x_pop_col = "N1_pred" if "N1_pred" in all_population_df.columns else "N1"
    y_pop_col = "Delta_pred" if "Delta_pred" in all_population_df.columns else "Delta"

    # 1. Explored candidates cloud
    ax.scatter(
        all_population_df[x_pop_col],
        all_population_df[y_pop_col],
        color="#A8A8A8",
        s=point_size_exp,
        alpha=0.35,
        edgecolors="none",
        label="Explored candidates",
        zorder=2,
    )

    # 2. Pareto front solutions
    resolved_labels = _resolve_cluster_labels(pareto_df, cluster_col=cluster_col, cluster_labels_map=cluster_labels_map)

    if cluster_col and cluster_col in pareto_df.columns:
        clusters = np.sort(pareto_df[cluster_col].unique())
        cluster_colors = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[5]]
        for c_idx, cluster_id in enumerate(clusters):
            sub_df = pareto_df[pareto_df[cluster_col] == cluster_id]
            c_color = cluster_colors[c_idx % len(cluster_colors)]
            c_marker = MARKERS[c_idx % len(MARKERS)]
            c_lbl = resolved_labels.get(cluster_id, f"Pareto cluster {cluster_id + 1}")
            ax.scatter(
                sub_df[x_col],
                sub_df[y_col],
                color=c_color,
                marker=c_marker,
                s=point_size_pareto,
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
            s=point_size_pareto,
            edgecolors="#111111",
            linewidth=0.7,
            label="Non-dominated Pareto front",
            zorder=4,
        )

    # Connect Pareto solutions with trade-off spline
    pareto_sorted = pareto_df.sort_values(by=x_col)
    ax.plot(
        pareto_sorted[x_col],
        pareto_sorted[y_col],
        color="#222222",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        zorder=3,
    )

    # 3. CFD baseline reference point
    if baseline_point is not None:
        ax.scatter(
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

    min_x = min(float(all_population_df[x_pop_col].min()), float(pareto_df[x_col].min())) if len(all_population_df) > 0 and len(pareto_df) > 0 else 0.0
    max_x = max(float(all_population_df[x_pop_col].max()), float(pareto_df[x_col].max())) if len(all_population_df) > 0 and len(pareto_df) > 0 else 500.0
    min_y = min(float(all_population_df[y_pop_col].min()), float(pareto_df[y_col].min())) if len(all_population_df) > 0 and len(pareto_df) > 0 else 3500.0
    max_y = max(float(all_population_df[y_pop_col].max()), float(pareto_df[y_col].max())) if len(all_population_df) > 0 and len(pareto_df) > 0 else 6000.0

    xlim_lower = min(-15.0, min_x - 0.05 * abs(max_x - min_x))
    xlim_upper = max(525.0, max_x + 0.05 * abs(max_x - min_x))
    ylim_lower = min(3400.0, min_y - 0.05 * abs(max_y - min_y))
    ylim_upper = max(6100.0, max_y + 0.05 * abs(max_y - min_y))

    ax.set_xlabel(r"Particle loss $N_1$ [particles/s] (Minimization $\rightarrow$)", fontsize=8.2)
    ax.set_ylabel(r"Capture advantage $\Delta$ [particles/s] (Maximization $\rightarrow$)", fontsize=8.2)
    ax.set_xlim(xlim_lower, xlim_upper)
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if show_legend:
        ax.legend(loc="lower left", fontsize=text_fontsize, framealpha=0.90)

    if panel_label:
        add_panel_label(ax, panel_label, loc="outside_top_left")


def draw_inset_zoom(
    ax_parent: plt.Axes,
    pareto_df: pd.DataFrame,
    cluster_col: Optional[str] = "cluster",
    cluster_labels_map: Optional[Dict[int, str]] = None,
    inset_xlim: Tuple[float, float] = (-0.05, 1.0),
    inset_ylim: Tuple[float, float] = (5840.0, 5866.0),
    width: str = "44%",
    height: str = "37%",
    loc: str = "upper right",
    bbox_to_anchor: Tuple[float, float, float, float] = (0.0, -0.02, 0.98, 0.98),
) -> plt.Axes:
    """
    Renders inset zoom detailing micro-scale Pareto front cluster separation.
    """
    x_col = "N1_pred" if "N1_pred" in pareto_df.columns else "N1"
    y_col = "Delta_pred" if "Delta_pred" in pareto_df.columns else "Delta"

    if len(pareto_df) > 0:
        in_bounds = (
            (pareto_df[x_col] >= inset_xlim[0]) & (pareto_df[x_col] <= inset_xlim[1]) &
            (pareto_df[y_col] >= inset_ylim[0]) & (pareto_df[y_col] <= inset_ylim[1])
        )
        if not in_bounds.any():
            x_min = float(pareto_df[x_col].min())
            x_max = float(pareto_df[x_col].max())
            x_span = max(x_max - x_min, 1.0)
            inset_xlim = (x_min - 0.1 * x_span, x_max + 0.1 * x_span)

            y_min = float(pareto_df[y_col].min())
            y_max = float(pareto_df[y_col].max())
            y_span = max(y_max - y_min, 10.0)
            inset_ylim = (y_min - 0.1 * y_span, y_max + 0.1 * y_span)

    ax_ins = inset_axes(
        ax_parent,
        width=width,
        height=height,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax_parent.transAxes,
        borderpad=1.0,
    )
    ax_ins.set_facecolor("#FFFFFF")

    if cluster_col and cluster_col in pareto_df.columns:
        clusters = np.sort(pareto_df[cluster_col].unique())
        cluster_colors = [OKABE_ITO[0], OKABE_ITO[2], OKABE_ITO[3], OKABE_ITO[5]]
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

    pareto_sorted = pareto_df.sort_values(by=x_col)
    ax_ins.plot(
        pareto_sorted[x_col],
        pareto_sorted[y_col],
        color="#222222",
        linestyle="--",
        linewidth=0.85,
        alpha=0.75,
        zorder=3,
    )

    ax_ins.set_xlim(inset_xlim)
    ax_ins.set_ylim(inset_ylim)
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

    mark_inset(ax_parent, ax_ins, loc1=2, loc2=3, fc="none", ec="#666666", ls=":", lw=0.9, alpha=0.8)
    return ax_ins


def plot_convergence(
    log_df: pd.DataFrame,
    panel_labels: Tuple[str, str] = ("(a)", "(b)"),
    figure_name: str = "Fig5a_nsga2_convergence",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.5, 4.8),
) -> Dict[str, Path]:
    """
    Generates Figure 5a: Multi-Objective NSGA-II Generational Convergence.
    """
    with publication_style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        draw_convergence(
            ax1=ax1,
            ax2=ax2,
            log_df=log_df,
            panel_labels=panel_labels,
            show_legend=True,
            text_fontsize=8.0,
            line_width=1.5,
        )
        fig.tight_layout(pad=0.8)
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


plot_convergence_single = plot_convergence


def plot_pareto_front(
    all_population_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    baseline_point: Optional[Dict[str, float]] = None,
    cluster_col: Optional[str] = "cluster",
    cluster_labels_map: Optional[Dict[int, str]] = None,
    panel_label: Optional[str] = "(b)",
    figure_name: str = "Fig5b_pareto_front",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (5.2, 4.4),
    show_inset: bool = True,
    inset_xlim: Tuple[float, float] = (-0.05, 1.0),
    inset_ylim: Tuple[float, float] = (5840.0, 5866.0),
) -> Dict[str, Path]:
    """
    Generates Figure 5b: Objective Trade-off Space and Pareto Optimal Front.
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_pareto_front(
            ax=ax,
            all_population_df=all_population_df,
            pareto_df=pareto_df,
            baseline_point=baseline_point,
            cluster_col=cluster_col,
            cluster_labels_map=cluster_labels_map,
            panel_label=panel_label,
            show_legend=True,
            point_size_pareto=46.0,
            point_size_exp=16.0,
            text_fontsize=8.0,
        )
        fig.tight_layout(pad=0.8)
        if show_inset:
            draw_inset_zoom(
                ax_parent=ax,
                pareto_df=pareto_df,
                cluster_col=cluster_col,
                cluster_labels_map=cluster_labels_map,
                inset_xlim=inset_xlim,
                inset_ylim=inset_ylim,
                width="45%",
                height="38%",
                loc="upper right",
            )
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


plot_pareto_front_single = plot_pareto_front


def plot_optimization_figure_5(
    log_df: pd.DataFrame,
    all_population_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    baseline_point: Optional[Dict[str, float]] = None,
    cluster_col: Optional[str] = "cluster",
    cluster_labels_map: Optional[Dict[int, str]] = None,
    figure_name: str = "Fig5_optimization_pareto",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig5",
    show_inset: bool = True,
) -> Dict[str, Any]:
    """
    Generates complete publication Figure 5:
    Left column: generational convergence history (N1 and Delta)
    Right column: multi-objective trade-off space, Pareto front with CFD baseline and cluster inset zoom.
    """
    saved_all: Dict[str, Any] = {}

    resolved_labels = _resolve_cluster_labels(
        pareto_df, cluster_col=cluster_col, cluster_labels_map=cluster_labels_map
    )

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
            cluster_labels_map=resolved_labels,
            panel_label="(b)",
            figure_name=fname_b,
            output_dir=output_dir,
            formats=formats,
            show_inset=show_inset,
        )
        saved_all[fname_b] = res_b

    # Composite layout
    with publication_style():
        fig = plt.figure(figsize=(7.6, 4.0))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], hspace=0.25, wspace=0.32)

        # 1. Left column: convergence
        ax_conv1 = fig.add_subplot(gs[0, 0])
        ax_conv2 = fig.add_subplot(gs[1, 0], sharex=ax_conv1)
        draw_convergence(
            ax1=ax_conv1,
            ax2=ax_conv2,
            log_df=log_df,
            panel_labels=("(a)", ""),
            show_legend=True,
            text_fontsize=7.2,
            line_width=1.3,
        )

        # 2. Right column: Pareto front
        ax_pareto = fig.add_subplot(gs[:, 1])
        draw_pareto_front(
            ax=ax_pareto,
            all_population_df=all_population_df,
            pareto_df=pareto_df,
            baseline_point=baseline_point,
            cluster_col=cluster_col,
            cluster_labels_map=resolved_labels,
            panel_label="(b)",
            show_legend=True,
            point_size_pareto=44.0,
            point_size_exp=14.0,
            text_fontsize=7.5,
        )

        if show_inset:
            draw_inset_zoom(
                ax_parent=ax_pareto,
                pareto_df=pareto_df,
                cluster_col=cluster_col,
                cluster_labels_map=resolved_labels,
                inset_xlim=(-0.05, 1.0),
                inset_ylim=(5840.0, 5866.0),
                width="44%",
                height="37%",
                loc="upper right",
            )

        res_grid = save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)
        saved_all["combined_grid"] = res_grid

    return saved_all


__all__ = [
    "draw_convergence",
    "draw_pareto_front",
    "draw_inset_zoom",
    "plot_convergence",
    "plot_convergence_single",
    "plot_pareto_front",
    "plot_pareto_front_single",
    "plot_optimization_figure_5",
]
