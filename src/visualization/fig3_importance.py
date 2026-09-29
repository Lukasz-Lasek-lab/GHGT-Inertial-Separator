"""
Figure 3: Geometric & Aerodynamic Feature Importance (Permutation Importance / XAI).
Provides panel renderers and publication generators for feature importance bars and multi-objective comparisons.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

from src.visualization.style import (
    FEATURE_LABEL_MAP,
    OKABE_ITO,
    add_panel_label,
    publication_style,
    save_publication_figure,
)


def draw_importance_bars(
    ax: plt.Axes,
    df_importance: pd.DataFrame,
    feature_col: str = "feature",
    importance_col: str = "importance_pct",
    std_col: Optional[str] = "std_pct",
    base_features: Optional[Sequence[str]] = None,
    target_name: str = "$N_1$",
    target_label: str = "Particle loss",
    panel_label: Optional[str] = None,
    show_legend: bool = True,
    text_fontsize: float = 7.5,
    bar_height: float = 0.65,
    capsize: float = 2.5,
) -> pd.DataFrame:
    """
    Renders horizontal feature importance bar chart distinguishing base and engineered features on Axes.
    """
    if base_features is None:
        base_features = ["Alfa", "Beta", "H1", "H2"]

    # Filter features with negligible importance (< 0.1%)
    df_plot = df_importance[df_importance[importance_col] >= 0.1].copy()
    df_sorted = df_plot.sort_values(by=importance_col, ascending=True).copy()

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
        height=bar_height,
        capsize=capsize,
        error_kw=dict(elinewidth=0.8, ecolor="#222222"),
        zorder=3,
    )

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Percentage annotations next to bars
    max_val = float(df_sorted[importance_col].max()) if len(df_sorted) > 0 else 1.0
    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        val = float(row[importance_col])
        err = float(row[std_col]) if (std_col and std_col in row and pd.notnull(row[std_col])) else 0.0
        if val > 55.0:
            ax.text(
                val - 2.5,
                idx,
                f"{val:.1f}%",
                va="center",
                ha="right",
                fontsize=text_fontsize,
                color="#FFFFFF",
                fontweight="bold",
                zorder=5,
            )
        else:
            x_pos = val + err + 1.2
            ax.text(
                x_pos,
                idx,
                f"{val:.1f}%",
                va="center",
                ha="left",
                fontsize=text_fontsize,
                color="#222222",
                zorder=5,
            )

    tgt_desc = f"{target_label} {target_name}".strip()
    ax.set_xlabel(f"Relative importance for {tgt_desc} [%]")
    ax.set_ylabel("Geometric parameter")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(0, max(max_val * 1.20, max_val + 10.0) if max_val < 55 else 100)

    if show_legend:
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

    return df_sorted


def draw_importance_comparison(
    ax: plt.Axes,
    importance_dict: Dict[str, pd.DataFrame],
    feature_col: str = "feature",
    importance_col: str = "importance_pct",
    panel_label: Optional[str] = "(d)",
    show_legend: bool = True,
    bar_height: float = 0.25,
) -> List[str]:
    """
    Renders grouped horizontal bar chart comparing feature importances across N1, N2, and Delta on Axes.
    """
    all_feats = set()
    for df in importance_dict.values():
        all_feats.update(df[df[importance_col] >= 0.1][feature_col].tolist())

    feat_avg = {}
    for f in all_feats:
        vals = []
        for df in importance_dict.values():
            m = df[df[feature_col] == f]
            vals.append(float(m[importance_col].values[0]) if len(m) > 0 else 0.0)
        feat_avg[f] = float(np.mean(vals))

    sorted_feats = sorted(feat_avg.keys(), key=lambda x: feat_avg[x], reverse=False)
    y_labels = [FEATURE_LABEL_MAP.get(f, f) for f in sorted_feats]

    targets_info = [
        ("N1", "Loss $N_1$", OKABE_ITO[0], ""),
        ("N2", "Capture $N_2$", OKABE_ITO[1], "//"),
        ("Delta", "Advantage $\\Delta$", OKABE_ITO[2], "\\\\"),
    ]

    active_targets = [t for t in targets_info if t[0] in importance_dict]
    n_active = max(len(active_targets), 1)
    n_feats = len(sorted_feats)
    y_indices = np.arange(n_feats)

    for a_idx, (t_key, t_label, t_color, t_hatch) in enumerate(active_targets):
        df_t = importance_dict[t_key]
        vals = []
        for f in sorted_feats:
            m = df_t[df_t[feature_col] == f]
            vals.append(float(m[importance_col].values[0]) if len(m) > 0 else 0.0)

        offset = (a_idx - (n_active - 1) / 2.0) * bar_height
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

    if show_legend:
        ax.legend(loc="lower right", framealpha=0.92)

    if panel_label:
        add_panel_label(ax, panel_label, loc="outside_top_left")

    return sorted_feats


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
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.0, 3.8),
) -> Dict[str, Path]:
    """
    Generates an autonomous feature importance figure for a single target.
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_importance_bars(
            ax=ax,
            df_importance=importance_df,
            feature_col=feature_col,
            importance_col=importance_col,
            std_col=std_col,
            base_features=base_features,
            target_name=target_name,
            target_label=target_label,
            panel_label=panel_label,
            show_legend=True,
            text_fontsize=7.5,
            bar_height=0.65,
            capsize=2.5,
        )
        fig.tight_layout(pad=1.0)
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


def plot_feature_importance_comparison(
    importance_dict: Dict[str, pd.DataFrame],
    feature_col: str = "feature",
    importance_col: str = "importance_pct",
    panel_label: Optional[str] = "(d)",
    figure_name: str = "Fig3d_importance_comparison",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.4, 3.8),
) -> Dict[str, Path]:
    """
    Generates Figure 3d: Comparative feature importance across objectives (N1 vs. N2 vs. Delta).
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_importance_comparison(
            ax=ax,
            importance_dict=importance_dict,
            feature_col=feature_col,
            importance_col=importance_col,
            panel_label=panel_label,
            show_legend=True,
            bar_height=0.25,
        )
        fig.tight_layout(pad=1.0)
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


def plot_feature_importance(
    importance_data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    figure_name: str = "Fig3_feature_importance",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig3",
) -> Dict[str, Any]:
    """
    Generates Figure 3: Geometric Feature Importance (Explainable AI / Permutation Importance).
    Combines 3 single-target panels and 1 multi-target comparative panel into a 2x2 grid.
    """
    saved_all: Dict[str, Any] = {}

    if isinstance(importance_data, pd.DataFrame):
        importance_dict = {"N1": importance_data}
    else:
        importance_dict = importance_data

    target_configs = [
        ("N1", "$N_1$", "Particle loss", "(a)", f"{individual_prefix}a_importance_N1"),
        ("N2", "$N_2$", "Captured particles", "(b)", f"{individual_prefix}b_importance_N2"),
        ("Delta", "$\\Delta$", "Net capture advantage", "(c)", f"{individual_prefix}c_importance_Delta"),
    ]

    # 1. Standalone panel figures
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

        if len(importance_dict) >= 2:
            res_comp = plot_feature_importance_comparison(
                importance_dict=importance_dict,
                panel_label="(d)",
                figure_name=f"{individual_prefix}d_importance_comparison",
                output_dir=output_dir,
                formats=formats,
            )
            saved_all[f"{individual_prefix}d_importance_comparison"] = res_comp

    # 2. Composite 2x2 grid
    with publication_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.8, 7.0))
        axes_flat = axes.flatten()

        for i, (t_key, t_sym, t_name, p_lbl, _) in enumerate(target_configs):
            ax = axes_flat[i]
            if t_key not in importance_dict:
                continue
            draw_importance_bars(
                ax=ax,
                df_importance=importance_dict[t_key],
                target_name=t_sym,
                target_label=t_name,
                panel_label=p_lbl,
                show_legend=True,
                text_fontsize=7.2,
                bar_height=0.65,
                capsize=2.2,
            )

        # Panel (d): Comparative
        ax_d = axes_flat[3]
        draw_importance_comparison(
            ax=ax_d,
            importance_dict=importance_dict,
            panel_label="(d)",
            show_legend=True,
            bar_height=0.25,
        )

        fig.tight_layout(pad=1.4)
        res_grid = save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)
        saved_all["combined_grid"] = res_grid

    return saved_all


__all__ = [
    "draw_importance_bars",
    "draw_importance_comparison",
    "plot_feature_importance_single",
    "plot_feature_importance_comparison",
    "plot_feature_importance",
]
