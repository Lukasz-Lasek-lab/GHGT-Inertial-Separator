"""
Figure 6: Engineering Validation Case Study (Baseline CFD vs. Pareto Optima).
Provides panel renderers and publication generators for separation efficiencies (eta_1, eta_2)
and net collection advantage (Delta), with dynamic selection from Pareto optimization results and CFD fallback.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import pandas as pd

from config.path import optimization_results_dir
from src.constants import DEFAULT_REFERENCE_PARAMS, TOTAL_PARTICLES
from src.features import create_features
from src.visualization.style import (
    OKABE_ITO,
    add_panel_label,
    publication_style,
    save_publication_figure,
)


def resolve_case_study_data(
    model: Optional[Any] = None,
    pareto_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Dynamically extracts optimal design variants (Opt 1: minimum loss N1, Opt 2: maximum advantage Delta)
    from Pareto results (pareto_optimal_designs.csv) with automatic fallback to historical CFD baseline values.
    """
    geo_baseline = dict(DEFAULT_REFERENCE_PARAMS)
    geo_opt1 = {"Alfa": 43.56, "Beta": 48.24, "H1": 0.0323, "H2": 0.0588}
    geo_opt2 = {"Alfa": 42.97, "Beta": 48.22, "H1": 0.0274, "H2": 0.0386}

    # Search for Pareto optimal designs CSV
    candidates = []
    if pareto_csv_path is not None:
        candidates.append(Path(pareto_csv_path))
    candidates.extend([
        optimization_results_dir / "pareto_optimal_designs.csv",
        optimization_results_dir / "pareto_front.csv",
        optimization_results_dir / "Test5_Results_src.csv",
    ])
    candidates.extend(list(optimization_results_dir.glob("pareto_optimal*.csv")))
    candidates.extend(list(optimization_results_dir.glob("pareto_front*.csv")))

    found_path = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)

    extracted_performance = None
    if found_path is not None:
        try:
            df_pareto = pd.read_csv(found_path)
            if len(df_pareto) >= 2 and all(col in df_pareto.columns for col in ["Alfa", "Beta", "H1", "H2"]):
                n1_col = "N1_pred" if "N1_pred" in df_pareto.columns else ("N1" if "N1" in df_pareto.columns else None)
                delta_col = "Delta_pred" if "Delta_pred" in df_pareto.columns else ("Delta" if "Delta" in df_pareto.columns else None)

                if n1_col and delta_col:
                    idx_opt1 = df_pareto[n1_col].idxmin()
                    idx_opt2 = df_pareto[delta_col].idxmax()
                    row1 = df_pareto.loc[idx_opt1]
                    row2 = df_pareto.loc[idx_opt2]

                    geo_opt1 = {
                        "Alfa": float(row1["Alfa"]),
                        "Beta": float(row1["Beta"]),
                        "H1": float(row1["H1"]),
                        "H2": float(row1["H2"]),
                    }
                    geo_opt2 = {
                        "Alfa": float(row2["Alfa"]),
                        "Beta": float(row2["Beta"]),
                        "H1": float(row2["H1"]),
                        "H2": float(row2["H2"]),
                    }

                    opt1_n1 = float(row1[n1_col])
                    opt1_delta = float(row1[delta_col])
                    opt1_n2 = float(row1["N2_pred"]) if "N2_pred" in row1 else (float(row1["N2"]) if "N2" in row1 else opt1_n1 + opt1_delta)

                    opt2_n1 = float(row2[n1_col])
                    opt2_delta = float(row2[delta_col])
                    opt2_n2 = float(row2["N2_pred"]) if "N2_pred" in row2 else (float(row2["N2"]) if "N2" in row2 else opt2_n1 + opt2_delta)

                    extracted_performance = {
                        "opt1": (opt1_n1, opt1_n2, opt1_delta),
                        "opt2": (opt2_n1, opt2_n2, opt2_delta),
                    }
                    print(f"[INFO] Selected Pareto Optima from {found_path.name}: Opt1 (H2={geo_opt1['H2']:.4f}m), Opt2 (H2={geo_opt2['H2']:.4f}m)")
        except Exception as e:
            print(f"[WARNING] Failed to extract designs from {found_path}: {e}. Utilizing CFD reference constants.")

    df_variants = pd.DataFrame([geo_baseline, geo_opt1, geo_opt2])

    if model is not None:
        from src.models import predict
        X_vars = create_features(df_variants)
        n1_vals, n2_vals, delta_vals = predict(model, X_vars)
    elif extracted_performance is not None:
        # Use performance directly extracted from Pareto CSV
        n1_vals = [242.0, extracted_performance["opt1"][0], extracted_performance["opt2"][0]]
        n2_vals = [5706.0, extracted_performance["opt1"][1], extracted_performance["opt2"][1]]
        delta_vals = [5464.0, extracted_performance["opt1"][2], extracted_performance["opt2"][2]]
    else:
        # Physical reference values when neither model nor CSV is supplied
        n1_vals = [242.0, 0.0, 15.0]
        n2_vals = [5706.0, 5849.0, 5865.0]
        delta_vals = [5464.0, 5849.0, 5850.0]

    h2_opt1_mm = geo_opt1["H2"] * 1000.0
    h2_opt2_mm = geo_opt2["H2"] * 1000.0

    case_study_records = [
        {
            "variant": "Baseline CFD\n(Nominal)",
            "description": "CFD reference geometry",
            "Alfa": geo_baseline["Alfa"],
            "Beta": geo_baseline["Beta"],
            "H1": geo_baseline["H1"],
            "H2": geo_baseline["H2"],
            "N1": float(n1_vals[0]),
            "N2": float(n2_vals[0]),
            "Delta": float(delta_vals[0]),
            "eta_1_pct": (float(n1_vals[0]) / TOTAL_PARTICLES) * 100.0,
            "eta_2_pct": (float(n2_vals[0]) / TOTAL_PARTICLES) * 100.0,
            "eta_delta_pct": (float(delta_vals[0]) / TOTAL_PARTICLES) * 100.0,
        },
        {
            "variant": "Pareto Opt. 1\n(Zero loss)",
            "description": f"Zero-loss optimum (H2 ~ {h2_opt1_mm:.0f} mm)",
            "Alfa": geo_opt1["Alfa"],
            "Beta": geo_opt1["Beta"],
            "H1": geo_opt1["H1"],
            "H2": geo_opt1["H2"],
            "N1": float(n1_vals[1]),
            "N2": float(n2_vals[1]),
            "Delta": float(delta_vals[1]),
            "eta_1_pct": (float(n1_vals[1]) / TOTAL_PARTICLES) * 100.0,
            "eta_2_pct": (float(n2_vals[1]) / TOTAL_PARTICLES) * 100.0,
            "eta_delta_pct": (float(delta_vals[1]) / TOTAL_PARTICLES) * 100.0,
        },
        {
            "variant": "Pareto Opt. 2\n(Max capture)",
            "description": f"Max-collection optimum (H2 ~ {h2_opt2_mm:.0f} mm)",
            "Alfa": geo_opt2["Alfa"],
            "Beta": geo_opt2["Beta"],
            "H1": geo_opt2["H1"],
            "H2": geo_opt2["H2"],
            "N1": float(n1_vals[2]),
            "N2": float(n2_vals[2]),
            "Delta": float(delta_vals[2]),
            "eta_1_pct": (float(n1_vals[2]) / TOTAL_PARTICLES) * 100.0,
            "eta_2_pct": (float(n2_vals[2]) / TOTAL_PARTICLES) * 100.0,
            "eta_delta_pct": (float(delta_vals[2]) / TOTAL_PARTICLES) * 100.0,
        },
    ]

    return pd.DataFrame(case_study_records)


def draw_efficiency_bars(
    ax: plt.Axes,
    comparison_df: pd.DataFrame,
    labels_col: str = "variant",
    loss_col: str = "eta_1_pct",
    capture_col: str = "eta_2_pct",
    panel_label: Optional[str] = "(a)",
    show_legend: bool = True,
    text_fontsize: float = 7.2,
    bar_width: float = 0.30,
) -> None:
    """
    Renders separation efficiency bars (eta_1 loss and eta_2 capture [%]) on Axes.
    """
    x = np.arange(len(comparison_df))
    rects1 = ax.bar(
        x - bar_width / 2,
        comparison_df[loss_col],
        bar_width,
        label=r"Particle loss $\eta_1$",
        color=OKABE_ITO[1],
        edgecolor="#111111",
        linewidth=0.7,
    )
    rects2 = ax.bar(
        x + bar_width / 2,
        comparison_df[capture_col],
        bar_width,
        label=r"Capture efficiency $\eta_2$",
        color=OKABE_ITO[0],
        hatch="//",
        edgecolor="#111111",
        linewidth=0.7,
    )

    for rect in list(rects1) + list(rects2):
        h = float(rect.get_height())
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=text_fontsize,
            fontweight="bold",
        )

    ax.set_ylabel("Separation efficiency [%]", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df[labels_col], fontsize=7.8)
    ax.set_xlim(-0.55, len(comparison_df) - 0.45)
    ax.set_ylim(0, 118)
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if show_legend:
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92, ncol=2)

    if panel_label:
        add_panel_label(ax, panel_label, loc="outside_top_left")


def draw_advantage_bars(
    ax: plt.Axes,
    comparison_df: pd.DataFrame,
    labels_col: str = "variant",
    delta_col: str = "Delta",
    panel_label: Optional[str] = "(b)",
    show_legend: bool = True,
    text_fontsize: float = 7.0,
    bar_width: float = 0.42,
) -> None:
    """
    Renders net collection advantage Delta [particles/s] bars on Axes.
    """
    x = np.arange(len(comparison_df))
    base_delta = float(comparison_df[delta_col].iloc[0]) if len(comparison_df) > 0 else 5464.0

    n_bars = len(comparison_df)
    colors = [OKABE_ITO[3]] + [OKABE_ITO[2]] * max(n_bars - 1, 0)
    hatches = [""] + ["\\\\"] * max(n_bars - 1, 0)

    rects = ax.bar(
        x,
        comparison_df[delta_col],
        bar_width,
        color=colors,
        edgecolor="#111111",
        linewidth=0.7,
    )
    for rect, htc in zip(rects, hatches):
        rect.set_hatch(htc)

    for idx, rect in enumerate(rects):
        h = float(rect.get_height())
        val_str = f"{int(round(h))} N/s"
        if idx == 0:
            gain_str = "(Baseline)"
        else:
            gain = ((h - base_delta) / base_delta) * 100.0 if base_delta > 0 else 0.0
            gain_str = f"(+{gain:.1f}%)"
        ax.annotate(
            f"{val_str}\n{gain_str}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3.0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=text_fontsize,
            fontweight="bold",
            multialignment="center",
        )

    ax.set_ylabel(r"Net capture advantage $\Delta$ [particles/s]", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df[labels_col], fontsize=7.8)
    ax.set_xlim(-0.55, len(comparison_df) - 0.45)
    max_d = float(comparison_df[delta_col].max()) if len(comparison_df) > 0 else 5500.0
    ax.set_ylim(0, max(7800.0, max_d * 1.25))
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    if show_legend:
        custom_handles = [
            Patch(facecolor=OKABE_ITO[3], edgecolor="#111111", linewidth=0.7, label="Baseline design"),
            Patch(facecolor=OKABE_ITO[2], hatch="\\\\", edgecolor="#111111", linewidth=0.7, label="Pareto optimal designs"),
        ]
        ax.legend(handles=custom_handles, loc="upper left", fontsize=7.5, framealpha=0.92, ncol=2)

    if panel_label:
        add_panel_label(ax, panel_label, loc="outside_top_left")


def draw_comparison_bars(
    ax1: plt.Axes,
    ax2: plt.Axes,
    comparison_df: pd.DataFrame,
    labels_col: str = "variant",
    loss_col: str = "eta_1_pct",
    capture_col: str = "eta_2_pct",
    delta_col: str = "Delta",
    panel_labels: Tuple[str, str] = ("(a)", "(b)"),
    show_legend: bool = True,
) -> None:
    """
    Renders both efficiency bars (ax1) and net advantage bars (ax2) side-by-side.
    """
    draw_efficiency_bars(
        ax=ax1,
        comparison_df=comparison_df,
        labels_col=labels_col,
        loss_col=loss_col,
        capture_col=capture_col,
        panel_label=panel_labels[0] if panel_labels else None,
        show_legend=show_legend,
    )
    draw_advantage_bars(
        ax=ax2,
        comparison_df=comparison_df,
        labels_col=labels_col,
        delta_col=delta_col,
        panel_label=panel_labels[1] if len(panel_labels) > 1 else None,
        show_legend=show_legend,
    )


def plot_case_study_efficiency(
    comparison_df: pd.DataFrame,
    labels_col: str = "variant",
    loss_col: str = "eta_1_pct",
    capture_col: str = "eta_2_pct",
    panel_label: Optional[str] = "(a)",
    figure_name: str = "Fig6a_efficiency_comparison",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.5, 3.8),
) -> Dict[str, Path]:
    """
    Generates Figure 6a: Engineering Validation - Separation Efficiencies.
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_efficiency_bars(
            ax=ax,
            comparison_df=comparison_df,
            labels_col=labels_col,
            loss_col=loss_col,
            capture_col=capture_col,
            panel_label=panel_label,
            show_legend=True,
        )
        fig.tight_layout(pad=1.0)
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


def plot_case_study_advantage(
    comparison_df: pd.DataFrame,
    labels_col: str = "variant",
    delta_col: str = "Delta",
    panel_label: Optional[str] = "(b)",
    figure_name: str = "Fig6b_advantage_comparison",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    figsize: Tuple[float, float] = (4.5, 3.8),
) -> Dict[str, Path]:
    """
    Generates Figure 6b: Engineering Validation - Net Collection Advantage Delta.
    """
    with publication_style():
        fig, ax = plt.subplots(figsize=figsize)
        draw_advantage_bars(
            ax=ax,
            comparison_df=comparison_df,
            labels_col=labels_col,
            delta_col=delta_col,
            panel_label=panel_label,
            show_legend=True,
        )
        fig.tight_layout(pad=1.0)
        return save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)


def plot_case_study_comparison(
    comparison_df: Optional[pd.DataFrame] = None,
    labels_col: str = "variant",
    loss_col: str = "eta_1_pct",
    capture_col: str = "eta_2_pct",
    delta_col: str = "Delta",
    figure_name: str = "Fig6_case_study_comparison",
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    save_individual: bool = True,
    individual_prefix: str = "Fig6",
    figsize: Tuple[float, float] = (7.6, 3.8),
) -> Dict[str, Any]:
    """
    Generates complete publication Figure 6:
    Panel (a): Separation efficiencies (loss eta_1 and capture eta_2 [%])
    Panel (b): Net collection advantage Delta [particles/s] with percentage gain over CFD baseline.
    """
    if comparison_df is None:
        comparison_df = resolve_case_study_data()

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

    with publication_style():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        draw_comparison_bars(
            ax1=ax1,
            ax2=ax2,
            comparison_df=comparison_df,
            labels_col=labels_col,
            loss_col=loss_col,
            capture_col=capture_col,
            delta_col=delta_col,
            panel_labels=("(a)", "(b)"),
            show_legend=True,
        )
        fig.tight_layout(pad=1.2)
        res_grid = save_publication_figure(fig, figure_name, output_dir=output_dir, formats=formats)
        saved_all["combined_grid"] = res_grid

    return saved_all


__all__ = [
    "resolve_case_study_data",
    "draw_efficiency_bars",
    "draw_advantage_bars",
    "draw_comparison_bars",
    "plot_case_study_efficiency",
    "plot_case_study_advantage",
    "plot_case_study_comparison",
]
