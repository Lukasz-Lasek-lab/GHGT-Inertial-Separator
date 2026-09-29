"""
Unified Publication Figures Orchestrator for MsCO2limit:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems'

Orchestrates automated end-to-end generation of publication-standard figures:
- Figure 2: Model Diagnostics & Residual Analysis (5-Fold CV OOF)
- Figure 3: Geometric & Aerodynamic Feature Importance (Permutation Importance / XAI)
- Figure 4: Sensitivity Sweep Profiles (1D Dual-Y and 2D Interaction Contours)
- Figure 5: Multi-Objective NSGA-II Convergence & Pareto Optimal Trade-off
- Figure 6: Engineering Validation Case Study (Baseline CFD vs. Pareto Optima)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from config.path import (
    demo_data_file,
    figures_dir as default_figures_dir,
    optimization_results_dir,
    processed_data_dir,
    surrogate_models_dir,
)
from src.constants import (
    BASE_FEATURES,
    DEFAULT_PARAM_RANGES,
    DEFAULT_REFERENCE_PARAMS,
    TOTAL_PARTICLES,
)
from src.features import FEATURE_NAMES, create_features, get_selected_features
from src.models import (
    ensure_features_in_df,
    evaluate_cv,
    load_model,
    predict,
    train_final_model,
)
from src.visualization.fig2_diagnostics import plot_model_diagnostics
from src.visualization.fig3_importance import plot_feature_importance
from src.visualization.fig4_sensitivity import (
    plot_sensitivity_heatmaps_2d,
    plot_sensitivity_sweeps_1d,
)
from src.visualization.fig5_pareto import plot_optimization_figure_5
from src.visualization.fig6_case_study import (
    plot_case_study_comparison,
    resolve_case_study_data,
)


class _N2PredictorWrapper:
    """Wrapper predicting N2 = N1 + Delta for Permutation Feature Importance."""
    def __init__(self, multi_model):
        self.m = multi_model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        raw = self.m.predict(X)
        n1 = np.clip(raw[:, 0], 0, None)
        delta = np.clip(raw[:, 1], 0, None)
        return n1 + delta


def _ensure_active_dataset(use_demo: bool = False) -> Path:
    """
    Ensures a valid dataset exists (processed CFD data or demonstration data).
    Explicitly respects the use_demo flag and guarantees that engineered features are present.
    """
    def _ensure_demo_processed() -> Path:
        demo_processed = processed_data_dir / "df_selected_demo.csv"
        if demo_processed.exists():
            try:
                existing_df = pd.read_csv(demo_processed)
                missing_feats = [c for c in get_selected_features() if c not in existing_df.columns]
                if not missing_feats:
                    return demo_processed
            except Exception:
                pass
        if demo_data_file.exists():
            raw_df = pd.read_csv(demo_data_file)
            feat_df = create_features(raw_df)
            for col in ["N1", "N2", "N3", "Delta"]:
                if col in raw_df.columns:
                    feat_df[col] = raw_df[col]
                elif col == "Delta" and "N1" in raw_df.columns and "N2" in raw_df.columns:
                    feat_df["Delta"] = raw_df["N2"] - raw_df["N1"]
            demo_processed.parent.mkdir(parents=True, exist_ok=True)
            feat_df.to_csv(demo_processed, index=False)
            print(f"[INFO] Created processed demo dataset with engineered features: {demo_processed.name}")
            return demo_processed
        raise FileNotFoundError(f"Demonstration dataset not found at {demo_data_file}")

    if use_demo:
        return _ensure_demo_processed()

    default_data = processed_data_dir / "df_selected.csv"
    if default_data.exists():
        return default_data

    print("[INFO] Proprietary CFD dataset not detected. Utilizing demonstration dataset.")
    return _ensure_demo_processed()


def _ensure_active_model(use_demo: bool = False):
    """
    Ensures a trained surrogate model exists, training on available dataset if necessary.
    """
    try:
        return load_model()
    except FileNotFoundError:
        print("[INFO] Model weights not found. Training surrogate model on active dataset...")
        data_path = _ensure_active_dataset(use_demo=use_demo)
        model, _ = train_final_model(data_path=data_path)
        return model


def generate_figure_2(
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    use_demo: bool = False,
) -> Dict[str, Any]:
    """Generates Figure 2: Model Diagnostics and Residuals (5-Fold CV OOF)."""
    out = Path(output_dir) if output_dir else default_figures_dir
    oof_filename = "oof_predictions_demo.csv" if use_demo else "oof_predictions.csv"
    oof_cache = processed_data_dir / oof_filename

    if oof_cache.exists():
        print(f"[INFO] Loading cached OOF predictions from: {oof_cache.name}")
        oof_df = pd.read_csv(oof_cache)
    else:
        data_file = _ensure_active_dataset(use_demo=use_demo)
        print(f"[INFO] Computing 5-fold cross-validation OOF on: {data_file.name}")
        _, df_summary, oof_df = evaluate_cv(data_path=data_file)
        oof_df.to_csv(oof_cache, index=False)
        print(df_summary.to_string(index=False))

    y_true_df = pd.DataFrame({
        "N1": oof_df["N1_true"],
        "N2": oof_df["N2_true"],
        "Delta": oof_df["Delta_true"],
    })
    y_pred_df = pd.DataFrame({
        "N1": oof_df["N1_pred_oof"],
        "N2": oof_df["N2_pred_oof"],
        "Delta": oof_df["Delta_pred_oof"],
    })

    print(f"[INFO] Generating Figure 2 into: {out}")
    return plot_model_diagnostics(
        y_true_df=y_true_df,
        y_pred_df=y_pred_df,
        figure_name="Fig2_model_diagnostics",
        output_dir=out,
        formats=formats,
    )


def generate_figure_3(
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    n_repeats: int = 25,
    random_state: int = 42,
    use_demo: bool = False,
) -> Dict[str, Any]:
    """Generates Figure 3: Permutation Feature Importance for N1, N2, and Delta."""
    out = Path(output_dir) if output_dir else default_figures_dir
    data_file = _ensure_active_dataset(use_demo=use_demo)
    cache_filename = "feature_importance_cache_demo.json" if use_demo else "feature_importance_cache.json"
    cache_file = processed_data_dir / cache_filename

    if cache_file.exists() and not use_demo:
        print(f"[INFO] Loading cached feature importance from: {cache_file.name}")
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        importance_dict = {k: pd.DataFrame(v) for k, v in cache_data.items()}
    else:
        print(f"[INFO] Computing Permutation Feature Importance ({n_repeats} repeats)...")
        df = pd.read_csv(data_file)
        df, active_feats = ensure_features_in_df(df)
        X = df[active_feats]
        y_n1 = df["N1"]
        y_n2 = df["N2"]
        y_delta = df["Delta"]

        model = _ensure_active_model(use_demo=use_demo)
        est_n1 = model.estimators_[0]
        est_delta = model.estimators_[1]
        est_n2 = _N2PredictorWrapper(model)

        p_n1 = permutation_importance(est_n1, X, y_n1, n_repeats=n_repeats, random_state=random_state, scoring="r2")
        p_n2 = permutation_importance(est_n2, X, y_n2, n_repeats=n_repeats, random_state=random_state, scoring="r2")
        p_delta = permutation_importance(est_delta, X, y_delta, n_repeats=n_repeats, random_state=random_state, scoring="r2")

        importance_dict = {}
        json_cache = {}
        targets_raw = [("N1", p_n1), ("N2", p_n2), ("Delta", p_delta)]

        for t_name, p_res in targets_raw:
            tot_drop = sum([max(0.0, float(m)) for m in p_res.importances_mean])
            rows = []
            for feat in FEATURE_NAMES:
                idx = FEATURE_NAMES.index(feat)
                raw_mean = max(0.0, float(p_res.importances_mean[idx]))
                raw_std = float(p_res.importances_std[idx])
                pct_mean = (raw_mean / tot_drop * 100.0) if tot_drop > 0 else 0.0
                pct_std = (raw_std / tot_drop * 100.0) if tot_drop > 0 else 0.0
                rows.append({
                    "feature": feat,
                    "importance_pct": round(pct_mean, 3),
                    "std_pct": round(pct_std, 3),
                    "raw_drop_mean": round(raw_mean, 5),
                    "raw_drop_std": round(raw_std, 5),
                    "is_base": feat in BASE_FEATURES,
                })
            importance_dict[t_name] = pd.DataFrame(rows)
            json_cache[t_name] = rows

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(json_cache, f, indent=4)

    print(f"[INFO] Generating Figure 3 into: {out}")
    return plot_feature_importance(
        importance_data=importance_dict,
        figure_name="Fig3_feature_importance",
        output_dir=out,
        formats=formats,
        save_individual=True,
        individual_prefix="Fig3",
    )


def generate_figure_4(
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    n_points_1d: int = 150,
    grid_size_2d: int = 50,
    use_demo: bool = False,
) -> Dict[str, Any]:
    """Generates Figure 4: Sensitivity 1D sweeps and 2D contour interaction maps."""
    out = Path(output_dir) if output_dir else default_figures_dir
    model = _ensure_active_model(use_demo=use_demo)

    # 1. 1D Sweeps
    sweep_results = {}
    for param, (p_min, p_max) in DEFAULT_PARAM_RANGES.items():
        vals = np.linspace(p_min, p_max, n_points_1d)
        rows = []
        for val in vals:
            row = DEFAULT_REFERENCE_PARAMS.copy()
            row[param] = float(val)
            rows.append(row)
        df_sweep = pd.DataFrame(rows)
        X_feat = create_features(df_sweep)
        n1_pred, n2_pred, delta_pred = predict(model, X_feat)
        sweep_results[param] = pd.DataFrame({
            param: vals,
            "N1_pred": n1_pred,
            "N2_pred": n2_pred,
            "Delta_pred": delta_pred,
        })

    print(f"[INFO] Generating Figure 4 (1D sweeps) into: {out}")
    res_1d = plot_sensitivity_sweeps_1d(
        sensitivity_data=sweep_results,
        ref_params=DEFAULT_REFERENCE_PARAMS,
        figure_name="Fig4_sensitivity_sweeps_1d",
        output_dir=out,
        formats=formats,
        save_individual=True,
        individual_prefix="Fig4",
    )

    # 2. 2D Contours
    def predict_wrapper(df_in: pd.DataFrame):
        X_feat = create_features(df_in)
        return predict(model, X_feat)

    param_pairs = [
        ("Alfa", "Beta", DEFAULT_PARAM_RANGES["Alfa"], DEFAULT_PARAM_RANGES["Beta"]),
        ("H1", "H2", DEFAULT_PARAM_RANGES["H1"], DEFAULT_PARAM_RANGES["H2"]),
    ]

    print(f"[INFO] Generating Figure 4 (2D contours) into: {out}")
    res_2d = plot_sensitivity_heatmaps_2d(
        predict_fn=predict_wrapper,
        ref_params=DEFAULT_REFERENCE_PARAMS,
        param_pairs=param_pairs,
        grid_size=grid_size_2d,
        figure_name="Fig4_sensitivity_heatmaps_2d",
        output_dir=out,
        formats=formats,
        save_individual=True,
        individual_prefix="Fig4",
    )

    return {"sweeps_1d": res_1d, "contours_2d": res_2d}


def generate_figure_5(
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    use_demo: bool = False,
) -> Dict[str, Any]:
    """Generates Figure 5: NSGA-II Multi-Objective Convergence and Pareto Front Trade-off."""
    out = Path(output_dir) if output_dir else default_figures_dir
    log_path = optimization_results_dir / "convergence_log.csv"

    pareto_candidates = [
        optimization_results_dir / "pareto_optimal_designs.csv",
        optimization_results_dir / "pareto_front.csv",
        optimization_results_dir / "Test5_Results_src.csv",
    ]
    pareto_candidates.extend(list(optimization_results_dir.glob("pareto_front*.csv")))
    pareto_candidates.extend(list(optimization_results_dir.glob("genetic_results_pareto*.csv")))
    pareto_path = next((p for p in pareto_candidates if p.exists() and p.stat().st_size > 0), None)

    model = _ensure_active_model(use_demo=use_demo)

    if pareto_path is None or not log_path.exists():
        print("[INFO] Optimization history not found. Running NSGA-II optimization...")
        from src.genetic import run_genetic_optimization
        _, _, pareto_df, log_df = run_genetic_optimization(pop_size=60, n_gen=15)
    else:
        log_df = pd.read_csv(log_path)
        pareto_df = pd.read_csv(pareto_path)

    X_pareto = create_features(pareto_df[["Alfa", "Beta", "H1", "H2"]])
    n1_par, n2_par, delta_par = predict(model, X_pareto)
    pareto_df["N1_pred"] = n1_par
    pareto_df["N2_pred"] = n2_par
    pareto_df["Delta_pred"] = delta_par

    # Baseline CFD reference point
    df_ref = pd.DataFrame([DEFAULT_REFERENCE_PARAMS])
    X_ref = create_features(df_ref)
    n1_cfd, n2_cfd, delta_cfd = predict(model, X_ref)
    baseline_point = {
        "N1": float(n1_cfd[0]),
        "N2": float(n2_cfd[0]),
        "Delta": float(delta_cfd[0]),
    }

    # Design space exploration background cloud
    np.random.seed(42)
    n_sample = 1200
    df_explored = pd.DataFrame({
        "Alfa": np.random.uniform(DEFAULT_PARAM_RANGES["Alfa"][0], DEFAULT_PARAM_RANGES["Alfa"][1], n_sample),
        "Beta": np.random.uniform(DEFAULT_PARAM_RANGES["Beta"][0], DEFAULT_PARAM_RANGES["Beta"][1], n_sample),
        "H1": np.random.uniform(DEFAULT_PARAM_RANGES["H1"][0], DEFAULT_PARAM_RANGES["H1"][1], n_sample),
        "H2": np.random.uniform(DEFAULT_PARAM_RANGES["H2"][0], DEFAULT_PARAM_RANGES["H2"][1], n_sample),
    })
    X_explored = create_features(df_explored)
    n1_exp, n2_exp, delta_exp = predict(model, X_explored)
    df_explored["N1_pred"] = n1_exp
    df_explored["N2_pred"] = n2_exp
    df_explored["Delta_pred"] = delta_exp

    pop_candidates = list(optimization_results_dir.glob("population*.csv")) + list(optimization_results_dir.glob("genetic_results_pop*.csv"))
    if pop_candidates:
        df_pop = pd.read_csv(pop_candidates[0])
        all_explored_df = pd.concat([df_explored, df_pop], ignore_index=True)
    else:
        all_explored_df = df_explored

    print(f"[INFO] Generating Figure 5 into: {out}")
    return plot_optimization_figure_5(
        log_df=log_df,
        all_population_df=all_explored_df,
        pareto_df=pareto_df,
        baseline_point=baseline_point,
        cluster_col="cluster" if "cluster" in pareto_df.columns else None,
        figure_name="Fig5_optimization_pareto",
        output_dir=out,
        formats=formats,
        save_individual=True,
        individual_prefix="Fig5",
        show_inset=True,
    )


def generate_figure_6(
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    use_demo: bool = False,
) -> Dict[str, Any]:
    """Generates Figure 6: Engineering Validation Case Study (Baseline vs. Pareto Optima)."""
    out = Path(output_dir) if output_dir else default_figures_dir
    model = _ensure_active_model(use_demo=use_demo)

    comp_df = resolve_case_study_data(model=model)

    print(f"[INFO] Generating Figure 6 into: {out}")
    return plot_case_study_comparison(
        comparison_df=comp_df,
        labels_col="variant",
        loss_col="eta_1_pct",
        capture_col="eta_2_pct",
        delta_col="Delta",
        figure_name="Fig6_case_study_comparison",
        output_dir=out,
        formats=formats,
        save_individual=True,
        individual_prefix="Fig6",
        figsize=(7.6, 3.8),
    )


def generate_all_figures(
    output_dir: Optional[Union[str, Path]] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    use_demo: bool = False,
    n_points_1d: int = 100,
) -> Dict[str, Any]:
    """Generates all publication figures (Fig 2 - Fig 6) in sequence."""
    out = Path(output_dir) if output_dir else default_figures_dir
    print("=" * 70)
    print("  GENERATING ALL PUBLICATION FIGURES (Fig 2 - Fig 6)  ")
    print(f"  Target directory: {out} (Mode: {'DEMO' if use_demo else 'FULL'})")
    print("=" * 70)

    results = {}
    print("\n>>> [1/5] Figure 2: Model Diagnostics")
    results["Fig2"] = generate_figure_2(output_dir=out, formats=formats, use_demo=use_demo)

    print("\n>>> [2/5] Figure 3: Feature Importance")
    results["Fig3"] = generate_figure_3(output_dir=out, formats=formats, use_demo=use_demo)

    print("\n>>> [3/5] Figure 4: Sensitivity Sweeps & Contours")
    results["Fig4"] = generate_figure_4(
        output_dir=out,
        formats=formats,
        n_points_1d=n_points_1d,
        use_demo=use_demo,
    )

    print("\n>>> [4/5] Figure 5: NSGA-II Optimization & Pareto Front")
    results["Fig5"] = generate_figure_5(output_dir=out, formats=formats, use_demo=use_demo)

    print("\n>>> [5/5] Figure 6: Case Study Validation")
    results["Fig6"] = generate_figure_6(output_dir=out, formats=formats, use_demo=use_demo)

    print("\n" + "=" * 70)
    print("  ALL FIGURES GENERATED SUCCESSFULLY!  ")
    print("=" * 70)
    return results


def main():
    """Command-line entry point for Figure Orchestrator."""
    parser = argparse.ArgumentParser(
        description="Publication Figure Generator for MsCO2limit Scientific Framework."
    )
    parser.add_argument(
        "--figure",
        type=int,
        choices=[2, 3, 4, 5, 6],
        help="Generate a specific publication figure (2, 3, 4, 5, or 6).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all publication figures (Fig 2 - Fig 6).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Utilize synthetic benchmark dataset regardless of proprietary data presence.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom destination directory for generated figures.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        help="Image formats to export (e.g. png pdf svg).",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else default_figures_dir
    formats = tuple(args.formats)

    if args.figure == 2:
        generate_figure_2(output_dir=out_dir, formats=formats, use_demo=args.demo)
    elif args.figure == 3:
        generate_figure_3(output_dir=out_dir, formats=formats, use_demo=args.demo)
    elif args.figure == 4:
        generate_figure_4(output_dir=out_dir, formats=formats, use_demo=args.demo)
    elif args.figure == 5:
        generate_figure_5(output_dir=out_dir, formats=formats, use_demo=args.demo)
    elif args.figure == 6:
        generate_figure_6(output_dir=out_dir, formats=formats, use_demo=args.demo)
    elif args.all or args.figure is None:
        generate_all_figures(output_dir=out_dir, formats=formats, use_demo=args.demo)


if __name__ == "__main__":
    main()


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
