"""
Unified Publication Figures Generator for Q1 Chemical Engineering Journal:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of Inertial Separators in Chemical Looping Systems'

Generates publication-quality vector (PDF, SVG) and raster (PNG, 300+ DPI) figures:
- Figure 2: Model Diagnostics & Residual Analysis (5-Fold CV OOF)
- Figure 3: Geometric & Aerodynamic Feature Importance (Permutation Importance / XAI)
- Figure 4: Sensitivity Sweep Profiles (1D Dual-Y and 2D Interaction Contours)
- Figure 5: Multi-Objective NSGA-II Convergence & Pareto Optimal Trade-off
- Figure 6: Engineering Validation Case Study (Baseline CFD vs. Pareto Optima)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Sequence, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import joblib

from config.path import (
    figures_dir as default_figures_dir,
    processed_data_dir,
    demo_data_file,
    surrogate_models_dir,
    optimization_results_dir,
)
from src.constants import (
    BASE_FEATURES,
    DEFAULT_PARAM_RANGES,
    DEFAULT_REFERENCE_PARAMS,
    TOTAL_PARTICLES,
)
from src.features import FEATURE_NAMES, create_features
from src.models import evaluate_cv, load_model, predict, train_final_model
from src.visualization.publication_plots import (
    plot_model_diagnostics,
    plot_feature_importance,
    plot_sensitivity_sweeps_1d,
    plot_sensitivity_heatmaps_2d,
    plot_optimization_figure_5,
    plot_case_study_comparison,
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


def _ensure_active_dataset() -> Path:
    """Ensures a valid dataset exists (processed CFD data or demonstration data)."""
    default_data = processed_data_dir / "df_selected.csv"
    if default_data.exists():
        return default_data
    if demo_data_file.exists():
        print(f"[INFO] Using demonstration dataset: {demo_data_file.name}")
        return demo_data_file
    raise FileNotFoundError("Neither processed CFD dataset nor demonstration dataset found.")


def _ensure_active_model():
    """Ensures a trained surrogate model exists, training on available dataset if necessary."""
    try:
        return load_model()
    except FileNotFoundError:
        print("[INFO] Model weights not found. Training surrogate model on active dataset...")
        data_path = _ensure_active_dataset()
        model, _ = train_final_model(data_path=data_path)
        return model


def generate_figure_2(
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> Dict[str, Any]:
    """Generates Figure 2: Model Diagnostics and Residuals (5-Fold CV OOF)."""
    out = Path(output_dir) if output_dir else default_figures_dir
    oof_cache = processed_data_dir / "oof_predictions.csv"

    if oof_cache.exists():
        print(f"[INFO] Loading cached OOF predictions from: {oof_cache.name}")
        oof_df = pd.read_csv(oof_cache)
    else:
        data_file = _ensure_active_dataset()
        print(f"[INFO] Computing 5-fold cross-validation OOF on: {data_file.name}")
        _, df_summary, oof_df = evaluate_cv(data_path=data_file)
        if data_file == processed_data_dir / "df_selected.csv":
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
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    n_repeats: int = 25,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Generates Figure 3: Permutation Feature Importance for N1, N2, and Delta."""
    out = Path(output_dir) if output_dir else default_figures_dir
    data_file = _ensure_active_dataset()
    cache_file = processed_data_dir / "feature_importance_cache.json"

    if cache_file.exists() and data_file == processed_data_dir / "df_selected.csv":
        print(f"[INFO] Loading cached feature importance from: {cache_file.name}")
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        importance_dict = {k: pd.DataFrame(v) for k, v in cache_data.items()}
    else:
        print(f"[INFO] Computing Permutation Feature Importance ({n_repeats} repeats)...")
        df = pd.read_csv(data_file)
        X = df[FEATURE_NAMES]
        y_n1 = df["N1"]
        y_n2 = df["N2"]
        y_delta = df["Delta"]

        model = _ensure_active_model()
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

        if data_file == processed_data_dir / "df_selected.csv":
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
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
    n_points_1d: int = 150,
    grid_size_2d: int = 50,
) -> Dict[str, Any]:
    """Generates Figure 4: Sensitivity 1D sweeps and 2D contour interaction maps."""
    out = Path(output_dir) if output_dir else default_figures_dir
    model = _ensure_active_model()

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
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> Dict[str, Any]:
    """Generates Figure 5: NSGA-II Multi-Objective Convergence and Pareto Front Trade-off."""
    out = Path(output_dir) if output_dir else default_figures_dir
    log_path = optimization_results_dir / "convergence_log.csv"

    # Dynamic search for Pareto front solutions
    pareto_candidates = [
        optimization_results_dir / "pareto_optimal_designs.csv",
        optimization_results_dir / "pareto_front.csv",
        optimization_results_dir / "Test5_Results_src.csv",
    ]
    pareto_candidates.extend(list(optimization_results_dir.glob("pareto_front*.csv")))
    pareto_candidates.extend(list(optimization_results_dir.glob("genetic_results_pareto*.csv")))
    pareto_path = next((p for p in pareto_candidates if p.exists()), None)

    model = _ensure_active_model()

    # If optimization files are missing, run NSGA-II to generate them
    if pareto_path is None or not log_path.exists():
        print("[INFO] Optimization history not found. Running NSGA-II optimization...")
        from src.genetic import run_genetic_optimization
        _, _, pareto_df, log_df = run_genetic_optimization(pop_size=60, n_gen=15)
    else:
        log_df = pd.read_csv(log_path)
        pareto_df = pd.read_csv(pareto_path)

    # Re-evaluate model predictions for Pareto points
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

    # Search for population export files
    pop_candidates = list(optimization_results_dir.glob("population*.csv")) + list(optimization_results_dir.glob("genetic_results_pop*.csv"))
    if pop_candidates:
        df_pop = pd.read_csv(pop_candidates[0])
        all_explored_df = pd.concat([df_explored, df_pop], ignore_index=True)
    else:
        all_explored_df = df_explored

    cluster_labels_map = {
        0: r"Pareto Cluster 1 ($H_2 \approx 59\text{ mm}$)",
        1: r"Pareto Cluster 2 ($H_2 \approx 39\text{ mm}$)",
    }

    print(f"[INFO] Generating Figure 5 into: {out}")
    return plot_optimization_figure_5(
        log_df=log_df,
        all_population_df=all_explored_df,
        pareto_df=pareto_df,
        baseline_point=baseline_point,
        cluster_col="cluster" if "cluster" in pareto_df.columns else None,
        cluster_labels_map=cluster_labels_map,
        figure_name="Fig5_optimization_pareto",
        output_dir=out,
        formats=formats,
        save_individual=True,
        individual_prefix="Fig5",
        show_inset=True,
    )


def generate_figure_6(
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> Dict[str, Any]:
    """Generates Figure 6: Engineering Validation Case Study (Baseline vs. Pareto Optima)."""
    out = Path(output_dir) if output_dir else default_figures_dir
    model = _ensure_active_model()

    geo_baseline = dict(DEFAULT_REFERENCE_PARAMS)
    geo_opt1 = {"Alfa": 43.56, "Beta": 48.24, "H1": 0.0323, "H2": 0.0588}
    geo_opt2 = {"Alfa": 42.97, "Beta": 48.22, "H1": 0.0274, "H2": 0.0386}

    df_variants = pd.DataFrame([geo_baseline, geo_opt1, geo_opt2])
    X_vars = create_features(df_variants)
    n1_vals, n2_vals, delta_vals = predict(model, X_vars)

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
            "description": "Zero-emission optimum (H2 ~ 59 mm)",
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
            "description": "Max-collection optimum (H2 ~ 39 mm)",
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

    comp_df = pd.DataFrame(case_study_records)

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
    output_dir: Optional[Path] = None,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> Dict[str, Any]:
    """Generates all publication figures (Fig 2 - Fig 6) in sequence."""
    out = Path(output_dir) if output_dir else default_figures_dir
    print("=" * 70)
    print("  GENERATING ALL PUBLICATION FIGURES (Fig 2 - Fig 6)  ")
    print(f"  Target directory: {out}")
    print("=" * 70)

    results = {}
    print("\n>>> [1/5] Figure 2: Model Diagnostics")
    results["Fig2"] = generate_figure_2(output_dir=out, formats=formats)

    print("\n>>> [2/5] Figure 3: Feature Importance")
    results["Fig3"] = generate_figure_3(output_dir=out, formats=formats)

    print("\n>>> [3/5] Figure 4: Sensitivity Sweeps & Contours")
    results["Fig4"] = generate_figure_4(output_dir=out, formats=formats)

    print("\n>>> [4/5] Figure 5: NSGA-II Optimization & Pareto Front")
    results["Fig5"] = generate_figure_5(output_dir=out, formats=formats)

    print("\n>>> [5/5] Figure 6: Case Study Validation")
    results["Fig6"] = generate_figure_6(output_dir=out, formats=formats)

    print("\n" + "=" * 70)
    print("  ALL FIGURES GENERATED SUCCESSFULLY!  ")
    print("=" * 70)
    return results


__all__ = [
    "generate_figure_2",
    "generate_figure_3",
    "generate_figure_4",
    "generate_figure_5",
    "generate_figure_6",
    "generate_all_figures",
    "DEFAULT_REFERENCE_PARAMS",
    "DEFAULT_PARAM_RANGES",
]
