"""
Automated Feature Engineering and Selection module for the MsCO2limit project.
Employs an ensemble committee of estimators (Random Forest, f-regression, LassoCV)
to evaluate candidate polynomial, trigonometric, logarithmic, and interaction terms
against multi-objective targets (particle loss N1 and net collection advantage Delta).

Selected features are persisted to config/selected_features.json, and the engineered
dataset is stored in data/processed/df_selected.csv.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
from sklearn.linear_model import LassoCV

from config.path import config_dir, processed_data_dir, raw_data_dir, demo_data_file
from src.constants import BASE_FEATURES, TARGET_NAMES, TOTAL_PARTICLES
from src.features.registry import FeatureRegistry


def load_raw_dataset(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads raw experimental or CFD simulation dataset (Excel or CSV) and normalizes column headers.
    """
    if dataset_path is None:
        default_excel = raw_data_dir / "inertial_separator_cfd.xlsx"
        legacy_excel = raw_data_dir / "Dane_T5.xlsx"
        if default_excel.exists():
            dataset_path = default_excel
        elif legacy_excel.exists():
            dataset_path = legacy_excel
        elif demo_data_file.exists():
            dataset_path = demo_data_file
        else:
            raise FileNotFoundError(
                f"Raw dataset not found at '{default_excel}'. Please refer to the Data Availability "
                "Statement in README.md or run with '--demo' for synthetic demonstration."
            )

    if not dataset_path.exists():
        raise FileNotFoundError(f"Raw dataset file not found: {dataset_path}")

    if dataset_path.suffix in [".xlsx", ".xls"]:
        df_raw = pd.read_excel(dataset_path)
    else:
        df_raw = pd.read_csv(dataset_path)

    # Normalize column names regardless of case and degree symbols
    col_map = {}
    for col in df_raw.columns:
        col_clean = str(col).strip()
        col_lower = col_clean.lower()
        if col_clean in ["Alfa", "Beta", "H1", "H2", "N1", "N2", "N3", "Delta", "eta_1_pct", "eta_2_pct"]:
            continue
        if any(term in col_lower for term in ["cubed", "squared", "plus", "minus", "div", "log", "sin", "cos"]):
            continue
        if "alfa" in col_lower or "alpha" in col_lower:
            col_map[col] = "Alfa"
        elif "beta" in col_lower:
            col_map[col] = "Beta"
        elif "h1" in col_lower:
            col_map[col] = "H1"
        elif "h2" in col_lower:
            col_map[col] = "H2"
        elif "n1" in col_lower or "loss" in col_lower:
            col_map[col] = "N1"
        elif "n2" in col_lower or "capture" in col_lower:
            col_map[col] = "N2"
        elif "n3" in col_lower or "uncollected" in col_lower or "particle" in col_lower:
            col_map[col] = "N3"

    df = df_raw.rename(columns=col_map)
    required = ["Alfa", "Beta", "H1", "H2", "N1", "N2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"File {dataset_path} is missing required columns: {missing}")

    if "N3" not in df.columns:
        df["N3"] = TOTAL_PARTICLES - df["N1"] - df["N2"]

    # Calculate net separation advantage Delta = N2 - N1
    df["Delta"] = df["N2"] - df["N1"]

    return df


def generate_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands base geometric inputs into a rich aerodynamic candidate feature space
    via the centralized declarative FeatureRegistry.
    Guarantees SSOT parity across all pipeline stages without memory fragmentation.
    """
    return FeatureRegistry.compute_all(df)


def select_best_features(
    X_cand: pd.DataFrame,
    y_n1: pd.Series,
    y_delta: pd.Series,
    max_engineered_features: int = 7,
    random_state: int = 42,
) -> List[str]:
    """
    Performs multi-objective feature selection:
    1. Variance filtering (VarianceThreshold)
    2. Random Forest Regressor Permutation Importance for N1 and Delta
    3. Univariate feature scoring (f_regression / SelectKBest)
    4. L1 Regularization (LassoCV)
    5. Rank aggregation and collinearity pruning (r > 0.985).
    """
    candidate_cols = [c for c in X_cand.columns if c not in BASE_FEATURES]

    # 1. Variance threshold filter
    vt = VarianceThreshold(threshold=1e-5)
    vt.fit(X_cand[candidate_cols])
    valid_candidates = [c for c, sup in zip(candidate_cols, vt.get_support()) if sup]

    X_sub = X_cand[valid_candidates]

    # Feature scoring dictionary
    feature_scores: Dict[str, float] = {c: 0.0 for c in valid_candidates}

    for target_name, y in [("N1", y_n1), ("Delta", y_delta)]:
        y_std = (y - y.mean()) / (y.std() + 1e-6)

        # 2. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=250, random_state=random_state, n_jobs=-1)
        rf.fit(X_sub, y_std)
        importances = rf.feature_importances_
        imp_norm = importances / (importances.sum() + 1e-9)
        for c, score in zip(valid_candidates, imp_norm):
            feature_scores[c] += float(score) * 2.0

        # 3. f_regression (SelectKBest)
        f_vals, _ = f_regression(X_sub, y_std)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        f_norm = f_vals / (f_vals.sum() + 1e-9)
        for c, score in zip(valid_candidates, f_norm):
            feature_scores[c] += float(score) * 1.0

        # 4. LassoCV
        try:
            lasso = LassoCV(cv=5, random_state=random_state, max_iter=2000).fit(X_sub, y_std)
            lasso_weights = np.abs(lasso.coef_)
            if lasso_weights.sum() > 0:
                l_norm = lasso_weights / lasso_weights.sum()
                for c, score in zip(valid_candidates, l_norm):
                    feature_scores[c] += float(score) * 1.5
        except Exception:
            pass

    # Sort features by aggregated ensemble score
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Candidate Engineered Features Ranking (Top 15) ---")
    for rank, (feat, score) in enumerate(sorted_features[:15], 1):
        print(f" {rank:2d}. {feat:<20} score: {score:.4f}")

    # Select top features avoiding pairwise collinearity > 0.985
    selected_engineered: List[str] = []
    corr_matrix = X_cand[valid_candidates].corr().abs()

    for feat, _ in sorted_features:
        if len(selected_engineered) >= max_engineered_features:
            break
        too_correlated = False
        for chosen in selected_engineered:
            if corr_matrix.loc[feat, chosen] > 0.985:
                too_correlated = True
                break
        if not too_correlated:
            selected_engineered.append(feat)

    # Total feature set: 4 base design parameters + top non-redundant engineered features
    final_features = BASE_FEATURES + selected_engineered
    return final_features


def run_feature_selection(
    raw_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    max_engineered: int = 7,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Main feature selection execution workflow:
    1. Loads dataset
    2. Generates candidate aerodynamic features
    3. Executes multi-objective ensemble selection
    4. Exports selected_features.json
    5. Saves transformed training dataset to df_selected.csv
    """
    if raw_path is None:
        default_raw = raw_data_dir / "inertial_separator_cfd.xlsx"
        legacy_raw = raw_data_dir / "Dane_T5.xlsx"
        if default_raw.exists():
            raw_path = default_raw
        elif legacy_raw.exists():
            raw_path = legacy_raw
        else:
            raw_path = demo_data_file

    if output_csv_path is None:
        output_csv_path = processed_data_dir / "df_selected.csv"
    if json_path is None:
        json_path = config_dir / "selected_features.json"

    print(f"[INFO] Loading dataset from: {raw_path}")
    df_raw = load_raw_dataset(raw_path)

    print(f"[INFO] Samples: {len(df_raw)}, Particle balance (N1+N2+N3): {(df_raw['N1']+df_raw['N2']+df_raw['N3']).unique()}")
    print("[INFO] Generating candidate geometric and aerodynamic features...")
    X_cand = generate_candidate_features(df_raw)
    print(f"[INFO] Generated {len(X_cand.columns)} candidate features.")

    print("[INFO] Performing ensemble feature selection (Random Forest, f-regression, Lasso)...")
    selected_features = select_best_features(
        X_cand,
        y_n1=df_raw["N1"],
        y_delta=df_raw["Delta"],
        max_engineered_features=max_engineered,
    )

    print(f"\n[OK] Selected feature set ({len(selected_features)}): {selected_features}")

    # Save metadata JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    features_meta = {
        "dataset_source": str(raw_path.name),
        "total_particles": TOTAL_PARTICLES,
        "base_features": BASE_FEATURES,
        "engineered_features": [f for f in selected_features if f not in BASE_FEATURES],
        "all_selected_features": selected_features,
        "target_names": TARGET_NAMES,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features_meta, f, indent=4)
    print(f"[SAVE] Selected feature metadata saved to: {json_path}")

    # Prepare DataFrame for surrogate modeling
    df_out = X_cand[selected_features].copy()
    df_out["N1"] = df_raw["N1"]
    df_out["N2"] = df_raw["N2"]
    df_out["N3"] = df_raw["N3"]
    df_out["Delta"] = df_raw["Delta"]
    df_out["eta_1_pct"] = (df_raw["N1"] / TOTAL_PARTICLES) * 100.0
    df_out["eta_2_pct"] = (df_raw["N2"] / TOTAL_PARTICLES) * 100.0

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)
    print(f"[SAVE] Processed dataset saved to: {output_csv_path} (shape: {df_out.shape})")

    return selected_features, df_out


if __name__ == "__main__":
    run_feature_selection()
