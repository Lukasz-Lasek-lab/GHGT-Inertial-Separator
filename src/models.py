"""
Surrogate Modeling and ML Management module for the MsCO2limit project.
Handles multi-target regression using HistGradientBoostingRegressor within MultiOutputRegressor,
hyperparameter tuning via Optuna with caching in config/best_params.json, full-dataset training,
5-fold cross-validation (CV), serialization, and physically constrained inference.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from config.path import (
    config_dir,
    surrogate_models_dir,
    diagnostic_plots_dir,
    processed_data_dir,
    evaluation_results_dir,
    demo_data_file,
)
from src.constants import TARGET_NAMES, TOTAL_PARTICLES
from src.features import (
    FEATURE_NAMES,
    create_features,
    get_selected_features,
    prepare_targets,
)

# Default fallback hyperparameters
DEFAULT_FALLBACK_PARAMS: Dict[str, Any] = {
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_samples_leaf": 3,
    "max_leaf_nodes": 64,
    "l2_regularization": 0.5,
    "max_iter": 1000,
    "max_bins": 192,
    "n_iter_no_change": 50,
    "random_state": 42,
}


def load_best_params(json_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads optimized hyperparameters from config/best_params.json.
    Returns fallback hyperparameters if configuration file is not present.
    """
    if json_path is None:
        json_path = config_dir / "best_params.json"

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                params = json.load(f)
                if isinstance(params, dict) and "learning_rate" in params:
                    return params
        except Exception as e:
            print(f"[WARN] Failed to load {json_path}: {e}. Utilizing default parameters.")

    return DEFAULT_FALLBACK_PARAMS.copy()


def save_best_params(params: Dict[str, Any], json_path: Optional[Path] = None) -> Path:
    """
    Saves optimized hyperparameters to JSON file.
    """
    if json_path is None:
        json_path = config_dir / "best_params.json"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
    print(f"[SAVE] Hyperparameters saved to: {json_path}")
    return json_path


def get_best_params(reload: bool = False, json_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Lazy getter for surrogate model hyperparameters.
    Avoids stale configuration across pipeline steps or hyperparameter tuning runs.
    """
    global BEST_PARAMS
    if reload or json_path is not None or not BEST_PARAMS:
        BEST_PARAMS = load_best_params(json_path=json_path)
    return BEST_PARAMS.copy()


# Global dictionary of active hyperparameters (maintained for backward compatibility)
BEST_PARAMS: Dict[str, Any] = load_best_params()


def build_regressor(params: Optional[Dict[str, Any]] = None) -> MultiOutputRegressor:
    """
    Constructs MultiOutputRegressor instance wrapping HistGradientBoostingRegressor.
    Pulls defaults from get_best_params() if not provided.
    """
    model_params = get_best_params()
    if params:
        model_params.update(params)

    # Ensure integer parameter casting for scikit-learn compatibility
    if "max_depth" in model_params and model_params["max_depth"] is not None:
        model_params["max_depth"] = int(model_params["max_depth"])
    if "min_samples_leaf" in model_params:
        model_params["min_samples_leaf"] = int(model_params["min_samples_leaf"])
    if "max_leaf_nodes" in model_params and model_params["max_leaf_nodes"] is not None:
        model_params["max_leaf_nodes"] = int(model_params["max_leaf_nodes"])
    if "max_iter" in model_params:
        model_params["max_iter"] = int(model_params["max_iter"])
    if "max_bins" in model_params:
        model_params["max_bins"] = int(model_params["max_bins"])
    if "random_state" in model_params:
        model_params["random_state"] = int(model_params["random_state"])
    if "early_stopping" not in model_params:
        model_params["early_stopping"] = True

    base_regressor = HistGradientBoostingRegressor(**model_params)
    return MultiOutputRegressor(base_regressor, n_jobs=1)


def ensure_features_in_df(
    df: pd.DataFrame, feature_names: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Ensures all requested feature columns exist in DataFrame, computing missing ones if needed.
    Guarantees robustness when loading raw datasets or older preprocessed CSVs.
    """
    if feature_names is None:
        feature_names = get_selected_features()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        df_feats = create_features(df, feature_names=feature_names)
        df_out = df.copy()
        for col in feature_names:
            df_out[col] = df_feats[col]
        return df_out, feature_names
    return df, feature_names


def tune_hyperparameters(
    data_path: Optional[Path] = None,
    n_trials: int = 300,
    save_json_path: Optional[Path] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes automated Bayesian hyperparameter tuning via Optuna.
    Minimizes normalized joint prediction loss for N1 and Delta across 5-fold cross-validation.
    Persists optimal configuration to config/best_params.json.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if data_path is None:
        default_data = processed_data_dir / "df_selected.csv"
        data_path = default_data if default_data.exists() else demo_data_file
    if save_json_path is None:
        save_json_path = config_dir / "best_params.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset for hyperparameter tuning not found: {data_path}")

    df = pd.read_csv(data_path)
    df, feature_names = ensure_features_in_df(df)
    X = df[feature_names].values
    y_targets = prepare_targets(df)
    y = y_targets.values

    std_n1 = float(np.std(y[:, 0])) if np.std(y[:, 0]) > 0 else 1.0
    std_delta = float(np.std(y[:, 1])) if np.std(y[:, 1]) > 0 else 1.0

    print(f"[OPTUNA] Starting surrogate hyperparameter optimization ({n_trials} trials)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.25, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 25),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 120, step=5),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 3.0, step=0.1),
            "max_iter": trial.suggest_int("max_iter", 150, 1500, step=50),
            "max_bins": trial.suggest_categorical("max_bins", [64, 128, 192, 255]),
            "n_iter_no_change": trial.suggest_int("n_iter_no_change", 20, 80, step=10),
            "random_state": random_state,
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        scores = []

        for train_idx, test_idx in cv.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            reg = MultiOutputRegressor(HistGradientBoostingRegressor(**params), n_jobs=1)
            reg.fit(X_tr, y_tr)
            pred = reg.predict(X_te)

            n1_pred = np.clip(pred[:, 0], 0, None)
            delta_pred = np.clip(pred[:, 1], 0, None)

            n1_true = y_te[:, 0]
            delta_true = y_te[:, 1]

            mae_n1 = mean_absolute_error(n1_true, n1_pred) / std_n1
            mae_delta = mean_absolute_error(delta_true, delta_pred) / std_delta
            rmse_n1 = np.sqrt(mean_squared_error(n1_true, n1_pred)) / std_n1
            rmse_delta = np.sqrt(mean_squared_error(delta_true, delta_pred)) / std_delta

            fold_loss = 0.4 * (mae_n1 + mae_delta) + 0.1 * (rmse_n1 + rmse_delta)
            scores.append(fold_loss)

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_p = study.best_trial.params
    best_p["random_state"] = random_state

    print("\n" + "=" * 60)
    print("OPTUNA: Optimal Hyperparameters Discovered:")
    print("-" * 60)
    for k, v in best_p.items():
        print(f"  {k:20s}: {v}")
    print(f"  Best 5-Fold CV Loss : {study.best_value:.5f}")
    print("=" * 60)

    save_best_params(best_p, save_json_path)

    global BEST_PARAMS
    BEST_PARAMS = best_p
    return best_p


def train_model(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.DataFrame, np.ndarray],
    params: Optional[Dict[str, Any]] = None,
) -> MultiOutputRegressor:
    """
    Trains surrogate MultiOutputRegressor on input features X and multi-targets y.
    """
    model = build_regressor(params)
    model.fit(X, y)
    return model


def train_final_model(
    data_path: Optional[Path] = None,
    save_path: Optional[Path] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[MultiOutputRegressor, Path]:
    """
    Trains final production surrogate model on 100% available data
    using optimized hyperparameters and serializes model to disk.
    """
    if data_path is None:
        default_data = processed_data_dir / "df_selected.csv"
        data_path = default_data if default_data.exists() else demo_data_file
    if save_path is None:
        save_path = surrogate_models_dir / "surrogate_regressor.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df, feature_names = ensure_features_in_df(df)
    X = df[feature_names]
    y = prepare_targets(df)

    if params is None:
        params = get_best_params()

    model = train_model(X, y, params=params)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    # Save legacy copy for backward compatibility with deprecation status
    legacy_save = surrogate_models_dir / "Final_Model.joblib"
    if legacy_save.resolve() != save_path.resolve():
        joblib.dump(model, legacy_save)

    print(f"[OK] Production surrogate model trained on {len(df)} samples and saved to: {save_path}")
    return model, save_path


def load_model(model_path: Optional[Path] = None) -> MultiOutputRegressor:
    """
    Loads pre-trained surrogate model from disk.
    Standard path is surrogate_regressor.joblib. Falls back to legacy Final_Model.joblib.
    """
    if model_path is None:
        canonical_path = surrogate_models_dir / "surrogate_regressor.joblib"
        legacy_path = surrogate_models_dir / "Final_Model.joblib"
        if canonical_path.exists():
            model_path = canonical_path
        elif legacy_path.exists():
            import warnings
            warnings.warn(
                f"Loading legacy model artifact from {legacy_path.name}. "
                "Standard canonical model artifact is surrogate_regressor.joblib.",
                DeprecationWarning,
                stacklevel=2,
            )
            model_path = legacy_path
        else:
            model_path = canonical_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pre-trained surrogate model weights not found at: {model_path}.\n"
            "Please train the model first ('python -m src --train --demo') or consult the "
            "Data and Model Availability Statement in README.md."
        )

    model = joblib.load(model_path)
    return model


def predict(
    model: MultiOutputRegressor,
    X: Union[pd.DataFrame, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs inference with physical consistency constraints:
    - N1 >= 0 (non-negative particle loss)
    - Delta >= 0 (net capture advantage)
    - N2 = N1 + Delta (captured carrier particles)
    
    Returns:
        Tuple of (N1_pred, N2_pred, Delta_pred)
    """
    if isinstance(X, pd.DataFrame):
        if all(col in X.columns for col in FEATURE_NAMES):
            X = X[FEATURE_NAMES]

    raw_pred = model.predict(X)

    N1_pred = np.clip(raw_pred[:, 0], 0, TOTAL_PARTICLES)
    Delta_pred = np.clip(raw_pred[:, 1], 0, TOTAL_PARTICLES)
    N2_pred = N1_pred + Delta_pred

    return N1_pred, N2_pred, Delta_pred


def plot_actual_vs_predicted(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    title_prefix: str = "Predictions vs. Actual (OOF 5-Fold CV)",
) -> None:
    """
    Deprecated facade delegating to src.visualization.fig2_diagnostics.plot_model_diagnostics.
    Eliminates direct matplotlib dependency from core ML logic.
    """
    import warnings
    warnings.warn(
        "plot_actual_vs_predicted in src.models is deprecated. "
        "Use src.visualization.fig2_diagnostics.plot_model_diagnostics instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.visualization.fig2_diagnostics import plot_model_diagnostics

    output_dir = save_path.parent if save_path is not None else None
    figure_name = save_path.stem if save_path is not None else "Fig2_model_diagnostics"
    formats = (save_path.suffix.lstrip(".").lower() or "png",) if save_path is not None else ("png",)

    plot_model_diagnostics(
        y_true_df=y_true_df,
        y_pred_df=y_pred_df,
        figure_name=figure_name,
        output_dir=output_dir,
        formats=formats,
        save_individual=False,
    )


def evaluate_cv(
    data_path: Optional[Path] = None,
    n_splits: int = 5,
    random_state: int = 42,
    save_plot_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes 5-fold cross-validation, aggregates Out-Of-Fold (OOF) predictions,
    and returns (df_folds, df_summary, oof_df).
    """
    if data_path is None:
        default_data = processed_data_dir / "df_selected.csv"
        data_path = default_data if default_data.exists() else demo_data_file

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}.\n"
            "Run with '--demo' for synthetic demonstration or see README.md."
        )

    df = pd.read_csv(data_path)
    df, feature_names = ensure_features_in_df(df)
    X = df[feature_names].values
    y_targets = prepare_targets(df)
    y = y_targets.values

    oof_n1 = np.zeros(len(df))
    oof_n2 = np.zeros(len(df))
    oof_delta = np.zeros(len(df))

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = build_regressor()

    fold_results = []
    for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        N1_p, N2_p, Delta_p = predict(model, X_te)

        oof_n1[test_idx] = N1_p
        oof_n2[test_idx] = N2_p
        oof_delta[test_idx] = Delta_p

        N1_true = y_te[:, 0]
        Delta_true = y_te[:, 1]
        N2_true = N1_true + Delta_true

        fold_results.append({
            "fold": i + 1,
            "MAE_N1": mean_absolute_error(N1_true, N1_p),
            "RMSE_N1": np.sqrt(mean_squared_error(N1_true, N1_p)),
            "R2_N1": r2_score(N1_true, N1_p),
            "MAE_N2": mean_absolute_error(N2_true, N2_p),
            "RMSE_N2": np.sqrt(mean_squared_error(N2_true, N2_p)),
            "R2_N2": r2_score(N2_true, N2_p),
            "MAE_Delta": mean_absolute_error(Delta_true, Delta_p),
            "RMSE_Delta": np.sqrt(mean_squared_error(Delta_true, Delta_p)),
            "R2_Delta": r2_score(Delta_true, Delta_p),
        })

    df_folds = pd.DataFrame(fold_results)

    summary = []
    for col in df_folds.columns:
        if col != "fold":
            summary.append({
                "metric": col,
                "mean": df_folds[col].mean(),
                "std": df_folds[col].std(),
            })
    df_summary = pd.DataFrame(summary)

    real_n1 = df["N1"].values
    real_n2 = df["N2"].values
    real_delta = real_n2 - real_n1

    y_true_df = pd.DataFrame({"N1": real_n1, "N2": real_n2, "Delta": real_delta})
    y_pred_df = pd.DataFrame({"N1": oof_n1, "N2": oof_n2, "Delta": oof_delta})

    oof_df = pd.DataFrame({
        "N1_true": real_n1,
        "N1_pred_oof": oof_n1,
        "N2_true": real_n2,
        "N2_pred_oof": oof_n2,
        "Delta_true": real_delta,
        "Delta_pred_oof": oof_delta,
    })

    if save_plot_path is not None:
        plot_actual_vs_predicted(
            y_true_df,
            y_pred_df,
            save_path=save_plot_path,
            title_prefix=f"Surrogate Model ({len(df)} samples): Actual vs. OOF CV Predictions",
        )

    return df_folds, df_summary, oof_df


def evaluate_train_test(
    data_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_plot_path: Optional[Path] = None,
) -> Tuple[MultiOutputRegressor, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Performs train/test split evaluation (default: 80/20 train/test split).
    """
    if data_path is None:
        default_data = processed_data_dir / "df_selected.csv"
        data_path = default_data if default_data.exists() else demo_data_file

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    df = pd.read_csv(data_path)
    df, feature_names = ensure_features_in_df(df)
    X = df[feature_names]
    y_targets = prepare_targets(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_targets, test_size=test_size, random_state=random_state
    )

    model = train_model(X_train, y_train)
    N1_pred, N2_pred, Delta_pred = predict(model, X_test)

    real_N1 = y_test["N1"].values
    real_Delta = y_test["Delta"].values
    real_N2 = real_N1 + real_Delta

    metrics = {
        "MAE_N1": mean_absolute_error(real_N1, N1_pred),
        "RMSE_N1": float(np.sqrt(mean_squared_error(real_N1, N1_pred))),
        "R2_N1": r2_score(real_N1, N1_pred),
        "MAE_N2": mean_absolute_error(real_N2, N2_pred),
        "RMSE_N2": float(np.sqrt(mean_squared_error(real_N2, N2_pred))),
        "R2_N2": r2_score(real_N2, N2_pred),
        "MAE_Delta": mean_absolute_error(real_Delta, Delta_pred),
        "RMSE_Delta": float(np.sqrt(mean_squared_error(real_Delta, Delta_pred))),
        "R2_Delta": r2_score(real_Delta, Delta_pred),
    }

    y_true_df = pd.DataFrame({"N1": real_N1, "N2": real_N2, "Delta": real_Delta})
    y_pred_df = pd.DataFrame({"N1": N1_pred, "N2": N2_pred, "Delta": Delta_pred})

    if save_plot_path is not None:
        plot_actual_vs_predicted(
            y_true_df,
            y_pred_df,
            save_path=save_plot_path,
            title_prefix=f"Surrogate Model: Test Set Evaluation ({int(test_size * 100)}% held-out)",
        )

    return model, metrics, y_true_df, y_pred_df


__all__ = [
    "DEFAULT_FALLBACK_PARAMS",
    "load_best_params",
    "save_best_params",
    "get_best_params",
    "BEST_PARAMS",
    "build_regressor",
    "tune_hyperparameters",
    "train_model",
    "train_final_model",
    "load_model",
    "predict",
    "plot_actual_vs_predicted",
    "evaluate_cv",
    "evaluate_train_test",
]


if __name__ == "__main__":
    print("[INFO] Executing surrogate model cross-validation evaluation...")
    plot_cv_path = diagnostic_plots_dir / "actual_vs_predicted_cv.png"
    df_folds, df_summary, oof_df = evaluate_cv(save_plot_path=plot_cv_path)
    print("\n--- 5-Fold Cross-Validation Metrics Summary ---")
    print(df_summary.to_string(index=False))
