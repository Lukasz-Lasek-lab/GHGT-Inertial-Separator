"""
Moduł zarządzania modelem ML dla projektu MsCO2limit.
Obsługuje architekturę HistGradientBoostingRegressor w MultiOutputRegressor,
strojenie hiperparametrów przez Optuna z buforowaniem w config/best_params.json,
trening na pełnym zbiorze danych, walidację krzyżową (CV), serializację i predykcję fizyczną.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor

from config.path import (
    config_dir,
    models_test_5_dir,
    plots_test_5_dir,
    processed_data_dir,
    results_test_5_dir,
)
from src.features import FEATURE_NAMES, prepare_targets

# Domyślne hiperparametry bazowe
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
    Wczytuje zoptymalizowane hiperparametry z config/best_params.json.
    W przypadku braku pliku zwraca domyślne parametry fallback.
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
            print(f"[WARN] Nie mozna wczytac {json_path}: {e}. Uzycie parametrow domyslnych.")

    return DEFAULT_FALLBACK_PARAMS.copy()


def save_best_params(params: Dict[str, Any], json_path: Optional[Path] = None) -> Path:
    """
    Zapisuje wyznaczone hiperparametry do pliku JSON.
    """
    if json_path is None:
        json_path = config_dir / "best_params.json"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)
    print(f"[SAVE] Hiperparametry zapisane do: {json_path}")
    return json_path


# Globalny słownik aktywnych parametrów
BEST_PARAMS: Dict[str, Any] = load_best_params()


def build_regressor(params: Optional[Dict[str, Any]] = None) -> MultiOutputRegressor:
    """
    Tworzy instancję MultiOutputRegressor z HistGradientBoostingRegressor.
    Jeśli nie podano parametrów, pobiera je z load_best_params().
    """
    model_params = load_best_params()
    if params:
        model_params.update(params)

    # Upewnij się, że typy parametrów są poprawne dla scikit-learn
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

    base_regressor = HistGradientBoostingRegressor(**model_params)
    return MultiOutputRegressor(base_regressor, n_jobs=1)


def tune_hyperparameters(
    data_path: Optional[Path] = None,
    n_trials: int = 300,
    save_json_path: Optional[Path] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Wykonuje automatyczne strojenie hiperparametrów za pomocą Optuna bezpośrednio na zbiorze danych.
    Optymalizuje ważony błąd predykcji N1 i Delta w 5-krotnej walidacji krzyżowej (CV).
    Wynik zapisuje do config/best_params.json.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if data_path is None:
        data_path = processed_data_dir / "df_selected.csv"
    if save_json_path is None:
        save_json_path = config_dir / "best_params.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z danymi do strojenia: {data_path}")

    df = pd.read_csv(data_path)
    X = df[FEATURE_NAMES].values
    y_targets = prepare_targets(df)
    y = y_targets.values

    std_n1 = float(np.std(y[:, 0])) if np.std(y[:, 0]) > 0 else 1.0
    std_delta = float(np.std(y[:, 1])) if np.std(y[:, 1]) > 0 else 1.0

    print(f"[OPTUNA] Rozpoczynanie strojenia hiperparametrow ({n_trials} prob)...")

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
    print("OPTUNA: Znaleziono najlepsze hiperparametry:")
    print("-" * 60)
    for k, v in best_p.items():
        print(f"  {k:20s}: {v}")
    print(f"  Najlepszy wynik CV loss : {study.best_value:.5f}")
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
    Trenuje model na zadanych danych X i y.
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
    Trenuje model produkcyjny na 100% dostępnych danych (df_selected.csv)
    z użyciem wyznaczonych hiperparametrów i zapisuje go na dysku.
    """
    if data_path is None:
        data_path = processed_data_dir / "df_selected.csv"
    if save_path is None:
        save_path = models_test_5_dir / "Final_Model.joblib"

    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z danymi: {data_path}")

    df = pd.read_csv(data_path)
    X = df[FEATURE_NAMES]
    y = prepare_targets(df)

    if params is None:
        params = load_best_params()

    model = train_model(X, y, params=params)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[OK] Model produkcyjny wytrenowany na 100% danych ({len(df)} rekordow) i zapisany do: {save_path}")

    return model, save_path


def load_model(model_path: Optional[Path] = None) -> MultiOutputRegressor:
    """
    Wczytuje wytrenowany model z dysku. Domyślnie models/test_5/Final_Model.joblib.
    """
    if model_path is None:
        model_path = models_test_5_dir / "Final_Model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono modelu pod ścieżką: {model_path}")

    model = joblib.load(model_path)
    return model


def predict(
    model: MultiOutputRegressor,
    X: Union[pd.DataFrame, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Wykonuje predykcję z uwzględnieniem fizycznych ograniczeń:
    - N1 >= 0 (brak ujemnych cząstek)
    - Delta >= 0 (Delta = N2 - N1)
    - N2 = N1 + Delta (N2 >= N1)
    
    Returns:
        (N1_pred, N2_pred, Delta_pred)
    """
    if isinstance(X, pd.DataFrame):
        if all(col in X.columns for col in FEATURE_NAMES):
            X = X[FEATURE_NAMES]

    raw_pred = model.predict(X)

    N1_pred = np.clip(raw_pred[:, 0], 0, None)
    Delta_pred = np.clip(raw_pred[:, 1], 0, None)
    N2_pred = N1_pred + Delta_pred

    return N1_pred, N2_pred, Delta_pred


def plot_actual_vs_predicted(
    y_true_df: pd.DataFrame,
    y_pred_df: pd.DataFrame,
    save_path: Optional[Path] = None,
    title_prefix: str = "Predykcje vs Rzeczywistosc (OOF CV)",
) -> None:
    """
    Generuje diagnostyczny wykres jakości modelu:
    - 3 wykresy rozrzutu: N1, N2 i Delta (y_true vs y_pred) z linią idealnego dopasowania y=x
    - 1 wykres rozkładu reszt (residuals = y_true - y_pred)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    targets = [
        ("N1", "tab:blue", axes[0, 0]),
        ("N2", "tab:orange", axes[0, 1]),
        ("Delta", "tab:green", axes[1, 0]),
    ]

    for target_name, color, ax in targets:
        y_t = y_true_df[target_name].values
        y_p = y_pred_df[target_name].values

        r2 = r2_score(y_t, y_p)
        mae = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))

        ax.scatter(y_t, y_p, color=color, alpha=0.75, edgecolors="black", linewidth=0.5, s=55, label="Punkty pomiarowe")

        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        margin = 0.05 * (max_val - min_val) if max_val != min_val else 1.0
        line_vals = np.linspace(min_val - margin, max_val + margin, 100)
        ax.plot(line_vals, line_vals, color="red", linestyle="--", linewidth=1.8, label="Idealne dopasowanie (y = x)")

        ax.set_title(f"{target_name}: Rzeczywiste vs Przewidywane")
        ax.set_xlabel(f"{target_name} rzeczywiste [CFD]")
        ax.set_ylabel(f"{target_name} przewidywane [Model]")
        ax.grid(True, linestyle=":", alpha=0.6)

        stats_text = f"R² = {r2:.4f}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
        ax.text(
            0.05,
            0.92,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.85),
            fontsize=10,
        )
        ax.legend(loc="lower right", fontsize=9)

    # 4. Wykres rozkładu reszt
    ax_res = axes[1, 1]
    res_n1 = y_true_df["N1"].values - y_pred_df["N1"].values
    res_n2 = y_true_df["N2"].values - y_pred_df["N2"].values

    ax_res.hist(res_n1, bins=20, alpha=0.6, color="tab:blue", label="Reszty N1 (y_true - y_pred)", edgecolor="black")
    ax_res.hist(res_n2, bins=20, alpha=0.6, color="tab:orange", label="Reszty N2 (y_true - y_pred)", edgecolor="black")
    ax_res.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax_res.set_title("Rozklad bledow predykcji (Residua)")
    ax_res.set_xlabel("Blad (y_true - y_pred)")
    ax_res.set_ylabel("Licznosc")
    ax_res.grid(True, linestyle=":", alpha=0.6)
    ax_res.legend(loc="upper right", fontsize=9)

    plt.suptitle(f"{title_prefix}", fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[PLOT] Wykres diagnostyczny zapisany do: {save_path}")
    plt.close()


def evaluate_cv(
    data_path: Optional[Path] = None,
    n_splits: int = 5,
    random_state: int = 42,
    save_plot_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Przeprowadza 5-krotną walidację krzyżową (K-Fold CV), zbiera predykcje out-of-fold (OOF),
    generuje wykres diagnostyczny i zwraca: (df_folds, df_summary, oof_df).
    """
    if data_path is None:
        data_path = processed_data_dir / "df_selected.csv"

    df = pd.read_csv(data_path)
    X = df[FEATURE_NAMES].values
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
            title_prefix=f"Model Test 5 ({len(df)} rekordow): Rzeczywiste vs OOF CV Predykcje",
        )

    return df_folds, df_summary, oof_df


def evaluate_train_test(
    data_path: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    save_plot_path: Optional[Path] = None,
) -> Tuple[MultiOutputRegressor, Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Dzieli dane na zbiór treningowy i testowy (80/20), trenuje model na zbiorze
    treningowym, wykonuje predykcję na teście, oblicza metryki i generuje wykres.
    """
    if data_path is None:
        data_path = processed_data_dir / "df_selected.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z danymi: {data_path}")

    df = pd.read_csv(data_path)
    X = df[FEATURE_NAMES]
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
            title_prefix=f"Model Test 5: Zbiór testowy ({int(test_size * 100)}% danych)",
        )

    return model, metrics, y_true_df, y_pred_df


if __name__ == "__main__":
    print("[INFO] Uruchamianie ewaluacji modelu...")
    plot_cv_path = plots_test_5_dir / "actual_vs_predicted_cv.png"
    df_folds, df_summary, oof_df = evaluate_cv(save_plot_path=plot_cv_path)
    print("\n--- Podsumowanie 5-fold CV ---")
    print(df_summary.to_string(index=False))
