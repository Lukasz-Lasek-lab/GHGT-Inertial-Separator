"""
Moduł automatycznej inżynierii i selekcji cech (Feature Selection) dla projektu MsCO2limit.
Implementuje metodologię doboru cech zgodną z notebookami:
- notebooks/Wstępne_prace_dane Test 5.ipynb
- notebooks/Wybór cech dla modelu Test 5.ipynb

Generuje pełną pulę cech geometrycznych dla nowego zbioru danych (Dane_T5.xlsx),
ocenia ich ważność komitetem modeli dla N1 i Delta, a następnie zapisuje:
- listę wyselekcjonowanych cech do config/selected_features.json
- znormalizowany zbiór z wyliczonymi cechami do data/processed/df_selected.csv
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

from config.path import config_dir, processed_data_dir, raw_data_dir

# 4 podstawowe nastawy geometrii (zawsze zachowywane)
BASE_FEATURES: List[str] = ["Alfa", "Beta", "H1", "H2"]
TOTAL_PARTICLES: int = 6417


def load_raw_dataset(excel_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Wczytuje surowy arkusz Excel (np. Dane_T5.xlsx) i normalizuje nazwy kolumn.
    """
    if excel_path is None:
        excel_path = raw_data_dir / "Dane_T5.xlsx"

    if not excel_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku surowego: {excel_path}")

    df_raw = pd.read_excel(excel_path)

    # Mapowanie kolumn bez względu na formatowanie stopni i znaków specjalnych
    col_map = {}
    for col in df_raw.columns:
        col_lower = str(col).lower()
        if "alfa" in col_lower:
            col_map[col] = "Alfa"
        elif "beta" in col_lower:
            col_map[col] = "Beta"
        elif "h1" in col_lower:
            col_map[col] = "H1"
        elif "h2" in col_lower:
            col_map[col] = "H2"
        elif "n1" in col_lower:
            col_map[col] = "N1"
        elif "n2" in col_lower:
            col_map[col] = "N2"
        elif "n3" in col_lower or "kulek" in col_lower:
            col_map[col] = "N3"

    df = df_raw.rename(columns=col_map)
    required = ["Alfa", "Beta", "H1", "H2", "N1", "N2"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"W pliku {excel_path} brakuje wymaganych kolumn: {missing}")

    if "N3" not in df.columns:
        df["N3"] = TOTAL_PARTICLES - df["N1"] - df["N2"]

    # Obliczenie targetu Delta = N2 - N1
    df["Delta"] = df["N2"] - df["N1"]

    return df


def generate_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy pełną pulę geometrycznych cech kandydujących na podstawie cech bazowych
    zgodnie z funkcją create_geometric_features z notebooka wstępnego:
    - Iloczyny i ilorazy
    - Sumy i różnice bezwzględne
    - Logarytmy
    - Funkcje trygonometryczne (sin, cos w radianach)
    - Potęgi 2. i 3. stopnia
    """
    df_feat = pd.DataFrame(index=df.index)

    alfa = df["Alfa"].astype(float)
    beta = df["Beta"].astype(float)
    h1 = df["H1"].astype(float)
    h2 = df["H2"].astype(float)

    # Cechy bazowe
    df_feat["Alfa"] = alfa
    df_feat["Beta"] = beta
    df_feat["H1"] = h1
    df_feat["H2"] = h2

    # Iloczyny i ilorazy
    df_feat["Alfa_Beta"] = alfa * beta
    df_feat["H1_H2"] = h1 * h2
    df_feat["Alfa_div_Beta"] = alfa / (beta + 1e-6)
    df_feat["H1_div_H2"] = h1 / (h2 + 1e-6)
    df_feat["log_H1"] = np.log(h1 + 1e-6)
    df_feat["log_H2"] = np.log(h2 + 1e-6)

    # Sumy i różnice bezwzględne
    df_feat["Alfa_plus_Beta"] = alfa + beta
    df_feat["Alfa_minus_Beta"] = np.abs(alfa - beta)
    df_feat["H1_plus_H2"] = h1 + h2
    df_feat["H1_minus_H2"] = np.abs(h1 - h2)

    # Funkcje trygonometryczne
    df_feat["sin_Alfa"] = np.sin(np.radians(alfa))
    df_feat["cos_Alfa"] = np.cos(np.radians(alfa))
    df_feat["sin_Beta"] = np.sin(np.radians(beta))
    df_feat["cos_Beta"] = np.cos(np.radians(beta))

    # Potęgi 2. i 3. stopnia
    for col_name, s in [("Alfa", alfa), ("Beta", beta), ("H1", h1), ("H2", h2)]:
        df_feat[f"{col_name}_squared"] = s ** 2
        df_feat[f"{col_name}_cubed"] = s ** 3

    return df_feat


def select_best_features(
    X_cand: pd.DataFrame,
    y_n1: pd.Series,
    y_delta: pd.Series,
    max_engineered_features: int = 7,
    random_state: int = 42,
) -> List[str]:
    """
    Dokonuje wielokryterialnej selekcji cech:
    1. Filtr wariancji (VarianceThreshold)
    2. Random Forest Feature Importance dla N1 i Delta
    3. SelectKBest (f_regression) dla N1 i Delta
    4. LassoCV dla N1 i Delta
    5. Agregacja punktacji i wybór top-K cech inżynieryjnych + 4 cechy bazowe.
    """
    candidate_cols = [c for c in X_cand.columns if c not in BASE_FEATURES]

    # 1. Filtr wariancji
    vt = VarianceThreshold(threshold=1e-5)
    vt.fit(X_cand[candidate_cols])
    valid_candidates = [c for c, sup in zip(candidate_cols, vt.get_support()) if sup]

    X_sub = X_cand[valid_candidates]

    # Scoring słownik
    feature_scores: Dict[str, float] = {c: 0.0 for c in valid_candidates}

    for target_name, y in [("N1", y_n1), ("Delta", y_delta)]:
        # Standardyzacja y do ważenia scoringów
        y_std = (y - y.mean()) / (y.std() + 1e-6)

        # 2. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=250, random_state=random_state, n_jobs=-1)
        rf.fit(X_sub, y_std)
        importances = rf.feature_importances_
        # Normalizacja wag do sumy 1
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

    # Sortowanie cech wg zagregowanego wyniku
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Ranking wygenerowanych cech inżynieryjnych (Top 15) ---")
    for rank, (feat, score) in enumerate(sorted_features[:15], 1):
        print(f" {rank:2d}. {feat:<20} score: {score:.4f}")

    # Wybór cech bez nadmiarowych par o korelacji > 0.985
    selected_engineered: List[str] = []
    corr_matrix = X_cand[valid_candidates].corr().abs()

    for feat, _ in sorted_features:
        if len(selected_engineered) >= max_engineered_features:
            break
        # Sprawdź korelację z już wybranymi cechami
        too_correlated = False
        for chosen in selected_engineered:
            if corr_matrix.loc[feat, chosen] > 0.985:
                too_correlated = True
                break
        if not too_correlated:
            selected_engineered.append(feat)

    # Łączna lista: 4 cechy bazowe + wyselekcjonowane cechy inżynieryjne
    final_features = BASE_FEATURES + selected_engineered
    return final_features


def run_feature_selection(
    raw_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    max_engineered: int = 7,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Główna funkcja wykonawcza:
    1. Wczytuje dane surowe Dane_T5.xlsx
    2. Generuje kandydatów
    3. Przeprowadza selekcję cech
    4. Zapisuje selected_features.json
    5. Zapisuje gotowy zbiór df_selected.csv z cechami i targetami.
    """
    if raw_path is None:
        raw_path = raw_data_dir / "Dane_T5.xlsx"
    if output_csv_path is None:
        output_csv_path = processed_data_dir / "df_selected.csv"
    if json_path is None:
        json_path = config_dir / "selected_features.json"

    print(f"[INFO] Wczytywanie surowych danych z: {raw_path}")
    df_raw = load_raw_dataset(raw_path)

    print(f"[INFO] Liczba wierszy: {len(df_raw)}, suma czastek N1+N2+N3: {(df_raw['N1']+df_raw['N2']+df_raw['N3']).unique()}")
    print("[INFO] Generowanie puli cech geometrycznych...")
    X_cand = generate_candidate_features(df_raw)
    print(f"[INFO] Wygenerowano {len(X_cand.columns)} cech kandydujacych.")

    print("[INFO] Selekcja cech za pomoca komitetu (RF, f_regression, Lasso)...")
    selected_features = select_best_features(
        X_cand,
        y_n1=df_raw["N1"],
        y_delta=df_raw["Delta"],
        max_engineered_features=max_engineered,
    )

    print(f"\n[OK] Wybrane cechy ({len(selected_features)}): {selected_features}")

    # Zapis do JSON
    json_path.parent.mkdir(parents=True, exist_ok=True)
    features_meta = {
        "dataset_source": str(raw_path.name),
        "total_particles": TOTAL_PARTICLES,
        "base_features": BASE_FEATURES,
        "engineered_features": [f for f in selected_features if f not in BASE_FEATURES],
        "all_selected_features": selected_features,
        "target_names": ["N1", "Delta"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features_meta, f, indent=4)
    print(f"[SAVE] Lista cech zapisana do: {json_path}")

    # Przygotowanie pełnego DataFrame do trenowania modelu
    df_out = X_cand[selected_features].copy()
    df_out["N1"] = df_raw["N1"]
    df_out["N2"] = df_raw["N2"]
    df_out["N3"] = df_raw["N3"]
    df_out["Delta"] = df_raw["Delta"]
    df_out["eta_1_pct"] = (df_raw["N1"] / TOTAL_PARTICLES) * 100.0
    df_out["eta_2_pct"] = (df_raw["N2"] / TOTAL_PARTICLES) * 100.0

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)
    print(f"[SAVE] Znormalizowany zbior zapisany do: {output_csv_path} (wymiar: {df_out.shape})")

    return selected_features, df_out


if __name__ == "__main__":
    run_feature_selection()
