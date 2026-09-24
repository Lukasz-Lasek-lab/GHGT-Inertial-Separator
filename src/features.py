"""
Moduł inżynierii cech (Feature Engineering) dla separatora cząstek MsCO2limit.
Dynamicznie wspiera zestaw cech wyselekcjonowany w config/selected_features.json
oraz udostępnia funkcje obliczania wektora cech dla modeli ML, NSGA-II i analizy wrażliwości.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config.path import config_dir

# 4 cechy bazowe (nastawy geometrii)
BASE_FEATURES: List[str] = ["Alfa", "Beta", "H1", "H2"]
TARGET_NAMES: List[str] = ["N1", "Delta"]


def load_selected_feature_names(json_path: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """
    Wczytuje listę wybranych cech z config/selected_features.json.
    W razie braku pliku stosuje bezpieczny zestaw domyślny dla Dane_T5.xlsx.
    """
    if json_path is None:
        json_path = config_dir / "selected_features.json"

    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_feats = data.get("all_selected_features", [])
                eng_feats = data.get("engineered_features", [])
                if all_feats:
                    return all_feats, eng_feats
        except Exception as e:
            print(f"[WARN] Nie udało się wczytać {json_path}: {e}. Użycie cech domyślnych.")

    # Zestaw domyślny zoptymalizowany dla Dane_T5.xlsx
    default_eng = [
        "Beta_cubed",
        "Alfa_cubed",
        "log_H1",
        "Alfa_plus_Beta",
        "H1_plus_H2",
        "H1_div_H2",
        "log_H2",
    ]
    return BASE_FEATURES + default_eng, default_eng


# Inicjalizacja globalnych list cech
FEATURE_NAMES, ENGINEERED_FEATURES = load_selected_feature_names()


def reload_features(json_path: Optional[Path] = None) -> List[str]:
    """
    Przeładowuje listę cech (np. po ponownym uruchomieniu feature_selection).
    """
    global FEATURE_NAMES, ENGINEERED_FEATURES
    FEATURE_NAMES, ENGINEERED_FEATURES = load_selected_feature_names(json_path)
    return FEATURE_NAMES


def compute_single_feature(
    feat_name: str,
    alfa: Union[float, np.ndarray, pd.Series],
    beta: Union[float, np.ndarray, pd.Series],
    h1: Union[float, np.ndarray, pd.Series],
    h2: Union[float, np.ndarray, pd.Series],
) -> Union[float, np.ndarray, pd.Series]:
    """
    Biblioteka formuł geometrycznych: oblicza pojedynczą cechę inżynieryjną na podstawie jej nazwy.
    """
    eps = 1e-6
    if feat_name == "Alfa":
        return alfa
    elif feat_name == "Beta":
        return beta
    elif feat_name == "H1":
        return h1
    elif feat_name == "H2":
        return h2

    # Iloczyny i ilorazy
    elif feat_name == "Alfa_Beta":
        return alfa * beta
    elif feat_name == "H1_H2":
        return h1 * h2
    elif feat_name == "Alfa_div_Beta":
        return alfa / (beta + eps)
    elif feat_name == "H1_div_H2":
        return h1 / (h2 + eps)
    elif feat_name == "log_H1":
        return np.log(h1 + eps)
    elif feat_name == "log_H2":
        return np.log(h2 + eps)

    # Sumy i różnice bezwzględne
    elif feat_name == "Alfa_plus_Beta":
        return alfa + beta
    elif feat_name == "Alfa_minus_Beta":
        return np.abs(alfa - beta)
    elif feat_name == "H1_plus_H2":
        return h1 + h2
    elif feat_name == "H1_minus_H2":
        return np.abs(h1 - h2)

    # Trygonometria
    elif feat_name == "sin_Alfa":
        return np.sin(np.radians(alfa))
    elif feat_name == "cos_Alfa":
        return np.cos(np.radians(alfa))
    elif feat_name == "sin_Beta":
        return np.sin(np.radians(beta))
    elif feat_name == "cos_Beta":
        return np.cos(np.radians(beta))

    # Potęgi 2 i 3 stopnia
    elif feat_name == "Alfa_squared":
        return alfa ** 2
    elif feat_name == "Alfa_cubed":
        return alfa ** 3
    elif feat_name == "Beta_squared":
        return beta ** 2
    elif feat_name == "Beta_cubed":
        return beta ** 3
    elif feat_name == "H1_squared":
        return h1 ** 2
    elif feat_name == "H1_cubed":
        return h1 ** 3
    elif feat_name == "H2_squared":
        return h2 ** 2
    elif feat_name == "H2_cubed":
        return h2 ** 3

    else:
        raise ValueError(f"Nieznana cecha geometryczna: {feat_name}")


def create_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Tworzy pełny zestaw wybranych cech dla zadanego DataFrame na podstawie 'Alfa', 'Beta', 'H1', 'H2'.
    Zwraca DataFrame z kolumnami ułożonymi ściśle wg kolejności oczekiwanej przez model.
    """
    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Brakujące kolumny bazowe w DataFrame: {missing}")

    if feature_names is None:
        feature_names = FEATURE_NAMES

    alfa = df["Alfa"].values.astype(float)
    beta = df["Beta"].values.astype(float)
    h1 = df["H1"].values.astype(float)
    h2 = df["H2"].values.astype(float)

    df_out = pd.DataFrame(index=df.index)
    for feat in feature_names:
        df_out[feat] = compute_single_feature(feat, alfa, beta, h1, h2)

    return df_out[feature_names]


def create_feature_dict(
    alfa: float,
    beta: float,
    h1: float,
    h2: float,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Szybkie obliczanie słownika cech dla pojedynczego zestawu parametrów (np. w pętli optymalizatora).
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    feat_dict: Dict[str, float] = {}
    for feat in feature_names:
        val = compute_single_feature(feat, float(alfa), float(beta), float(h1), float(h2))
        feat_dict[feat] = float(val)

    return feat_dict


def create_feature_array(
    alfa: float,
    beta: float,
    h1: float,
    h2: float,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Zwraca wektor numpy (N_features,) z wartościami cech w kolejności feature_names.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    f_dict = create_feature_dict(alfa, beta, h1, h2, feature_names=feature_names)
    return np.array([f_dict[name] for name in feature_names], dtype=np.float64)


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje zmienne docelowe: N1 oraz Delta = N2 - N1.
    """
    if "N1" not in df.columns or "N2" not in df.columns:
        raise ValueError("DataFrame musi zawierać kolumny 'N1' oraz 'N2'")

    y = pd.DataFrame(index=df.index)
    y["N1"] = df["N1"].values
    if "Delta" in df.columns:
        y["Delta"] = df["Delta"].values
    else:
        y["Delta"] = (df["N2"] - df["N1"]).values
    return y
