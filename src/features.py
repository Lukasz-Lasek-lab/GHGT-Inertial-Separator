"""
Feature Engineering module for MsCO2limit inertial particle separator.
Dynamically supports the engineered feature set configured in config/selected_features.json
and provides vector computation routines for ML surrogates, NSGA-II optimization, and sensitivity analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from config.path import config_dir

# 4 baseline geometric design parameters
BASE_FEATURES: List[str] = ["Alfa", "Beta", "H1", "H2"]
TARGET_NAMES: List[str] = ["N1", "Delta"]


def load_selected_feature_names(json_path: Optional[Path] = None) -> Tuple[List[str], List[str]]:
    """
    Loads selected feature names from config/selected_features.json.
    Falls back to a verified domain-specific default feature set if the configuration file is missing.
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
            print(f"[WARN] Failed to load {json_path}: {e}. Utilizing default aerodynamic features.")

    # Default verified feature set for inertial separator geometry
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


# Initialize global feature name lists
FEATURE_NAMES, ENGINEERED_FEATURES = load_selected_feature_names()


def reload_features(json_path: Optional[Path] = None) -> List[str]:
    """
    Reloads the active feature list (e.g. after running feature selection).
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
    Geometric formula registry: computes a single engineered feature from input dimensions.
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

    # Products and quotients
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

    # Sums and absolute differences
    elif feat_name == "Alfa_plus_Beta":
        return alfa + beta
    elif feat_name == "Alfa_minus_Beta":
        return np.abs(alfa - beta)
    elif feat_name == "H1_plus_H2":
        return h1 + h2
    elif feat_name == "H1_minus_H2":
        return np.abs(h1 - h2)

    # Trigonometric functions (angles in degrees converted to radians)
    elif feat_name == "sin_Alfa":
        return np.sin(np.radians(alfa))
    elif feat_name == "cos_Alfa":
        return np.cos(np.radians(alfa))
    elif feat_name == "sin_Beta":
        return np.sin(np.radians(beta))
    elif feat_name == "cos_Beta":
        return np.cos(np.radians(beta))

    # Second- and third-order powers
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
        raise ValueError(f"Unknown geometric feature: {feat_name}")


def create_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes full set of engineered features for input DataFrame from base parameters 'Alfa', 'Beta', 'H1', 'H2'.
    Returns DataFrame with column order strictly matching the trained surrogate expectations.
    """
    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing base geometric parameter columns in DataFrame: {missing}")

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
    Fast computation of feature dictionary for a single design point (e.g. in optimization loop).
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
    Returns 1D numpy vector (N_features,) with feature values in the order of feature_names.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    f_dict = create_feature_dict(alfa, beta, h1, h2, feature_names=feature_names)
    return np.array([f_dict[name] for name in feature_names], dtype=np.float64)


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares regression targets: N1 (particle loss) and Delta = N2 - N1 (net collection advantage).
    """
    if "N1" not in df.columns or "N2" not in df.columns:
        raise ValueError("DataFrame must contain 'N1' and 'N2' columns.")

    y = pd.DataFrame(index=df.index)
    y["N1"] = df["N1"].values
    if "Delta" in df.columns:
        y["Delta"] = df["Delta"].values
    else:
        y["Delta"] = (df["N2"] - df["N1"]).values
    return y

