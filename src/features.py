"""
Feature Engineering module for MsCO2limit inertial particle separator.
Dynamically supports the engineered feature set configured in config/selected_features.json
and provides vector computation routines for ML surrogates, NSGA-II optimization, and sensitivity analysis.
Backed by declarative FeatureRegistry as Single Source of Truth (SSOT).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Enable submodule resolution for src/features/ (e.g. src.features.registry)
__path__ = [str(Path(__file__).resolve().parent / "features")]

from config.path import config_dir
from src.constants import BASE_FEATURES, TARGET_NAMES
from src.features.registry import FeatureRegistry, EPSILON


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


def load_selected_features(json_path: Optional[Path] = None) -> List[str]:
    """
    Loads and returns all active selected feature names from JSON.
    Maintained for backward compatibility.
    """
    all_feats, _ = load_selected_feature_names(json_path)
    return all_feats


def get_selected_features(json_path: Optional[Path] = None) -> List[str]:
    """
    Lazy getter returning active selected feature names.
    If json_path is provided or FEATURE_NAMES is empty, reloads the configuration.
    """
    global FEATURE_NAMES
    if json_path is not None or not FEATURE_NAMES:
        reload_features(json_path)
    return list(FEATURE_NAMES)


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
    Computes a single engineered or base feature from input dimensions using FeatureRegistry.
    Maintains full backward compatibility for scalar, numpy array, and pandas Series inputs.
    """
    return FeatureRegistry.compute_feature(feat_name, alfa, beta, h1, h2)


def create_features(
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Computes full set of engineered features for input DataFrame from base parameters 'Alfa', 'Beta', 'H1', 'H2'.
    Returns DataFrame with column order strictly matching feature_names.
    Avoids memory fragmentation and PerformanceWarning by constructing DataFrame from column dictionary.
    Supports chained feature dependencies and auxiliary DataFrame columns in eval_context.
    """
    missing = [col for col in BASE_FEATURES if col not in df.columns]
    if missing:
        raise ValueError(f"Missing base geometric parameter columns in DataFrame: {missing}")

    if feature_names is None:
        feature_names = FEATURE_NAMES

    eval_context: Dict[str, Any] = {col: df[col] for col in df.columns}
    for col in BASE_FEATURES:
        eval_context[col] = df[col].astype(float)

    col_dict: Dict[str, Any] = {}

    for feat in feature_names:
        if feat in BASE_FEATURES:
            col_dict[feat] = eval_context[feat]
        else:
            val = FeatureRegistry.compute_feature(feat, eval_context)
            col_dict[feat] = val
            eval_context[feat] = val

    df_out = pd.DataFrame(col_dict, index=df.index)
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


__all__ = [
    "FeatureRegistry",
    "EPSILON",
    "FEATURE_NAMES",
    "ENGINEERED_FEATURES",
    "load_selected_feature_names",
    "load_selected_features",
    "get_selected_features",
    "reload_features",
    "compute_single_feature",
    "create_features",
    "create_feature_dict",
    "create_feature_array",
    "prepare_targets",
]
