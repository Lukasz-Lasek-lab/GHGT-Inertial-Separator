"""
Declarative Feature Registry for MsCO2limit inertial particle separator.
Single Source of Truth (SSOT) for all geometric, trigonometric, logarithmic,
and polynomial candidate and engineered feature formulas.
"""

import threading
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.constants import BASE_FEATURES

# Regularization constant to prevent division by zero or log of non-positive values
EPSILON: float = 1e-6


class FeatureRegistry:
    """
    Central declarative registry for aerodynamic and geometric feature transformations.
    Provides vector and scalar computation routines, dependency and metadata tracking,
    thread-safe dynamic registration, and single-pass unfragmented DataFrame generation.
    """
    _registry: Dict[str, Dict[str, Any]] = {}
    _feature_order: List[str] = []
    _lock: threading.RLock = threading.RLock()

    @classmethod
    def register(
        cls,
        name: str,
        dependencies: Optional[List[str]] = None,
        category: str = "custom",
        description: str = "",
    ) -> Callable:
        """
        Decorator to register a feature calculation function into the central registry.
        Thread-safe under concurrent registration calls.
        """
        if dependencies is None:
            dependencies = []

        def decorator(fn: Callable) -> Callable:
            with cls._lock:
                if name not in cls._feature_order:
                    cls._feature_order.append(name)
                cls._registry[name] = {
                    "func": fn,
                    "deps": list(dependencies),
                    "category": category,
                    "description": description,
                }
            return fn

        return decorator

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Checks whether a feature name is registered in the registry."""
        with cls._lock:
            return name in cls._registry

    @classmethod
    def get_registered_features(cls) -> List[str]:
        """Returns the ordered list of all registered feature names."""
        with cls._lock:
            return list(cls._feature_order)

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Returns metadata dictionary for a registered feature."""
        with cls._lock:
            if not cls.is_registered(name):
                raise ValueError(f"Unknown geometric feature: '{name}' is not registered.")
            return dict(cls._registry[name])

    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregisters a feature from the registry.
        Primarily used for test teardown and dynamic registration testing.
        """
        with cls._lock:
            if name in cls._registry:
                del cls._registry[name]
            if name in cls._feature_order:
                cls._feature_order.remove(name)

    @classmethod
    def compute_feature(
        cls,
        name: str,
        data: Any = None,
        *args: Any,
        _call_stack: Optional[set] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Computes a single registered feature for given inputs.

        Supported calling conventions:
        - DataFrame: FeatureRegistry.compute_feature("Alfa_Beta", df)
        - Row Series: FeatureRegistry.compute_feature("Alfa_Beta", df.iloc[0])
        - Dict/Mapping: FeatureRegistry.compute_feature("Alfa_Beta", {"Alfa": 50, "Beta": 40})
        - Canonical positional args: FeatureRegistry.compute_feature("Alfa_Beta", 50, 40, 0.02, 0.02)
        - Dependency positional args: FeatureRegistry.compute_feature("sin_Beta", 45.0)
        - Keyword args: FeatureRegistry.compute_feature("Alfa_Beta", alfa=50, beta=40)
        """
        with cls._lock:
            if not cls.is_registered(name):
                raise ValueError(f"Unknown geometric feature: '{name}' is not registered in FeatureRegistry.")
            spec = cls._registry[name]

        deps = spec["deps"]

        if _call_stack is None:
            _call_stack = set()
        if name in _call_stack:
            raise ValueError(f"Cyclic dependency detected computing feature '{name}'.")
        current_stack = _call_stack | {name}

        # Case 1: DataFrame
        if isinstance(data, pd.DataFrame):
            df_curr = data
            missing = [d for d in deps if d not in df_curr.columns]
            if missing:
                lower_cols = {str(c).lower(): c for c in df_curr.columns}
                rename_map = {}
                still_missing = []
                for d in missing:
                    if d.lower() in lower_cols:
                        rename_map[lower_cols[d.lower()]] = d
                    elif cls.is_registered(d) and d not in current_stack:
                        if df_curr is data:
                            df_curr = df_curr.copy()
                        df_curr[d] = cls.compute_feature(d, df_curr, _call_stack=current_stack)
                    else:
                        still_missing.append(d)

                if rename_map:
                    df_curr = df_curr.rename(columns=rename_map)

                if still_missing:
                    raise ValueError(
                        f"Missing required dependencies {still_missing} to compute feature '{name}'."
                    )
            return spec["func"](df_curr)

        # Case 2: Mapping (dict, OrderedDict) or a single-row pd.Series
        is_mapping = isinstance(data, Mapping)
        is_row_series = False
        if isinstance(data, pd.Series) and len(args) == 0 and not kwargs:
            series_keys = set(data.index)
            lower_s_keys = {str(k).lower(): k for k in series_keys}
            if all(d in series_keys or d.lower() in lower_s_keys or (cls.is_registered(d) and d not in current_stack) for d in deps):
                is_row_series = True

        if is_mapping or is_row_series:
            resolved_dict = dict(data)
            raw_keys = set(resolved_dict.keys())
            lower_keys = {str(k).lower(): k for k in raw_keys}
            missing = []
            for d in deps:
                if d in resolved_dict:
                    continue
                elif d.lower() in lower_keys:
                    resolved_dict[d] = resolved_dict[lower_keys[d.lower()]]
                elif cls.is_registered(d) and d not in current_stack:
                    resolved_dict[d] = cls.compute_feature(d, resolved_dict, _call_stack=current_stack)
                else:
                    missing.append(d)

            if missing:
                raise ValueError(
                    f"Missing required dependencies {missing} to compute feature '{name}'."
                )
            return spec["func"](resolved_dict)

        # Case 3: Positional arguments
        input_dict: Dict[str, Any] = {}
        if data is not None:
            pos_vals = [data] + list(args)
            if len(pos_vals) == len(BASE_FEATURES):
                # 4 arguments passed: canonical (Alfa, Beta, H1, H2) order
                for base_name, val in zip(BASE_FEATURES, pos_vals):
                    input_dict[base_name] = val
            elif len(pos_vals) == len(deps):
                # Positional arguments match feature dependencies directly
                for dep_name, val in zip(deps, pos_vals):
                    input_dict[dep_name] = val
            elif len(pos_vals) <= len(BASE_FEATURES) and all(d in BASE_FEATURES[:len(pos_vals)] for d in deps):
                # Prefix of BASE_FEATURES covers all deps
                for base_name, val in zip(BASE_FEATURES[:len(pos_vals)], pos_vals):
                    input_dict[base_name] = val
            elif len(pos_vals) <= len(deps):
                # Prefix of deps
                for dep_name, val in zip(deps[:len(pos_vals)], pos_vals):
                    input_dict[dep_name] = val
            else:
                for base_name, val in zip(BASE_FEATURES, pos_vals):
                    input_dict[base_name] = val

        # Case 4: Keyword arguments (case-insensitive for base feature names and deps)
        for k, v in kwargs.items():
            matched = False
            for target in list(BASE_FEATURES) + list(deps):
                if k.lower() == target.lower():
                    input_dict[target] = v
                    matched = True
                    break
            if not matched:
                input_dict[k] = v

        if input_dict:
            missing = []
            for d in deps:
                if d in input_dict:
                    continue
                elif cls.is_registered(d) and d not in current_stack:
                    input_dict[d] = cls.compute_feature(d, input_dict, _call_stack=current_stack)
                else:
                    missing.append(d)

            if missing:
                raise ValueError(
                    f"Missing required dependencies {missing} to compute feature '{name}'."
                )
            return spec["func"](input_dict)

        raise ValueError(f"No valid input data provided to compute feature '{name}'.")

    @classmethod
    def compute_all(
        cls,
        df: pd.DataFrame,
        include_base: bool = True,
    ) -> pd.DataFrame:
        """
        Computes all candidate features for an input DataFrame in a single unfragmented pass.
        Guarantees zero PerformanceWarning and memory optimization by compiling column dict.
        Supports chained feature dependencies and auxiliary DataFrame columns in eval_context.
        """
        missing_base = [b for b in BASE_FEATURES if b not in df.columns]
        if missing_base:
            raise ValueError(
                f"Missing base geometric parameter columns in DataFrame: {missing_base}"
            )

        eval_context: Dict[str, Any] = {col: df[col] for col in df.columns}
        for b in BASE_FEATURES:
            eval_context[b] = df[b].astype(float)

        col_dict: Dict[str, Any] = {}

        # 1. Base features if requested
        if include_base:
            for b in BASE_FEATURES:
                col_dict[b] = eval_context[b]

        # 2. Engineered candidate features
        with cls._lock:
            feature_order = list(cls._feature_order)

        for feat in feature_order:
            if feat in BASE_FEATURES:
                continue
            res = cls.compute_feature(feat, eval_context)
            col_dict[feat] = res
            eval_context[feat] = res

        return pd.DataFrame(col_dict, index=df.index)


# ==============================================================================
# Declarative Registration of Base Geometric Features
# ==============================================================================

@FeatureRegistry.register("Alfa", dependencies=["Alfa"], category="base", description="Inlet deflection angle Alfa (deg)")
def _feat_alfa(d: Any) -> Any:
    return d["Alfa"]


@FeatureRegistry.register("Beta", dependencies=["Beta"], category="base", description="Outlet deflection angle Beta (deg)")
def _feat_beta(d: Any) -> Any:
    return d["Beta"]


@FeatureRegistry.register("H1", dependencies=["H1"], category="base", description="Inlet slot height H1 (m)")
def _feat_h1(d: Any) -> Any:
    return d["H1"]


@FeatureRegistry.register("H2", dependencies=["H2"], category="base", description="Outlet slot height H2 (m)")
def _feat_h2(d: Any) -> Any:
    return d["H2"]


# ==============================================================================
# Declarative Registration of 22 Engineered Aerodynamic Features
# ==============================================================================

# 1. Products and Ratios (6 features)
@FeatureRegistry.register("Alfa_Beta", dependencies=["Alfa", "Beta"], category="product", description="Angle interaction product Alfa * Beta")
def _feat_alfa_beta(d: Any) -> Any:
    return d["Alfa"] * d["Beta"]


@FeatureRegistry.register("H1_H2", dependencies=["H1", "H2"], category="product", description="Slot area interaction product H1 * H2")
def _feat_h1_h2(d: Any) -> Any:
    return d["H1"] * d["H2"]


@FeatureRegistry.register("Alfa_div_Beta", dependencies=["Alfa", "Beta"], category="ratio", description="Deflection angle ratio Alfa / (Beta + eps)")
def _feat_alfa_div_beta(d: Any) -> Any:
    return d["Alfa"] / (d["Beta"] + EPSILON)


@FeatureRegistry.register("H1_div_H2", dependencies=["H1", "H2"], category="ratio", description="Slot height ratio H1 / (H2 + eps)")
def _feat_h1_div_h2(d: Any) -> Any:
    return d["H1"] / (d["H2"] + EPSILON)


@FeatureRegistry.register("log_H1", dependencies=["H1"], category="logarithm", description="Natural log of inlet height log(H1 + eps)")
def _feat_log_h1(d: Any) -> Any:
    return np.log(d["H1"] + EPSILON)


@FeatureRegistry.register("log_H2", dependencies=["H2"], category="logarithm", description="Natural log of outlet height log(H2 + eps)")
def _feat_log_h2(d: Any) -> Any:
    return np.log(d["H2"] + EPSILON)


# 2. Linear Sums and Absolute Differences (4 features)
@FeatureRegistry.register("Alfa_plus_Beta", dependencies=["Alfa", "Beta"], category="sum_difference", description="Combined deflection angle Alfa + Beta")
def _feat_alfa_plus_beta(d: Any) -> Any:
    return d["Alfa"] + d["Beta"]


@FeatureRegistry.register("Alfa_minus_Beta", dependencies=["Alfa", "Beta"], category="sum_difference", description="Deflection asymmetry |Alfa - Beta|")
def _feat_alfa_minus_beta(d: Any) -> Any:
    return np.abs(d["Alfa"] - d["Beta"])


@FeatureRegistry.register("H1_plus_H2", dependencies=["H1", "H2"], category="sum_difference", description="Combined slot height H1 + H2")
def _feat_h1_plus_h2(d: Any) -> Any:
    return d["H1"] + d["H2"]


@FeatureRegistry.register("H1_minus_H2", dependencies=["H1", "H2"], category="sum_difference", description="Slot height disparity |H1 - H2|")
def _feat_h1_minus_h2(d: Any) -> Any:
    return np.abs(d["H1"] - d["H2"])


# 3. Trigonometric Descriptors (4 features - degrees converted to radians)
@FeatureRegistry.register("sin_Alfa", dependencies=["Alfa"], category="trigonometric", description="Sine of inlet deflection angle sin(radians(Alfa))")
def _feat_sin_alfa(d: Any) -> Any:
    return np.sin(np.radians(d["Alfa"]))


@FeatureRegistry.register("cos_Alfa", dependencies=["Alfa"], category="trigonometric", description="Cosine of inlet deflection angle cos(radians(Alfa))")
def _feat_cos_alfa(d: Any) -> Any:
    return np.cos(np.radians(d["Alfa"]))


@FeatureRegistry.register("sin_Beta", dependencies=["Beta"], category="trigonometric", description="Sine of outlet deflection angle sin(radians(Beta))")
def _feat_sin_beta(d: Any) -> Any:
    return np.sin(np.radians(d["Beta"]))


@FeatureRegistry.register("cos_Beta", dependencies=["Beta"], category="trigonometric", description="Cosine of outlet deflection angle cos(radians(Beta))")
def _feat_cos_beta(d: Any) -> Any:
    return np.cos(np.radians(d["Beta"]))


# 4. Second- and Third-Order Polynomial Powers (8 features)
@FeatureRegistry.register("Alfa_squared", dependencies=["Alfa"], category="polynomial", description="Second-order power Alfa^2")
def _feat_alfa_squared(d: Any) -> Any:
    return d["Alfa"] ** 2


@FeatureRegistry.register("Alfa_cubed", dependencies=["Alfa"], category="polynomial", description="Third-order power Alfa^3")
def _feat_alfa_cubed(d: Any) -> Any:
    return d["Alfa"] ** 3


@FeatureRegistry.register("Beta_squared", dependencies=["Beta"], category="polynomial", description="Second-order power Beta^2")
def _feat_beta_squared(d: Any) -> Any:
    return d["Beta"] ** 2


@FeatureRegistry.register("Beta_cubed", dependencies=["Beta"], category="polynomial", description="Third-order power Beta^3")
def _feat_beta_cubed(d: Any) -> Any:
    return d["Beta"] ** 3


@FeatureRegistry.register("H1_squared", dependencies=["H1"], category="polynomial", description="Second-order power H1^2")
def _feat_h1_squared(d: Any) -> Any:
    return d["H1"] ** 2


@FeatureRegistry.register("H1_cubed", dependencies=["H1"], category="polynomial", description="Third-order power H1^3")
def _feat_h1_cubed(d: Any) -> Any:
    return d["H1"] ** 3


@FeatureRegistry.register("H2_squared", dependencies=["H2"], category="polynomial", description="Second-order power H2^2")
def _feat_h2_squared(d: Any) -> Any:
    return d["H2"] ** 2


@FeatureRegistry.register("H2_cubed", dependencies=["H2"], category="polynomial", description="Third-order power H2^3")
def _feat_h2_cubed(d: Any) -> Any:
    return d["H2"] ** 3


__all__ = ["FeatureRegistry", "EPSILON"]
