"""
Single Source of Truth (SSOT) for physical, geometric, and modeling constants
for the MsCO2limit inertial separator project:
'Hybrid AI-driven Approach for Optimization and Geometric Analysis of
Inertial Separators in Chemical Looping Systems'
"""

from typing import Dict, List, Tuple

# Total number of particles injected in CFD simulation / experimental design
TOTAL_PARTICLES: int = 6417

# 4 baseline geometric design parameters
BASE_FEATURES: List[str] = ["Alfa", "Beta", "H1", "H2"]

# Multi-objective regression target columns
TARGET_COLUMNS: List[str] = ["N1", "N2", "Delta"]

# Surrogate model output targets (N1: particle loss, Delta: N2 - N1 advantage)
TARGET_NAMES: List[str] = ["N1", "Delta"]

# Physical geometric ranges matching surrogate domain bounds:
# - Angles (Alfa, Beta): [42.75 deg, 60.00 deg]
# - Heights (H1, H2): [0.0080 m, 0.0609 m] (8.0 mm to 60.9 mm)
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Alfa": (42.75, 60.0),
    "Beta": (42.75, 60.0),
    "H1": (0.0080, 0.0609),
    "H2": (0.0080, 0.0609),
}

# CFD Experimental Reference Baseline Geometry:
NOMINAL_BASELINE: Dict[str, float] = {
    "Alfa": 60.0,
    "Beta": 60.0,
    "H1": 0.0380,
    "H2": 0.0380,
}

# Physical units for geometric parameters
PARAM_UNITS: Dict[str, str] = {
    "Alfa": "deg",
    "Beta": "deg",
    "H1": "m",
    "H2": "m",
}

# Backwards compatibility aliases for existing modules
DEFAULT_PARAM_BOUNDS: Dict[str, Tuple[float, float]] = PARAM_BOUNDS
DEFAULT_PARAM_RANGES: Dict[str, Tuple[float, float]] = PARAM_BOUNDS
PARAM_RANGES: Dict[str, Tuple[float, float]] = PARAM_BOUNDS
DEFAULT_REFERENCE_PARAMS: Dict[str, float] = NOMINAL_BASELINE
REFERENCE_PARAMS: Dict[str, float] = NOMINAL_BASELINE

__all__ = [
    "TOTAL_PARTICLES",
    "BASE_FEATURES",
    "TARGET_COLUMNS",
    "TARGET_NAMES",
    "PARAM_BOUNDS",
    "NOMINAL_BASELINE",
    "PARAM_UNITS",
    "DEFAULT_PARAM_BOUNDS",
    "DEFAULT_PARAM_RANGES",
    "PARAM_RANGES",
    "DEFAULT_REFERENCE_PARAMS",
    "REFERENCE_PARAMS",
]
