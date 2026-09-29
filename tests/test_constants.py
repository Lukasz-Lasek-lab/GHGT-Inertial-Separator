"""
Tests for constants and configuration boundaries in MsCO2limit project.
Verifies the Single Source of Truth (SSOT) defined in src/constants.py.
"""

import unittest
from src.constants import (
    TOTAL_PARTICLES,
    BASE_FEATURES,
    TARGET_COLUMNS,
    TARGET_NAMES,
    PARAM_BOUNDS,
    NOMINAL_BASELINE,
    PARAM_UNITS,
    DEFAULT_PARAM_BOUNDS,
    DEFAULT_PARAM_RANGES,
    PARAM_RANGES,
    DEFAULT_REFERENCE_PARAMS,
    REFERENCE_PARAMS,
)
import src.features as mod_features
import src.feature_selection as mod_feat_sel
import src.pareto_selection as mod_pareto
import src.sensitivity as mod_sensitivity
import src.genetic as mod_genetic
import src.visualization.figures as mod_figures


class TestConstants(unittest.TestCase):
    """Test suite verifying SSOT constants, physical particle counts, and parameter domains."""

    def test_total_particles_value_and_type(self):
        """Particle balance requires exact integer count of 6417 particles."""
        self.assertIsInstance(TOTAL_PARTICLES, int)
        self.assertEqual(TOTAL_PARTICLES, 6417)
        self.assertGreater(TOTAL_PARTICLES, 0)

    def test_base_features(self):
        """Four primary geometric design parameters: Alfa, Beta, H1, H2."""
        expected = ["Alfa", "Beta", "H1", "H2"]
        self.assertIsInstance(BASE_FEATURES, list)
        self.assertEqual(BASE_FEATURES, expected)
        self.assertEqual(len(BASE_FEATURES), 4)

    def test_target_columns_and_names(self):
        """Target columns must define particle loss, capture, and net separation advantage."""
        self.assertEqual(TARGET_COLUMNS, ["N1", "N2", "Delta"])
        self.assertEqual(TARGET_NAMES, ["N1", "Delta"])

    def test_param_bounds_validity(self):
        """Parameter bounds must cover all base features with positive, valid intervals."""
        self.assertEqual(set(PARAM_BOUNDS.keys()), set(BASE_FEATURES))

        for param, (low, high) in PARAM_BOUNDS.items():
            self.assertIsInstance(low, (int, float), f"Lower bound for {param} must be numeric")
            self.assertIsInstance(high, (int, float), f"Upper bound for {param} must be numeric")
            self.assertGreater(low, 0.0, f"Lower bound for {param} must be strictly positive")
            self.assertLess(low, high, f"Lower bound must be strictly less than upper bound for {param}")

        # Specific domain constraints from CFD simulation design
        self.assertEqual(PARAM_BOUNDS["Alfa"], (42.75, 60.0))
        self.assertEqual(PARAM_BOUNDS["Beta"], (42.75, 60.0))
        self.assertEqual(PARAM_BOUNDS["H1"], (0.0080, 0.0609))
        self.assertEqual(PARAM_BOUNDS["H2"], (0.0080, 0.0609))

    def test_nominal_baseline_validity(self):
        """Nominal CFD baseline parameters must fall strictly within the allowed parameter bounds."""
        self.assertEqual(set(NOMINAL_BASELINE.keys()), set(BASE_FEATURES))

        for param, val in NOMINAL_BASELINE.items():
            low, high = PARAM_BOUNDS[param]
            self.assertGreaterEqual(val, low, f"Nominal {param} is below lower bound {low}")
            self.assertLessEqual(val, high, f"Nominal {param} is above upper bound {high}")

        # Exact reference geometry: Alfa=60 deg, Beta=60 deg, H1=38 mm, H2=38 mm
        self.assertEqual(NOMINAL_BASELINE["Alfa"], 60.0)
        self.assertEqual(NOMINAL_BASELINE["Beta"], 60.0)
        self.assertAlmostEqual(NOMINAL_BASELINE["H1"], 0.0380, places=4)
        self.assertAlmostEqual(NOMINAL_BASELINE["H2"], 0.0380, places=4)

    def test_param_units(self):
        """Geometric angles must be in degrees ('deg') and heights in meters ('m')."""
        self.assertEqual(set(PARAM_UNITS.keys()), set(BASE_FEATURES))
        self.assertEqual(PARAM_UNITS["Alfa"], "deg")
        self.assertEqual(PARAM_UNITS["Beta"], "deg")
        self.assertEqual(PARAM_UNITS["H1"], "m")
        self.assertEqual(PARAM_UNITS["H2"], "m")

    def test_backwards_compatibility_aliases(self):
        """Legacy aliases must refer to the exact same SSOT structures."""
        self.assertEqual(DEFAULT_PARAM_BOUNDS, PARAM_BOUNDS)
        self.assertEqual(DEFAULT_PARAM_RANGES, PARAM_BOUNDS)
        self.assertEqual(PARAM_RANGES, PARAM_BOUNDS)
        self.assertEqual(DEFAULT_REFERENCE_PARAMS, NOMINAL_BASELINE)
        self.assertEqual(REFERENCE_PARAMS, NOMINAL_BASELINE)

    def test_ssot_module_integration(self):
        """Other repository modules must use the SSOT constants without divergent definitions."""
        self.assertEqual(mod_features.BASE_FEATURES, BASE_FEATURES)
        self.assertEqual(mod_features.TARGET_NAMES, TARGET_NAMES)
        self.assertEqual(mod_feat_sel.BASE_FEATURES, BASE_FEATURES)
        self.assertEqual(mod_feat_sel.TOTAL_PARTICLES, TOTAL_PARTICLES)
        self.assertEqual(mod_pareto.TOTAL_PARTICLES, TOTAL_PARTICLES)
        self.assertEqual(mod_pareto.BASE_FEATURES, BASE_FEATURES)
        self.assertEqual(mod_sensitivity.BASE_FEATURES, BASE_FEATURES)
        self.assertEqual(mod_sensitivity.PARAM_BOUNDS, PARAM_BOUNDS)
        self.assertEqual(mod_sensitivity.NOMINAL_BASELINE, NOMINAL_BASELINE)
        self.assertEqual(mod_genetic.BASE_FEATURES, BASE_FEATURES)
        self.assertEqual(mod_genetic.DEFAULT_PARAM_BOUNDS, PARAM_BOUNDS)
        self.assertEqual(mod_figures.TOTAL_PARTICLES, TOTAL_PARTICLES)
        self.assertEqual(mod_figures.DEFAULT_PARAM_RANGES, PARAM_BOUNDS)
        self.assertEqual(mod_figures.DEFAULT_REFERENCE_PARAMS, NOMINAL_BASELINE)

        import src.models as mod_models
        import src.visualization.publication_plots as mod_pub_plots
        self.assertEqual(mod_feat_sel.TARGET_NAMES, TARGET_NAMES)
        self.assertEqual(mod_models.TARGET_NAMES, TARGET_NAMES)
        self.assertEqual(mod_models.TOTAL_PARTICLES, TOTAL_PARTICLES)
        self.assertEqual(mod_pub_plots.PARAM_BOUNDS, PARAM_BOUNDS)


if __name__ == "__main__":
    unittest.main()
