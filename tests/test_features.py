"""
Tests for feature engineering logic and mathematical formulas in MsCO2limit:
- Analytical correctness of all 22 engineered features and 4 base features
- Numerical consistency between src.features and src.feature_selection
- Verification against reference engineered columns in demo dataset
- Vector, scalar, and array handling
- Error handling for missing inputs and invalid feature names
"""

import unittest
import numpy as np
import pandas as pd

from src.constants import BASE_FEATURES
from src.features import (
    compute_single_feature,
    create_features,
    create_feature_dict,
    create_feature_array,
    FEATURE_NAMES,
)
from src.feature_selection import generate_candidate_features
from config.path import demo_data_file


class TestFeatures(unittest.TestCase):
    """Test suite verifying mathematical definitions and numerical integrity of features."""

    def setUp(self):
        self.alfa = 60.0
        self.beta = 45.0
        self.h1 = 0.0380
        self.h2 = 0.0190
        self.eps = 1e-6

    def test_base_features_identity(self):
        """Base features must return exact inputs without alteration."""
        self.assertEqual(compute_single_feature("Alfa", self.alfa, self.beta, self.h1, self.h2), 60.0)
        self.assertEqual(compute_single_feature("Beta", self.alfa, self.beta, self.h1, self.h2), 45.0)
        self.assertEqual(compute_single_feature("H1", self.alfa, self.beta, self.h1, self.h2), 0.0380)
        self.assertEqual(compute_single_feature("H2", self.alfa, self.beta, self.h1, self.h2), 0.0190)

    def test_products_and_ratios(self):
        """Verify products, quotients, and logarithms including epsilon regularization."""
        self.assertAlmostEqual(
            compute_single_feature("Alfa_Beta", self.alfa, self.beta, self.h1, self.h2),
            60.0 * 45.0,
        )
        self.assertAlmostEqual(
            compute_single_feature("H1_H2", self.alfa, self.beta, self.h1, self.h2),
            0.0380 * 0.0190,
        )
        self.assertAlmostEqual(
            compute_single_feature("Alfa_div_Beta", self.alfa, self.beta, self.h1, self.h2),
            60.0 / (45.0 + self.eps),
        )
        self.assertAlmostEqual(
            compute_single_feature("H1_div_H2", self.alfa, self.beta, self.h1, self.h2),
            0.0380 / (0.0190 + self.eps),
        )
        self.assertAlmostEqual(
            compute_single_feature("log_H1", self.alfa, self.beta, self.h1, self.h2),
            np.log(0.0380 + self.eps),
        )
        self.assertAlmostEqual(
            compute_single_feature("log_H2", self.alfa, self.beta, self.h1, self.h2),
            np.log(0.0190 + self.eps),
        )

    def test_sums_and_differences(self):
        """Verify linear sums and symmetric absolute differences."""
        self.assertAlmostEqual(
            compute_single_feature("Alfa_plus_Beta", self.alfa, self.beta, self.h1, self.h2),
            105.0,
        )
        self.assertAlmostEqual(
            compute_single_feature("Alfa_minus_Beta", self.alfa, self.beta, self.h1, self.h2),
            15.0,
        )
        # Check symmetry: |Beta - Alfa| == |Alfa - Beta|
        self.assertAlmostEqual(
            compute_single_feature("Alfa_minus_Beta", 45.0, 60.0, self.h1, self.h2),
            15.0,
        )
        self.assertAlmostEqual(
            compute_single_feature("H1_plus_H2", self.alfa, self.beta, self.h1, self.h2),
            0.0570,
        )
        self.assertAlmostEqual(
            compute_single_feature("H1_minus_H2", self.alfa, self.beta, self.h1, self.h2),
            0.0190,
        )

    def test_trigonometric_features(self):
        """Verify trigonometric functions with degree-to-radian conversion."""
        # 60 degrees: sin = sqrt(3)/2, cos = 0.5
        self.assertAlmostEqual(
            compute_single_feature("sin_Alfa", self.alfa, self.beta, self.h1, self.h2),
            np.sin(np.radians(60.0)),
        )
        self.assertAlmostEqual(
            compute_single_feature("cos_Alfa", self.alfa, self.beta, self.h1, self.h2),
            0.5,
        )
        # 45 degrees: sin = cos = sqrt(2)/2
        self.assertAlmostEqual(
            compute_single_feature("sin_Beta", self.alfa, self.beta, self.h1, self.h2),
            np.sin(np.radians(45.0)),
        )
        self.assertAlmostEqual(
            compute_single_feature("cos_Beta", self.alfa, self.beta, self.h1, self.h2),
            np.cos(np.radians(45.0)),
        )

    def test_polynomial_powers(self):
        """Verify second- and third-order powers for angles and slot heights."""
        self.assertAlmostEqual(compute_single_feature("Alfa_squared", self.alfa, self.beta, self.h1, self.h2), 3600.0)
        self.assertAlmostEqual(compute_single_feature("Alfa_cubed", self.alfa, self.beta, self.h1, self.h2), 216000.0)
        self.assertAlmostEqual(compute_single_feature("Beta_squared", self.alfa, self.beta, self.h1, self.h2), 2025.0)
        self.assertAlmostEqual(compute_single_feature("Beta_cubed", self.alfa, self.beta, self.h1, self.h2), 91125.0)
        self.assertAlmostEqual(compute_single_feature("H1_squared", self.alfa, self.beta, self.h1, self.h2), 0.0380 ** 2)
        self.assertAlmostEqual(compute_single_feature("H1_cubed", self.alfa, self.beta, self.h1, self.h2), 0.0380 ** 3)
        self.assertAlmostEqual(compute_single_feature("H2_squared", self.alfa, self.beta, self.h1, self.h2), 0.0190 ** 2)
        self.assertAlmostEqual(compute_single_feature("H2_cubed", self.alfa, self.beta, self.h1, self.h2), 0.0190 ** 3)

    def test_feature_selection_parity(self):
        """Formulas in candidate generation must produce identical results to compute_single_feature."""
        test_df = pd.DataFrame({
            "Alfa": [45.0, 50.0, 60.0],
            "Beta": [42.75, 48.0, 60.0],
            "H1": [0.010, 0.038, 0.060],
            "H2": [0.060, 0.038, 0.010],
        })

        cand_df = generate_candidate_features(test_df)

        for col in cand_df.columns:
            computed_series = compute_single_feature(
                col,
                test_df["Alfa"].values,
                test_df["Beta"].values,
                test_df["H1"].values,
                test_df["H2"].values,
            )
            np.testing.assert_allclose(
                cand_df[col].values,
                computed_series,
                rtol=1e-6,
                err_msg=f"Discrepancy detected for feature {col} between module definitions",
            )

    def test_demo_dataset_fidelity(self):
        """Pre-computed engineered features in demo dataset must match calculated values."""
        df_demo = pd.read_csv(demo_data_file)
        engineered_in_demo = [
            "Beta_cubed",
            "Alfa_cubed",
            "log_H1",
            "Alfa_plus_Beta",
            "H1_plus_H2",
            "H1_div_H2",
            "log_H2",
        ]

        for feat in engineered_in_demo:
            self.assertIn(feat, df_demo.columns)
            recalculated = compute_single_feature(
                feat,
                df_demo["Alfa"].values,
                df_demo["Beta"].values,
                df_demo["H1"].values,
                df_demo["H2"].values,
            )
            # The base parameters in demo CSV are formatted to 2-4 decimals,
            # introducing small rounding divergence in logarithms and powers.
            np.testing.assert_allclose(
                df_demo[feat].values,
                recalculated,
                rtol=2e-3,
                atol=0.01,
                err_msg=f"Engineered feature {feat} in demo dataset does not match formula",
            )

    def test_create_features_vectorized(self):
        """create_features must accept a full DataFrame and return properly aligned columns."""
        df = pd.DataFrame({
            "Alfa": [50.0, 60.0],
            "Beta": [50.0, 60.0],
            "H1": [0.02, 0.038],
            "H2": [0.02, 0.038],
        })
        selected = ["Alfa", "Beta_cubed", "log_H1", "H1_plus_H2"]
        result = create_features(df, feature_names=selected)
        self.assertEqual(list(result.columns), selected)
        self.assertEqual(len(result), 2)

    def test_create_feature_dict_and_array(self):
        """create_feature_dict and create_feature_array must return scalar dicts and numpy vectors."""
        names = ["Alfa", "Beta", "Alfa_plus_Beta"]
        f_dict = create_feature_dict(50.0, 50.0, 0.02, 0.02, feature_names=names)
        self.assertEqual(f_dict, {"Alfa": 50.0, "Beta": 50.0, "Alfa_plus_Beta": 100.0})

        f_arr = create_feature_array(50.0, 50.0, 0.02, 0.02, feature_names=names)
        self.assertIsInstance(f_arr, np.ndarray)
        np.testing.assert_array_equal(f_arr, [50.0, 50.0, 100.0])

    def test_error_handling(self):
        """Missing base columns or unknown feature name must raise ValueError."""
        # Missing column
        bad_df = pd.DataFrame({"Alfa": [50.0], "Beta": [50.0]})
        with self.assertRaises(ValueError):
            create_features(bad_df)

        # Unknown feature
        with self.assertRaises(ValueError):
            compute_single_feature("non_existent_feature", 50.0, 50.0, 0.02, 0.02)

    def test_compute_single_feature_series_input(self):
        """compute_single_feature must correctly handle pandas Series inputs."""
        s_alfa = pd.Series([45.0, 60.0])
        s_beta = pd.Series([45.0, 60.0])
        s_h1 = pd.Series([0.020, 0.038])
        s_h2 = pd.Series([0.020, 0.038])

        res_sum = compute_single_feature("Alfa_plus_Beta", s_alfa, s_beta, s_h1, s_h2)
        np.testing.assert_array_equal(res_sum.values, [90.0, 120.0])

        res_prod = compute_single_feature("Alfa_Beta", s_alfa, s_beta, s_h1, s_h2)
        np.testing.assert_array_equal(res_prod.values, [2025.0, 3600.0])

        res_sin = compute_single_feature("sin_Alfa", s_alfa, s_beta, s_h1, s_h2)
        np.testing.assert_allclose(res_sin.values, [np.sin(np.radians(45.0)), np.sin(np.radians(60.0))])

        res_log = compute_single_feature("log_H1", s_alfa, s_beta, s_h1, s_h2)
        np.testing.assert_allclose(res_log.values, [np.log(0.020 + 1e-6), np.log(0.038 + 1e-6)])

        res_cube = compute_single_feature("Alfa_cubed", s_alfa, s_beta, s_h1, s_h2)
        np.testing.assert_array_equal(res_cube.values, [45.0 ** 3, 60.0 ** 3])

    def test_create_features_empty_dataframe(self):
        """create_features on an empty DataFrame with valid base columns must return empty DataFrame with feature columns."""
        empty_df = pd.DataFrame(columns=["Alfa", "Beta", "H1", "H2"])
        res = create_features(empty_df)
        self.assertEqual(len(res), 0)
        self.assertEqual(list(res.columns), FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
