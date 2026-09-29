"""
Tests for feature engineering logic and mathematical formulas in MsCO2limit:
- Analytical correctness of all 22 engineered features and 4 base features
- Numerical consistency between src.features and src.feature_selection
- Verification against reference engineered columns in demo dataset
- Vector, scalar, and array handling
- Error handling for missing inputs and invalid feature names
"""

import unittest
import warnings
import numpy as np
import pandas as pd

from src.constants import BASE_FEATURES
from src.features.registry import FeatureRegistry, EPSILON
from src.features import (
    compute_single_feature,
    create_features,
    create_feature_dict,
    create_feature_array,
    load_selected_features,
    get_selected_features,
    reload_features,
    FEATURE_NAMES,
    ENGINEERED_FEATURES,
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


class TestFeatureRegistry(unittest.TestCase):
    """Test suite for declarative FeatureRegistry functionality, metadata, and memory efficiency."""

    def setUp(self):
        self.test_df = pd.DataFrame({
            "Alfa": [45.0, 60.0],
            "Beta": [42.75, 55.0],
            "H1": [0.015, 0.038],
            "H2": [0.020, 0.050],
        })

    def test_registered_features_coverage(self):
        """All 4 base and 22 engineered features must be registered."""
        base_features = ["Alfa", "Beta", "H1", "H2"]
        engineered_features = [
            "Alfa_Beta", "H1_H2", "Alfa_div_Beta", "H1_div_H2", "log_H1", "log_H2",
            "Alfa_plus_Beta", "Alfa_minus_Beta", "H1_plus_H2", "H1_minus_H2",
            "sin_Alfa", "cos_Alfa", "sin_Beta", "cos_Beta",
            "Alfa_squared", "Alfa_cubed", "Beta_squared", "Beta_cubed",
            "H1_squared", "H1_cubed", "H2_squared", "H2_cubed",
        ]
        registered = FeatureRegistry.get_registered_features()
        for feat in base_features + engineered_features:
            self.assertTrue(FeatureRegistry.is_registered(feat), f"Feature '{feat}' not registered")
            self.assertIn(feat, registered)

        self.assertEqual(len(registered), 26)

    def test_get_metadata(self):
        """Feature metadata must contain valid dependencies, category, and description."""
        meta_sin = FeatureRegistry.get_metadata("sin_Alfa")
        self.assertEqual(meta_sin["deps"], ["Alfa"])
        self.assertEqual(meta_sin["category"], "trigonometric")
        self.assertTrue(len(meta_sin["description"]) > 0)

        meta_prod = FeatureRegistry.get_metadata("Alfa_Beta")
        self.assertEqual(meta_prod["deps"], ["Alfa", "Beta"])
        self.assertEqual(meta_prod["category"], "product")

        with self.assertRaises(ValueError):
            FeatureRegistry.get_metadata("non_existent_feature")

    def test_dynamic_registration_and_teardown(self):
        """User-defined feature can be registered, computed, and unregistered cleanly."""
        custom_name = "test_custom_ratio"

        @FeatureRegistry.register(
            custom_name,
            dependencies=["Alfa", "H1"],
            category="experimental",
            description="Experimental test ratio Alfa / H1",
        )
        def _calc_custom(d):
            return d["Alfa"] * d["H1"]

        self.assertTrue(FeatureRegistry.is_registered(custom_name))
        self.assertIn(custom_name, FeatureRegistry.get_registered_features())

        val = FeatureRegistry.compute_feature(custom_name, alfa=50.0, h1=0.02)
        self.assertAlmostEqual(val, 1.0)

        # Teardown
        FeatureRegistry.unregister(custom_name)
        self.assertFalse(FeatureRegistry.is_registered(custom_name))
        self.assertNotIn(custom_name, FeatureRegistry.get_registered_features())

    def test_compute_feature_dispatch(self):
        """compute_feature must dispatch DataFrame, dict, positional, and keyword arguments."""
        # DataFrame dispatch
        s = FeatureRegistry.compute_feature("Alfa_plus_Beta", self.test_df)
        np.testing.assert_allclose(s.values, [87.75, 115.0])

        # Dict dispatch
        d_val = FeatureRegistry.compute_feature("Alfa_plus_Beta", {"Alfa": 50.0, "Beta": 30.0})
        self.assertEqual(d_val, 80.0)

        # Positional dispatch (Alfa, Beta, H1, H2)
        pos_val = FeatureRegistry.compute_feature("Alfa_minus_Beta", 60.0, 45.0, 0.038, 0.019)
        self.assertEqual(pos_val, 15.0)

        # Keyword dispatch
        kw_val = FeatureRegistry.compute_feature("Alfa_minus_Beta", beta=45.0, alfa=60.0)
        self.assertEqual(kw_val, 15.0)

    def test_compute_feature_error_handling(self):
        """compute_feature must raise ValueError for missing dependencies or unregistered features."""
        with self.assertRaises(ValueError):
            FeatureRegistry.compute_feature("non_existent", self.test_df)

        with self.assertRaises(ValueError):
            FeatureRegistry.compute_feature("Alfa_plus_Beta", {"Alfa": 50.0})  # Missing Beta

        with self.assertRaises(ValueError):
            FeatureRegistry.compute_feature("Alfa_plus_Beta")  # No inputs

    def test_compute_all_features(self):
        """compute_all must compute all 26 candidate features with exact shapes."""
        df_all = FeatureRegistry.compute_all(self.test_df)
        self.assertEqual(df_all.shape, (2, 26))
        self.assertEqual(list(df_all.columns), FeatureRegistry.get_registered_features())

        # Without base features
        df_eng = FeatureRegistry.compute_all(self.test_df, include_base=False)
        self.assertEqual(df_eng.shape, (2, 22))
        for base in BASE_FEATURES:
            self.assertNotIn(base, df_eng.columns)

    def test_compute_all_missing_base_columns(self):
        """compute_all must raise ValueError if required base columns are missing."""
        bad_df = pd.DataFrame({"Alfa": [50.0], "Beta": [50.0]})
        with self.assertRaises(ValueError):
            FeatureRegistry.compute_all(bad_df)

    def test_compute_all_empty_dataframe(self):
        """compute_all on empty DataFrame must return empty DataFrame with all 26 columns."""
        empty_df = pd.DataFrame(columns=BASE_FEATURES)
        df_out = FeatureRegistry.compute_all(empty_df)
        self.assertEqual(len(df_out), 0)
        self.assertEqual(list(df_out.columns), FeatureRegistry.get_registered_features())

    def test_memory_fragmentation_prevention(self):
        """DataFrame construction must not emit PerformanceWarning for column fragmentation."""
        large_df = pd.DataFrame({
            "Alfa": np.linspace(42.75, 60.0, 500),
            "Beta": np.linspace(42.75, 60.0, 500),
            "H1": np.linspace(0.008, 0.0609, 500),
            "H2": np.linspace(0.008, 0.0609, 500),
        })

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = FeatureRegistry.compute_all(large_df)
            _ = create_features(large_df)

            perf_warnings = [
                w for w in caught_warnings
                if issubclass(w.category, pd.errors.PerformanceWarning)
            ]
            self.assertEqual(len(perf_warnings), 0, f"Memory fragmentation detected: {perf_warnings}")

    def test_compute_feature_partial_positional_args(self):
        """compute_feature must dispatch positional arguments matching feature dependencies directly."""
        # 1-arg dependencies (Beta, H1, H2)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("sin_Beta", 45.0), np.sin(np.radians(45.0)))
        self.assertAlmostEqual(FeatureRegistry.compute_feature("cos_Beta", 45.0), np.cos(np.radians(45.0)))
        self.assertAlmostEqual(FeatureRegistry.compute_feature("Beta_squared", 45.0), 2025.0)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("Beta_cubed", 45.0), 91125.0)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("log_H1", 0.038), np.log(0.038 + EPSILON))
        self.assertAlmostEqual(FeatureRegistry.compute_feature("log_H2", 0.019), np.log(0.019 + EPSILON))
        self.assertAlmostEqual(FeatureRegistry.compute_feature("H1_squared", 0.038), 0.038 ** 2)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("H2_cubed", 0.019), 0.019 ** 3)

        # 2-arg dependencies (H1, H2)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("H1_div_H2", 0.038, 0.019), 0.038 / (0.019 + EPSILON))
        self.assertAlmostEqual(FeatureRegistry.compute_feature("H1_H2", 0.038, 0.019), 0.038 * 0.019)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("H1_plus_H2", 0.038, 0.019), 0.057)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("H1_minus_H2", 0.038, 0.019), 0.019)

    def test_compute_feature_series_row_input(self):
        """compute_feature must correctly handle single-row pd.Series inputs and return scalar values."""
        row = pd.Series({"Alfa": 60.0, "Beta": 45.0, "H1": 0.038, "H2": 0.019})
        self.assertAlmostEqual(FeatureRegistry.compute_feature("Alfa_Beta", row), 2700.0)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("sin_Alfa", row), np.sin(np.radians(60.0)))
        self.assertAlmostEqual(FeatureRegistry.compute_feature("sin_Beta", row), np.sin(np.radians(45.0)))
        self.assertAlmostEqual(FeatureRegistry.compute_feature("H1_div_H2", row), 0.038 / (0.019 + EPSILON))

    def test_compute_feature_dict_case_insensitivity(self):
        """compute_feature must support case-insensitive keys in dict/Mapping inputs."""
        lower_dict = {"alfa": 60.0, "beta": 45.0}
        self.assertAlmostEqual(FeatureRegistry.compute_feature("Alfa_Beta", lower_dict), 2700.0)
        self.assertAlmostEqual(FeatureRegistry.compute_feature("sin_Alfa", lower_dict), np.sin(np.radians(60.0)))

        mixed_dict = {"h1": 0.038, "H2": 0.019}
        self.assertAlmostEqual(FeatureRegistry.compute_feature("H1_div_H2", mixed_dict), 0.038 / (0.019 + EPSILON))

    def test_chained_dependencies_and_dataframe_extra_columns(self):
        """compute_all and create_features must resolve chained dependencies and extra DataFrame columns."""
        # 1. Custom feature depending on another engineered feature
        custom_sq_name = "test_chained_ratio_sq"

        @FeatureRegistry.register(
            custom_sq_name,
            dependencies=["H1_div_H2"],
            category="experimental",
            description="Chained feature depending on H1_div_H2",
        )
        def _calc_sq(d):
            return d["H1_div_H2"] ** 2

        # 2. Custom feature depending on an auxiliary DataFrame column
        custom_aux_name = "test_aux_interaction"

        @FeatureRegistry.register(
            custom_aux_name,
            dependencies=["Alfa", "aux_flow"],
            category="experimental",
            description="Interaction with auxiliary column",
        )
        def _calc_aux(d):
            return d["Alfa"] * d["aux_flow"]

        try:
            df_with_aux = self.test_df.copy()
            df_with_aux["aux_flow"] = [1.5, 2.5]

            # compute_all must succeed and evaluate both
            df_computed = FeatureRegistry.compute_all(df_with_aux)
            self.assertIn(custom_sq_name, df_computed.columns)
            self.assertIn(custom_aux_name, df_computed.columns)

            expected_ratio = df_with_aux["H1"] / (df_with_aux["H2"] + EPSILON)
            np.testing.assert_allclose(df_computed[custom_sq_name].values, (expected_ratio ** 2).values)
            np.testing.assert_allclose(df_computed[custom_aux_name].values, [45.0 * 1.5, 60.0 * 2.5])

            # create_features must also succeed
            df_created = create_features(df_with_aux, feature_names=["Alfa", custom_sq_name, custom_aux_name])
            self.assertEqual(list(df_created.columns), ["Alfa", custom_sq_name, custom_aux_name])
            np.testing.assert_allclose(df_created[custom_sq_name].values, (expected_ratio ** 2).values)

        finally:
            FeatureRegistry.unregister(custom_sq_name)
            FeatureRegistry.unregister(custom_aux_name)

    def test_thread_safety_concurrent_access(self):
        """FeatureRegistry must maintain thread safety during concurrent registration and computation."""
        import threading

        errors = []

        def worker(thread_id):
            feat_name = f"thread_feat_{thread_id}"
            try:
                @FeatureRegistry.register(feat_name, dependencies=["Alfa"])
                def _fn(d):
                    return d["Alfa"] + thread_id

                val = FeatureRegistry.compute_feature(feat_name, alfa=10.0)
                if val != 10.0 + thread_id:
                    errors.append(f"Mismatch in thread {thread_id}: {val}")

                FeatureRegistry.unregister(feat_name)
            except Exception as e:
                errors.append(f"Exception in thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

    def test_lazy_loading_and_backward_compatibility(self):
        """get_selected_features, load_selected_features, and reload_features must function correctly."""
        feats = get_selected_features()
        self.assertIsInstance(feats, list)
        self.assertTrue(len(feats) >= 4)
        for base in BASE_FEATURES:
            self.assertIn(base, feats)

        loaded_feats = load_selected_features()
        self.assertEqual(feats, loaded_feats)

        reloaded = reload_features()
        self.assertEqual(feats, reloaded)


if __name__ == "__main__":
    unittest.main()

