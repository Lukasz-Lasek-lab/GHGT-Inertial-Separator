"""
Phase 4 Refactoring Test Suite:
1. Separation of Concerns (SRP): Zero matplotlib/seaborn imports in numerical modules
   (src.models, src.genetic, src.sensitivity, src.pareto_selection).
2. src.models: get_best_params lazy getter, plot_actual_vs_predicted delegation with DeprecationWarning,
   and surrogate_regressor.joblib standardization.
3. src.genetic: NSGA2Optimizer class encapsulation, collision-free DEAP creator type registration,
   and pure numerical optimization.
4. src.sensitivity: compute_2d_grid calculation and deprecated plotting facades.
5. src.pareto_selection: 2-regime physical H2 partitioning (H2 >= 0.05m -> cluster 0 / ~59mm,
   H2 < 0.05m -> cluster 1 / ~39mm) and deterministic 2-group clustering sorted by H2.
"""

import inspect
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from src.constants import BASE_FEATURES, PARAM_BOUNDS, TOTAL_PARTICLES
import src.models as mod_models
import src.genetic as mod_genetic
import src.sensitivity as mod_sensitivity
import src.pareto_selection as mod_pareto


class MockSurrogateModel:
    """Fast deterministic mock surrogate model for tests without heavy fitting."""

    def predict(self, X):
        X_arr = np.asarray(X)
        n = len(X_arr)
        # Produce plausible outputs within physical range
        n1 = np.full(n, 50.0)
        delta = np.full(n, 5500.0)
        return np.column_stack([n1, delta])


class TestNumericalModulesZeroMatplotlib(unittest.TestCase):
    """Ensures ML, genetic, sensitivity, and Pareto modules have zero matplotlib dependencies."""

    def test_no_matplotlib_in_module_namespaces(self):
        """None of the four numerical modules should expose matplotlib or seaborn at module level."""
        for mod, name in [
            (mod_models, "src.models"),
            (mod_genetic, "src.genetic"),
            (mod_sensitivity, "src.sensitivity"),
            (mod_pareto, "src.pareto_selection"),
        ]:
            mod_dict = mod.__dict__
            self.assertNotIn("matplotlib", mod_dict, f"{name} must not import matplotlib at module level")
            self.assertNotIn("plt", mod_dict, f"{name} must not import plt at module level")
            self.assertNotIn("seaborn", mod_dict, f"{name} must not import seaborn at module level")
            self.assertNotIn("sns", mod_dict, f"{name} must not import sns at module level")

    def test_module_source_ast_no_top_level_plotting_imports(self):
        """Top-level AST or source of numerical modules must not contain matplotlib/seaborn imports."""
        for mod, name in [
            (mod_models, "src.models"),
            (mod_genetic, "src.genetic"),
            (mod_sensitivity, "src.sensitivity"),
            (mod_pareto, "src.pareto_selection"),
        ]:
            src_file = inspect.getsourcefile(mod)
            with open(src_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line_no, line in enumerate(lines, 1):
                clean = line.strip()
                if clean.startswith("import matplotlib") or clean.startswith("from matplotlib"):
                    self.fail(f"{name}:{line_no} contains direct matplotlib import: {clean}")
                if clean.startswith("import seaborn") or clean.startswith("from seaborn"):
                    self.fail(f"{name}:{line_no} contains direct seaborn import: {clean}")


class TestModelsPhase4(unittest.TestCase):
    """Tests for Phase 4 improvements in src.models."""

    def test_get_best_params_lazy_getter(self):
        """get_best_params must return a fresh dict and support reloading."""
        params1 = mod_models.get_best_params()
        self.assertIsInstance(params1, dict)
        self.assertIn("learning_rate", params1)
        self.assertIn("max_depth", params1)

        # Mutating the returned copy must not corrupt internal cache
        params1["custom_key"] = 9999
        params2 = mod_models.get_best_params()
        self.assertNotIn("custom_key", params2)

        # Test reload=True
        params_reloaded = mod_models.get_best_params(reload=True)
        self.assertIsInstance(params_reloaded, dict)

        # Test legacy BEST_PARAMS module attribute exists
        self.assertIsInstance(mod_models.BEST_PARAMS, dict)

    def test_plot_actual_vs_predicted_facade(self):
        """plot_actual_vs_predicted must emit DeprecationWarning and delegate to fig2_diagnostics."""
        y_true = pd.DataFrame({"N1": [10.0, 20.0, 30.0], "N2": [5000.0, 5100.0, 5200.0], "Delta": [4990.0, 5080.0, 5170.0]})
        y_pred = pd.DataFrame({"N1": [12.0, 18.0, 31.0], "N2": [4980.0, 5120.0, 5190.0], "Delta": [4968.0, 5102.0, 5159.0]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_diag.png"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mod_models.plot_actual_vs_predicted(y_true, y_pred, save_path=save_path)
                dep_warns = [item for item in w if issubclass(item.category, DeprecationWarning)]
                self.assertGreaterEqual(len(dep_warns), 1)
                self.assertIn("deprecated", str(dep_warns[0].message).lower())

            self.assertTrue(save_path.exists())

    def test_model_serialization_standardization(self):
        """train_final_model should save surrogate_regressor.joblib and maintain legacy alias."""
        mock_model = MockSurrogateModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_canonical = tmp_path / "surrogate_regressor.joblib"
            save_legacy = tmp_path / "Final_Model.joblib"

            from config.path import demo_data_file
            with patch("src.models.train_model", return_value=mock_model), \
                 patch("src.models.surrogate_models_dir", tmp_path):
                model, path_out = mod_models.train_final_model(
                    data_path=demo_data_file,
                    save_path=save_canonical,
                )
                self.assertEqual(path_out.resolve(), save_canonical.resolve())
                self.assertTrue(save_canonical.exists())
                self.assertTrue(save_legacy.exists())

    def test_load_model_deprecation_warning_on_legacy_only(self):
        """load_model should emit DeprecationWarning if loading Final_Model.joblib fallback."""
        import joblib
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            legacy_file = tmp_path / "Final_Model.joblib"
            joblib.dump(MockSurrogateModel(), legacy_file)

            with patch("src.models.surrogate_models_dir", tmp_path):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    loaded = mod_models.load_model()
                    self.assertIsNotNone(loaded)
                    dep_warns = [item for item in w if issubclass(item.category, DeprecationWarning)]
                    self.assertGreaterEqual(len(dep_warns), 1)
                    self.assertIn("legacy", str(dep_warns[0].message).lower())


class TestNSGA2Optimizer(unittest.TestCase):
    """Tests for NSGA2Optimizer class in src.genetic."""

    def test_creator_registration_idempotence(self):
        """Repeated instantiation of NSGA2Optimizer must not crash or corrupt creator types."""
        model = MockSurrogateModel()
        opt1 = mod_genetic.NSGA2Optimizer(model=model, pop_size=10, n_gen=2)
        opt2 = mod_genetic.NSGA2Optimizer(model=model, pop_size=10, n_gen=2)
        self.assertIsNotNone(opt1.toolbox)
        self.assertIsNotNone(opt2.toolbox)

    def test_optimize_execution_pure_dataframe(self):
        """NSGA2Optimizer.optimize() should run without matplotlib and return 4 clean DataFrames."""
        model = MockSurrogateModel()
        optimizer = mod_genetic.NSGA2Optimizer(
            model=model,
            pop_size=12,
            n_gen=3,
            seed=42,
        )
        pop_df, hof_df, pareto_df, log_df = optimizer.optimize()

        self.assertIsInstance(pop_df, pd.DataFrame)
        self.assertIsInstance(hof_df, pd.DataFrame)
        self.assertIsInstance(pareto_df, pd.DataFrame)
        self.assertIsInstance(log_df, pd.DataFrame)

        self.assertEqual(len(pop_df), 12)
        self.assertIn("domination_rank", pop_df.columns)
        self.assertTrue((pop_df["domination_rank"] >= 0).all())
        self.assertGreaterEqual(len(pareto_df), 1)
        self.assertEqual(len(log_df), 4)  # Gen 0, 1, 2, 3

    def test_setup_toolbox_wrapper_compatibility(self):
        """setup_toolbox function must remain callable and return a valid Toolbox."""
        model = MockSurrogateModel()
        toolbox = mod_genetic.setup_toolbox(model, PARAM_BOUNDS)
        self.assertTrue(hasattr(toolbox, "individual"))
        self.assertTrue(hasattr(toolbox, "population"))
        self.assertTrue(hasattr(toolbox, "evaluate"))


class TestSensitivityPhase4(unittest.TestCase):
    """Tests for Phase 4 improvements in src.sensitivity."""

    def test_compute_2d_grid_pure_numerical(self):
        """compute_2d_grid must compute meshgrids and predictions without matplotlib."""
        model = MockSurrogateModel()
        ref_params = {"Alfa": 50.0, "Beta": 50.0, "H1": 0.03, "H2": 0.03}
        grid_size = 8
        G1, G2, n1_grid, n2_grid = mod_sensitivity.compute_2d_grid(
            model=model,
            param1="Alfa",
            param2="Beta",
            range1=(45.0, 55.0),
            range2=(45.0, 55.0),
            ref_params=ref_params,
            grid_size=grid_size,
        )
        self.assertEqual(G1.shape, (grid_size, grid_size))
        self.assertEqual(G2.shape, (grid_size, grid_size))
        self.assertEqual(n1_grid.shape, (grid_size, grid_size))
        self.assertEqual(n2_grid.shape, (grid_size, grid_size))

    def test_plot_sensitivity_analysis_facade(self):
        """plot_sensitivity_analysis must emit DeprecationWarning and generate file via delegation."""
        results = {
            "Alfa": pd.DataFrame({"Alfa": [45.0, 50.0], "N1_pred": [10.0, 20.0], "N2_pred": [5500.0, 5400.0], "Delta": [5490.0, 5380.0]}),
            "Beta": pd.DataFrame({"Beta": [45.0, 50.0], "N1_pred": [12.0, 18.0], "N2_pred": [5520.0, 5410.0], "Delta": [5508.0, 5392.0]}),
        }
        ref_dict = {"Alfa": 50.0, "Beta": 50.0, "N1_ref": 15.0, "N2_ref": 5450.0, "Delta_ref": 5435.0}

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "sens_test.png"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mod_sensitivity.plot_sensitivity_analysis(results, ref_dict, save_path=save_path)
                dep_warns = [item for item in w if issubclass(item.category, DeprecationWarning)]
                self.assertGreaterEqual(len(dep_warns), 1)

            self.assertTrue(save_path.exists())

    def test_plot_combined_view_facade(self):
        """plot_combined_view must emit DeprecationWarning and generate file via delegation."""
        results = {
            "Alfa": pd.DataFrame({"Alfa": [45.0, 50.0], "N1_pred": [10.0, 20.0], "N2_pred": [5500.0, 5400.0], "Delta": [5490.0, 5380.0]}),
        }
        ref_dict = {"Alfa": 50.0, "N1_ref": 15.0, "N2_ref": 5450.0, "Delta_ref": 5435.0}

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "comb_test.png"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mod_sensitivity.plot_combined_view(results, ref_dict, save_path=save_path)
                dep_warns = [item for item in w if issubclass(item.category, DeprecationWarning)]
                self.assertGreaterEqual(len(dep_warns), 1)

            self.assertTrue(save_path.exists())

    def test_plot_heatmap_interactions_facade(self):
        """plot_heatmap_interactions must emit DeprecationWarning and generate file via delegation."""
        model = MockSurrogateModel()
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "heat_test.png"
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                mod_sensitivity.plot_heatmap_interactions(model=model, save_path=save_path, grid_size=6)
                dep_warns = [item for item in w if issubclass(item.category, DeprecationWarning)]
                self.assertGreaterEqual(len(dep_warns), 1)

            self.assertTrue(save_path.exists())


class TestParetoSelectionPhase4(unittest.TestCase):
    """Tests for Phase 4 physical regime partitioning and Pareto postprocessing."""

    def setUp(self):
        # 4 Pareto points: 2 in high-gap regime (~59mm), 2 in low-gap regime (~39mm)
        self.mock_pareto = pd.DataFrame({
            "Alfa": [45.0, 50.0, 55.0, 58.0],
            "Beta": [45.0, 50.0, 52.0, 56.0],
            "H1": [0.03, 0.04, 0.02, 0.035],
            "H2": [0.0595, 0.0580, 0.0385, 0.0392],  # indices 0,1 >= 0.05; 2,3 < 0.05
            "N1_pred": [5.0, 20.0, 35.0, 60.0],
            "Delta_pred": [5850.0, 5800.0, 5750.0, 5650.0],
            "N2_pred": [5855.0, 5820.0, 5785.0, 5710.0],
        })

    def test_partition_pareto_regimes_threshold(self):
        """partition_pareto_regimes must assign 0 to H2 >= 0.05 and 1 to H2 < 0.05."""
        clusters = mod_pareto.partition_pareto_regimes(self.mock_pareto, h2_threshold=0.05)
        np.testing.assert_array_equal(clusters.values, [0, 0, 1, 1])

    def test_partition_pareto_regimes_boundary(self):
        """Exact threshold boundary value tests (0.0500 vs 0.0499)."""
        boundary_df = pd.DataFrame({"H2": [0.0500, 0.0499, 0.0609, 0.0080]})
        clusters = mod_pareto.partition_pareto_regimes(boundary_df, h2_threshold=0.05)
        np.testing.assert_array_equal(clusters.values, [0, 1, 0, 1])

    def test_partition_pareto_regimes_missing_h2_raises_error(self):
        """Calling partition_pareto_regimes without H2 column must raise ValueError."""
        no_h2 = pd.DataFrame({"N1": [10.0, 20.0]})
        with self.assertRaises(ValueError):
            mod_pareto.partition_pareto_regimes(no_h2)

    def test_cluster_pareto_deterministic_sorting(self):
        """cluster_pareto_deterministic must guarantee cluster 0 has higher mean H2 than cluster 1."""
        clusters = mod_pareto.cluster_pareto_deterministic(
            self.mock_pareto, n_clusters=2, n1_col="N1_pred", n2_col="N2_pred"
        )
        self.mock_pareto["cluster"] = clusters
        mean_h2_c0 = self.mock_pareto[self.mock_pareto["cluster"] == 0]["H2"].mean()
        mean_h2_c1 = self.mock_pareto[self.mock_pareto["cluster"] == 1]["H2"].mean()
        self.assertGreater(mean_h2_c0, mean_h2_c1)

    def test_select_optimal_configurations_physical_regimes_export(self):
        """select_optimal_configurations must export cluster and regime matching physics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_excel = Path(tmp_dir) / "pareto_out.xlsx"
            result_df = mod_pareto.select_optimal_configurations(
                pareto_df=self.mock_pareto,
                output_excel_path=out_excel,
                clustering_method="physics",
                h2_threshold=0.05,
            )

            self.assertIn("cluster", result_df.columns)
            self.assertIn("regime", result_df.columns)

            # Check that cluster 0 is High-gap and cluster 1 is Low-gap
            for _, row in result_df.iterrows():
                if row["H2"] >= 0.05:
                    self.assertEqual(row["cluster"], 0)
                    self.assertIn("59 mm", row["regime"])
                else:
                    self.assertEqual(row["cluster"], 1)
                    self.assertIn("39 mm", row["regime"])

            # Check particle conservation: N1 + N2 + N3 == 6417
            sums = result_df["N1"] + result_df["N2"] + result_df["N3"]
            np.testing.assert_array_equal(sums.values, np.full(len(result_df), TOTAL_PARTICLES))

            # Check files created
            self.assertTrue(out_excel.exists())
            self.assertTrue(out_excel.with_suffix(".csv").exists())


if __name__ == "__main__":
    unittest.main()
