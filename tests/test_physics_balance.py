"""
Tests for physical particle balance and separation efficiency invariants:
- Conservation law: N1 + N2 + N3 = 6417
- Non-negativity constraints: N1 >= 0, Delta >= 0, N3 >= 0
- Advantage relation: N2 = N1 + Delta
- Separation efficiency calculations: eta_1 (%) and eta_2 (%)
"""

import unittest
from unittest.mock import MagicMock
from pathlib import Path
import numpy as np
import pandas as pd

from src.constants import TOTAL_PARTICLES
from src.features import prepare_targets
from src.models import predict
from src.pareto_selection import select_optimal_configurations
from config.path import demo_data_file


class TestPhysicsBalance(unittest.TestCase):
    """Test suite verifying physical laws, balance invariants, and efficiency metrics."""

    def test_demo_dataset_particle_conservation(self):
        """Every row in the reference demo dataset must strictly conserve total particles (6417)."""
        self.assertTrue(demo_data_file.exists(), f"Demo file not found at {demo_data_file}")
        df = pd.read_csv(demo_data_file)

        # Check required columns
        for col in ["N1", "N2", "N3", "Delta", "eta_1_pct", "eta_2_pct"]:
            self.assertIn(col, df.columns, f"Demo dataset missing column {col}")

        # Particle sum conservation
        particle_sum = df["N1"] + df["N2"] + df["N3"]
        np.testing.assert_array_equal(
            particle_sum.values,
            np.full(len(df), TOTAL_PARTICLES),
            err_msg="Sum of N1 + N2 + N3 must equal TOTAL_PARTICLES (6417) for all rows",
        )

    def test_demo_dataset_physical_bounds(self):
        """Particle counts must be non-negative and Delta must equal N2 - N1."""
        df = pd.read_csv(demo_data_file)

        self.assertTrue((df["N1"] >= 0).all(), "N1 (particle loss) must be non-negative")
        self.assertTrue((df["N2"] >= 0).all(), "N2 (carrier capture) must be non-negative")
        self.assertTrue((df["N3"] >= 0).all(), "N3 (uncollected) must be non-negative")
        self.assertTrue((df["Delta"] >= 0).all(), "Delta (net collection advantage) must be non-negative")

        # N2 = N1 + Delta invariant
        np.testing.assert_array_equal(
            df["N2"].values,
            (df["N1"] + df["Delta"]).values,
            err_msg="N2 must strictly equal N1 + Delta",
        )

    def test_demo_dataset_efficiency_formulas(self):
        """Efficiency metrics eta_1 and eta_2 must strictly match particle fractions."""
        df = pd.read_csv(demo_data_file)

        expected_eta_1 = np.round((df["N1"] / TOTAL_PARTICLES) * 100.0, 2)
        expected_eta_2 = np.round((df["N2"] / TOTAL_PARTICLES) * 100.0, 2)

        np.testing.assert_allclose(
            df["eta_1_pct"].values,
            expected_eta_1.values,
            atol=0.01,
            err_msg="eta_1_pct calculation mismatch with physical formula",
        )
        np.testing.assert_allclose(
            df["eta_2_pct"].values,
            expected_eta_2.values,
            atol=0.01,
            err_msg="eta_2_pct calculation mismatch with physical formula",
        )

        # Combined efficiency cannot exceed 100%
        self.assertTrue(
            ((df["eta_1_pct"] + df["eta_2_pct"]) <= 100.01).all(),
            "Combined efficiency eta_1 + eta_2 cannot exceed 100%",
        )

    def test_prepare_targets_relationship(self):
        """prepare_targets must accurately construct Delta = N2 - N1."""
        dummy_df = pd.DataFrame({
            "N1": [100, 250, 0],
            "N2": [5000, 6000, 6417],
        })
        y = prepare_targets(dummy_df)
        self.assertIn("N1", y.columns)
        self.assertIn("Delta", y.columns)
        np.testing.assert_array_equal(y["N1"].values, [100, 250, 0])
        np.testing.assert_array_equal(y["Delta"].values, [4900, 5750, 6417])

    def test_predict_physical_clipping_constraints(self):
        """predict() must enforce N1 >= 0, Delta >= 0, and N2 = N1 + Delta regardless of raw ML outputs."""
        mock_model = MagicMock()
        # Mock ML output producing negative N1, negative Delta, and unbounded values
        mock_model.predict.return_value = np.array([
            [-50.0, 200.0],   # negative N1 -> must be clipped to 0
            [100.0, -30.0],   # negative Delta -> must be clipped to 0
            [-10.0, -20.0],   # both negative -> both clipped to 0
            [150.0, 4500.0],  # normal positive values
        ])

        dummy_X = np.zeros((4, 11))
        n1_pred, n2_pred, delta_pred = predict(mock_model, dummy_X)

        # Expected:
        # row 0: N1=0, Delta=200, N2=200
        # row 1: N1=100, Delta=0, N2=100
        # row 2: N1=0, Delta=0, N2=0
        # row 3: N1=150, Delta=4500, N2=4650
        np.testing.assert_array_equal(n1_pred, [0.0, 100.0, 0.0, 150.0])
        np.testing.assert_array_equal(delta_pred, [200.0, 0.0, 0.0, 4500.0])
        np.testing.assert_array_equal(n2_pred, [200.0, 100.0, 0.0, 4650.0])

        # Verify N1 >= 0, Delta >= 0, N2 = N1 + Delta
        self.assertTrue((n1_pred >= 0).all())
        self.assertTrue((delta_pred >= 0).all())
        np.testing.assert_array_equal(n2_pred, n1_pred + delta_pred)

    def test_pareto_selection_particle_balance(self):
        """select_optimal_configurations must satisfy N1 + N2 + N3 = 6417 and correct efficiency metrics."""
        pareto_mock_df = pd.DataFrame({
            "Alfa": [50.0, 55.0, 60.0],
            "Beta": [50.0, 55.0, 60.0],
            "H1": [0.02, 0.03, 0.04],
            "H2": [0.02, 0.03, 0.04],
            "N1_pred": [120.4, 200.1, 80.6],
            "Delta_pred": [5500.2, 5000.7, 5800.0],
            "N2_pred": [5620.6, 5200.8, 5880.6],
        })

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_excel = Path(tmp_dir) / "test_pareto.xlsx"
            result_df = select_optimal_configurations(
                pareto_df=pareto_mock_df,
                output_excel_path=out_excel,
                n_clusters=2,
                round_decimals=True,
            )

            # Check that N1 + N2 + N3 = TOTAL_PARTICLES (6417)
            particle_sums = result_df["N1"] + result_df["N2"] + result_df["N3"]
            np.testing.assert_array_equal(
                particle_sums.values,
                np.full(len(result_df), TOTAL_PARTICLES),
                err_msg="Pareto selection must enforce N1 + N2 + N3 == 6417",
            )

            # Check efficiency calculations
            for _, row in result_df.iterrows():
                self.assertAlmostEqual(row["eta_1_pct"], round((row["N1"] / TOTAL_PARTICLES) * 100.0, 2), places=2)
                self.assertAlmostEqual(row["eta_2_pct"], round((row["N2"] / TOTAL_PARTICLES) * 100.0, 2), places=2)
                self.assertGreaterEqual(row["N1"], 0)
                self.assertGreaterEqual(row["N2"], 0)
                self.assertGreaterEqual(row["N3"], 0)
                self.assertGreaterEqual(row["Delta"], 0)

    def test_edge_case_theoretical_extremes(self):
        """Extreme physical scenarios (100% capture, 0% capture, overflow) must maintain valid bounds."""
        # 100% collection: N1 = 0, N2 = 6417
        n1 = 0
        n2 = TOTAL_PARTICLES
        delta = n2 - n1
        n3 = TOTAL_PARTICLES - n1 - n2
        eta_1 = (n1 / TOTAL_PARTICLES) * 100.0
        eta_2 = (n2 / TOTAL_PARTICLES) * 100.0

        self.assertEqual(delta, 6417)
        self.assertEqual(n3, 0)
        self.assertEqual(eta_1, 0.0)
        self.assertEqual(eta_2, 100.0)

        # 0% capture, all lost to N1: N1 = 6417, N2 = 0
        n1 = TOTAL_PARTICLES
        n2 = 0
        n3 = TOTAL_PARTICLES - n1 - n2
        self.assertEqual(n3, 0)
        self.assertEqual((n1 / TOTAL_PARTICLES) * 100.0, 100.0)
        self.assertEqual((n2 / TOTAL_PARTICLES) * 100.0, 0.0)

    def test_pareto_selection_unrounded_and_overprediction_invariance(self):
        """Even with floating-point unrounded mode and aberrant surrogate overprediction, balance must hold."""
        import tempfile
        overpredict_df = pd.DataFrame({
            "Alfa": [45.0, 50.0],
            "Beta": [45.0, 50.0],
            "H1": [0.02, 0.03],
            "H2": [0.02, 0.03],
            # Row 0: normal
            # Row 1: severe overprediction (500 + 6200 = 6700 > 6417)
            "N1_pred": [100.25, 500.0],
            "Delta_pred": [5000.5, 5700.0],
            "N2_pred": [5100.75, 6200.0],
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_excel = Path(tmp_dir) / "test_unrounded.xlsx"
            res = select_optimal_configurations(
                pareto_df=overpredict_df,
                output_excel_path=out_excel,
                n_clusters=2,
                round_decimals=False,
            )

            # Strict conservation: N1 + N2 + N3 must equal TOTAL_PARTICLES exactly
            sums = res["N1"] + res["N2"] + res["N3"]
            np.testing.assert_allclose(sums.values, np.full(len(res), TOTAL_PARTICLES), rtol=1e-5)
            self.assertTrue((res["N1"] >= 0).all())
            self.assertTrue((res["N2"] >= 0).all())
            self.assertTrue((res["N3"] >= 0).all())
            self.assertTrue(((res["N1"] + res["N2"]) <= TOTAL_PARTICLES).all())

    def test_pareto_selection_input_column_variants(self):
        """select_optimal_configurations must accept DataFrames with N1/N2/Delta or N1_pred/N2_pred/Delta_pred."""
        import tempfile
        variant_df = pd.DataFrame({
            "Alfa": [50.0, 55.0],
            "Beta": [50.0, 55.0],
            "H1": [0.02, 0.03],
            "H2": [0.02, 0.03],
            "N1": [150.0, 200.0],
            "N2": [5500.0, 5200.0],
            "Delta": [5350.0, 5000.0],
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_excel = Path(tmp_dir) / "test_variant.xlsx"
            res = select_optimal_configurations(
                pareto_df=variant_df,
                output_excel_path=out_excel,
                n_clusters=2,
                round_decimals=True,
            )
            self.assertEqual(len(res), 2)
            self.assertEqual((res["N1"] + res["N2"] + res["N3"]).iloc[0], TOTAL_PARTICLES)

    def test_compute_non_dominated_fronts_without_pymoo(self):
        """Fast non-dominated sorting fallback (Deb et al.) must correctly partition fronts without pymoo."""
        from unittest.mock import patch
        from src.genetic import compute_non_dominated_fronts

        # Force pymoo import failure to exercise built-in fallback algorithm
        with patch.dict("sys.modules", {"pymoo": None, "pymoo.util.nds.non_dominated_sorting": None}):
            # Minimization objectives: (N1, -Delta)
            # Point 0: N1=10, -Delta=-5000 (Dominates point 2)
            # Point 1: N1=5,  -Delta=-3000 (Non-dominated with point 0: smaller N1, smaller Delta)
            # Point 2: N1=20, -Delta=-4000 (Dominated by point 0: worse N1, worse Delta)
            fitnesses = np.array([
                [10.0, -5000.0],
                [5.0,  -3000.0],
                [20.0, -4000.0],
            ])
            fronts = compute_non_dominated_fronts(fitnesses)

            self.assertEqual(len(fronts), 2)
            # Front 0: points 0 and 1
            self.assertEqual(set(fronts[0]), {0, 1})
            # Front 1: point 2
            self.assertEqual(set(fronts[1]), {2})


if __name__ == "__main__":
    unittest.main()
