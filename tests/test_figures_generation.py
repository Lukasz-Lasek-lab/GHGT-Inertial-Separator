"""
Unit tests for the decomposed visualization module (Phase 2):
- src.visualization.style
- src.visualization.fig2_diagnostics
- src.visualization.fig3_importance
- src.visualization.fig4_sensitivity
- src.visualization.fig5_pareto
- src.visualization.fig6_case_study
- src.visualization.orchestrator
- src.visualization.publication_plots (facade compatibility)
"""

from pathlib import Path
import tempfile
import unittest

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.constants import (
    BASE_FEATURES,
    DEFAULT_PARAM_RANGES,
    DEFAULT_REFERENCE_PARAMS,
    PARAM_BOUNDS,
    TOTAL_PARTICLES,
)
import src.visualization as viz
import src.visualization.fig2_diagnostics as f2
import src.visualization.fig3_importance as f3
import src.visualization.fig4_sensitivity as f4
import src.visualization.fig5_pareto as f5
import src.visualization.fig6_case_study as f6
import src.visualization.figures as legacy_figures
import src.visualization.orchestrator as orch
import src.visualization.publication_plots as pub_plots
import src.visualization.style as vstyle


class TestVisualizationStyle(unittest.TestCase):
    """Tests for publication styles, CVD palettes, and figure export utilities."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_okabe_ito_palette(self):
        self.assertEqual(len(vstyle.OKABE_ITO), 8)
        self.assertEqual(len(set(vstyle.OKABE_ITO)), 8)
        for hex_code in vstyle.OKABE_ITO:
            self.assertTrue(hex_code.startswith("#"))
            self.assertEqual(len(hex_code), 7)

    def test_publication_style_context(self):
        orig_dpi = mpl.rcParams["figure.dpi"]
        orig_font = mpl.rcParams["font.family"]

        with vstyle.publication_style():
            self.assertEqual(mpl.rcParams["figure.dpi"], 300)
            self.assertEqual(mpl.rcParams["font.family"], ["sans-serif"])
            self.assertTrue(mpl.rcParams["axes.grid"])

        # Reverted back
        self.assertEqual(mpl.rcParams["figure.dpi"], orig_dpi)
        self.assertEqual(mpl.rcParams["font.family"], orig_font)

    def test_add_panel_label(self):
        fig, ax = plt.subplots()
        vstyle.add_panel_label(ax, "(a)", loc="top_left")
        vstyle.add_panel_label(ax, "(b)", loc="outside_top_left")
        texts = [t.get_text() for t in ax.texts]
        self.assertIn("(a)", texts)
        self.assertIn("(b)", texts)

    def test_save_publication_figure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            res = vstyle.save_publication_figure(
                fig=fig,
                figure_name="test_fig",
                output_dir=tmp_dir,
                formats=["png", "pdf", "svg"],
                close_fig=True,
            )
            for fmt in ["png", "pdf", "svg"]:
                self.assertIn(fmt, res)
                p = Path(res[fmt])
                self.assertTrue(p.exists())
                self.assertGreater(p.stat().st_size, 0)

    def test_save_publication_figure_path_object(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [1, 2])
            res = vstyle.save_publication_figure(
                fig=fig,
                figure_name=Path("test_path_fig.png"),
                output_dir=Path(tmp_dir),
                formats=["png"],
                close_fig=True,
            )
            self.assertIn("png", res)
            self.assertTrue(Path(res["png"]).exists())


class TestFig2Diagnostics(unittest.TestCase):
    """Tests for Figure 2 parity panel and residual distribution renderers."""

    def setUp(self):
        plt.close("all")
        np.random.seed(42)
        n = 50
        self.y_true = np.linspace(100, 5000, n)
        self.y_pred = self.y_true + np.random.normal(0, 50, n)
        self.y_true_df = pd.DataFrame({
            "N1": self.y_true[:n],
            "N2": self.y_true[:n] + 500,
            "Delta": 500.0,
        })
        self.y_pred_df = pd.DataFrame({
            "N1": self.y_pred[:n],
            "N2": self.y_pred[:n] + 500,
            "Delta": 500.0,
        })

    def tearDown(self):
        plt.close("all")

    def test_draw_parity_panel(self):
        fig, ax = plt.subplots()
        metrics = f2.draw_parity_panel(
            ax=ax,
            y_true=self.y_true,
            y_pred=self.y_pred,
            target_name="$N_1$",
            target_label="Loss",
            panel_label="(a)",
        )
        self.assertIn("r2", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertGreater(metrics["r2"], 0.95)
        self.assertGreater(len(ax.collections), 0)  # scatter
        self.assertGreater(len(ax.lines), 0)  # identity line

    def test_draw_residuals_panel(self):
        fig, ax = plt.subplots()
        f2.draw_residuals_panel(
            ax=ax,
            y_true_df=self.y_true_df,
            y_pred_df=self.y_pred_df,
            panel_label="(d)",
        )
        self.assertGreater(len(ax.patches), 0)  # hist bars
        self.assertGreater(len(ax.lines), 0)  # zero line

    def test_draw_residuals_panel_direct_residuals(self):
        # Test draw_residuals_panel(ax, residuals) with 1D numpy array and with dict
        res_arr = np.random.normal(0, 10, 50)
        fig, ax = plt.subplots()
        f2.draw_residuals_panel(ax=ax, y_true_df=res_arr)
        self.assertGreater(len(ax.patches), 0)

        # Test with dict
        res_dict = {"N1": res_arr, "N2": res_arr * 1.5}
        fig2, ax2 = plt.subplots()
        f2.draw_residuals_panel(ax=ax2, residuals_dict=res_dict)
        self.assertGreater(len(ax2.patches), 0)

    def test_plot_parity_single_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f2.plot_parity_single(
                y_true=self.y_true,
                y_pred=self.y_pred,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_residuals_single_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f2.plot_residuals_single(
                y_true_df=self.y_true_df,
                y_pred_df=self.y_pred_df,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_model_diagnostics_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f2.plot_model_diagnostics(
                y_true_df=self.y_true_df,
                y_pred_df=self.y_pred_df,
                output_dir=tmp_dir,
                formats=["png"],
                save_individual=True,
                individual_prefix="TestFig2",
            )
            self.assertIn("combined_grid", res)
            self.assertTrue(Path(res["combined_grid"]["png"]).exists())
            self.assertIn("TestFig2a_parity_N1", res)
            self.assertIn("TestFig2d_residuals_distribution", res)


class TestFig3Importance(unittest.TestCase):
    """Tests for Figure 3 feature importance bars and multi-objective comparisons."""

    def setUp(self):
        plt.close("all")
        self.df_imp = pd.DataFrame([
            {"feature": "Alfa", "importance_pct": 25.5, "std_pct": 1.2},
            {"feature": "Beta", "importance_pct": 15.0, "std_pct": 0.8},
            {"feature": "H1", "importance_pct": 40.0, "std_pct": 2.0},
            {"feature": "H1_plus_H2", "importance_pct": 18.0, "std_pct": 1.0},
            {"feature": "negligible", "importance_pct": 0.05, "std_pct": 0.01},
        ])
        self.imp_dict = {
            "N1": self.df_imp,
            "N2": self.df_imp.copy(),
            "Delta": self.df_imp.copy(),
        }

    def tearDown(self):
        plt.close("all")

    def test_draw_importance_bars(self):
        fig, ax = plt.subplots()
        df_sorted = f3.draw_importance_bars(
            ax=ax,
            df_importance=self.df_imp,
            target_name="$N_1$",
            panel_label="(a)",
        )
        # Negligible feature (<0.1%) should be filtered out
        self.assertNotIn("negligible", df_sorted["feature"].values)
        self.assertEqual(len(ax.patches), 4)

    def test_draw_importance_comparison(self):
        fig, ax = plt.subplots()
        sorted_feats = f3.draw_importance_comparison(
            ax=ax,
            importance_dict=self.imp_dict,
            panel_label="(d)",
        )
        self.assertEqual(len(sorted_feats), 4)
        self.assertGreater(len(ax.patches), 0)

    def test_draw_importance_comparison_reduced_targets(self):
        # Test with 1 and 2 targets to verify dynamic centering
        fig1, ax1 = plt.subplots()
        f3.draw_importance_comparison(ax=ax1, importance_dict={"N1": self.df_imp})
        self.assertEqual(len(ax1.patches), 4)

        fig2, ax2 = plt.subplots()
        f3.draw_importance_comparison(ax=ax2, importance_dict={"N1": self.df_imp, "N2": self.df_imp})
        self.assertEqual(len(ax2.patches), 8)

    def test_plot_feature_importance_single_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f3.plot_feature_importance_single(
                importance_df=self.df_imp,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_feature_importance_comparison_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f3.plot_feature_importance_comparison(
                importance_dict=self.imp_dict,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_feature_importance_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f3.plot_feature_importance(
                importance_data=self.imp_dict,
                output_dir=tmp_dir,
                formats=["png"],
                save_individual=True,
                individual_prefix="TestFig3",
            )
            self.assertIn("combined_grid", res)
            self.assertTrue(Path(res["combined_grid"]["png"]).exists())
            self.assertIn("TestFig3a_importance_N1", res)
            self.assertIn("TestFig3d_importance_comparison", res)


class TestFig4Sensitivity(unittest.TestCase):
    """Tests for Figure 4 1D sweep profiles and 2D contour interactions."""

    def setUp(self):
        plt.close("all")
        vals = np.linspace(42.75, 60.0, 30)
        self.df_param = pd.DataFrame({
            "Alfa": vals,
            "N1_pred": np.linspace(300, 50, 30),
            "N2_pred": np.linspace(5000, 5800, 30),
            "Delta_pred": np.linspace(4700, 5750, 30),
        })
        self.sweep_data = {"Alfa": self.df_param}
        self.ref_params = {"Alfa": 60.0}

    def tearDown(self):
        plt.close("all")

    def test_draw_sweep_1d_panel(self):
        fig, ax1 = plt.subplots()
        ax2 = f4.draw_sweep_1d_panel(
            ax1=ax1,
            df_param=self.df_param,
            param="Alfa",
            ref_val=60.0,
            panel_label="(a)",
        )
        self.assertIsNotNone(ax2)
        self.assertGreater(len(ax1.lines), 0)
        self.assertGreater(len(ax2.lines), 0)

    def test_draw_sweep_1d_panel_alternate_columns(self):
        # Test with N1 / N2 instead of N1_pred / N2_pred
        df_alt = pd.DataFrame({
            "Alfa": np.linspace(43, 60, 20),
            "N1": np.linspace(200, 50, 20),
            "N2": np.linspace(5100, 5800, 20),
        })
        fig, ax1 = plt.subplots()
        ax2 = f4.draw_sweep_1d_panel(ax1=ax1, df_param=df_alt, param="Alfa", ref_val=50.0)
        self.assertIsNotNone(ax2)
        self.assertGreater(len(ax1.lines), 0)

    def test_draw_contour_2d_panel(self):
        fig, ax = plt.subplots()
        v = np.linspace(0, 1, 10)
        G1, G2 = np.meshgrid(v, v)
        Z = G1 + G2
        cf = f4.draw_contour_2d_panel(
            ax=ax,
            G1=G1,
            G2=G2,
            Z=Z,
            p1="Alfa",
            p2="Beta",
            target="N1",
            fig=fig,
        )
        self.assertIsNotNone(cf)

    def test_plot_sensitivity_1d_single_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f4.plot_sensitivity_1d_single(
                df_param=self.df_param,
                param="Alfa",
                ref_val=60.0,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_sensitivity_heatmap_single_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            v = np.linspace(0, 1, 10)
            G1, G2 = np.meshgrid(v, v)
            Z = G1 + G2
            res = f4.plot_sensitivity_heatmap_single(
                G1=G1,
                G2=G2,
                Z=Z,
                p1="Alfa",
                p2="Beta",
                target="N1",
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_sensitivity_sweeps_1d_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f4.plot_sensitivity_sweeps_1d(
                sensitivity_data=self.sweep_data,
                ref_params=self.ref_params,
                output_dir=tmp_dir,
                formats=["png"],
                save_individual=True,
                individual_prefix="TestFig4",
            )
            self.assertIn("combined_grid", res)
            self.assertTrue(Path(res["combined_grid"]["png"]).exists())


class TestFig5Pareto(unittest.TestCase):
    """Tests for Figure 5 generational convergence, Pareto front, and physics cluster resolution."""

    def setUp(self):
        plt.close("all")
        gens = np.arange(10)
        self.log_df = pd.DataFrame({
            "gen": gens,
            "n1_min": np.linspace(500, 5, 10),
            "n1_avg": np.linspace(1500, 200, 10),
            "n1_max": np.linspace(3000, 800, 10),
            "delta_min": np.linspace(3500, 5000, 10),
            "delta_avg": np.linspace(4000, 5500, 10),
            "delta_max": np.linspace(4500, 5860, 10),
        })

        self.all_pop_df = pd.DataFrame({
            "N1_pred": np.random.uniform(0, 600, 100),
            "Delta_pred": np.random.uniform(3500, 5800, 100),
        })

        # Test designs with high gap (H2 ~ 59 mm) and low gap (H2 ~ 39 mm)
        self.pareto_df = pd.DataFrame([
            {"N1_pred": 0.0, "Delta_pred": 5850.0, "H2": 0.0588, "cluster": 0, "Alfa": 43.5, "Beta": 48.2, "H1": 0.032},
            {"N1_pred": 12.0, "Delta_pred": 5865.0, "H2": 0.0386, "cluster": 1, "Alfa": 43.0, "Beta": 48.0, "H1": 0.027},
        ])

    def tearDown(self):
        plt.close("all")

    def test_draw_convergence(self):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        f5.draw_convergence(ax1, ax2, self.log_df)
        self.assertEqual(len(ax1.lines), 3)
        self.assertEqual(len(ax2.lines), 3)

    def test_physics_cluster_labels_resolution(self):
        # Cluster 0 has H2=0.0588 (~59mm), Cluster 1 has H2=0.0386 (~39mm)
        labels = f5._resolve_cluster_labels(self.pareto_df, cluster_col="cluster")
        self.assertIn("59", labels[0])
        self.assertIn("39", labels[1])

        # Test legacy wrong map inversion fix
        wrong_map = {0: "Cluster ($H_2 \\approx 39\\text{ mm}$)", 1: "Cluster ($H_2 \\approx 59\\text{ mm}$)"}
        corrected = f5._resolve_cluster_labels(self.pareto_df, cluster_col="cluster", cluster_labels_map=wrong_map)
        self.assertIn("59", corrected[0])
        self.assertIn("39", corrected[1])

    def test_physics_cluster_labels_with_generic_placeholder(self):
        # Even if legacy dictionary has generic "Pareto cluster 3", it should be resolved with physical H2
        generic_map = {0: "Pareto cluster 1", 1: "Pareto cluster 2"}
        corrected = f5._resolve_cluster_labels(self.pareto_df, cluster_col="cluster", cluster_labels_map=generic_map)
        self.assertIn("H_2", corrected[0])
        self.assertIn("H_2", corrected[1])

    def test_plot_convergence_single_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f5.plot_convergence_single(
                log_df=self.log_df,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_pareto_front_single_standalone(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f5.plot_pareto_front_single(
                all_population_df=self.all_pop_df,
                pareto_df=self.pareto_df,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_draw_inset_zoom_outlier_adaptation(self):
        # Data that falls completely outside default inset bounds (-0.05, 1.0) and (5840, 5866)
        outlier_pareto = pd.DataFrame([
            {"N1_pred": 50.0, "Delta_pred": 4900.0, "H2": 0.040, "cluster": 0},
            {"N1_pred": 60.0, "Delta_pred": 5100.0, "H2": 0.050, "cluster": 1},
        ])
        fig, ax = plt.subplots()
        f5.draw_pareto_front(ax=ax, all_population_df=self.all_pop_df, pareto_df=outlier_pareto)
        ax_ins = f5.draw_inset_zoom(ax_parent=ax, pareto_df=outlier_pareto)
        self.assertIsNotNone(ax_ins)
        xlim = ax_ins.get_xlim()
        ylim = ax_ins.get_ylim()
        # Verify inset adapted around 50 and 5100 rather than 0 and 5850
        self.assertTrue(xlim[0] <= 55 <= xlim[1])
        self.assertTrue(ylim[0] <= 5050 <= ylim[1])

    def test_draw_pareto_front_and_inset(self):
        fig, ax = plt.subplots()
        f5.draw_pareto_front(
            ax=ax,
            all_population_df=self.all_pop_df,
            pareto_df=self.pareto_df,
            baseline_point={"N1": 242.0, "Delta": 5464.0},
        )
        self.assertGreater(len(ax.collections), 0)

        ax_ins = f5.draw_inset_zoom(
            ax_parent=ax,
            pareto_df=self.pareto_df,
        )
        self.assertIsNotNone(ax_ins)

    def test_plot_optimization_figure_5_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f5.plot_optimization_figure_5(
                log_df=self.log_df,
                all_population_df=self.all_pop_df,
                pareto_df=self.pareto_df,
                baseline_point={"N1": 242.0, "Delta": 5464.0},
                output_dir=tmp_dir,
                formats=["png"],
                save_individual=True,
                individual_prefix="TestFig5",
            )
            self.assertIn("combined_grid", res)
            self.assertTrue(Path(res["combined_grid"]["png"]).exists())
            self.assertIn("TestFig5a_nsga2_convergence", res)
            self.assertIn("TestFig5b_pareto_front", res)


class TestFig6CaseStudy(unittest.TestCase):
    """Tests for Figure 6 engineering validation case study and dynamic Pareto fallback."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_resolve_case_study_data_fallback(self):
        # Without model and without file, returns CFD baseline + paper constants
        df_comp = f6.resolve_case_study_data(model=None, pareto_csv_path="non_existent_file.csv")
        self.assertEqual(len(df_comp), 3)
        self.assertIn("Baseline CFD\n(Nominal)", df_comp["variant"].values)
        self.assertIn("Pareto Opt. 1\n(Zero loss)", df_comp["variant"].values)
        self.assertIn("Pareto Opt. 2\n(Max capture)", df_comp["variant"].values)
        for col in ["eta_1_pct", "eta_2_pct", "Delta"]:
            self.assertIn(col, df_comp.columns)

    def test_resolve_case_study_data_from_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "pareto_optimal_designs.csv"
            df_mock = pd.DataFrame([
                {"Alfa": 44.0, "Beta": 47.0, "H1": 0.030, "H2": 0.059, "N1_pred": 1.0, "Delta_pred": 5840.0},
                {"Alfa": 43.0, "Beta": 48.0, "H1": 0.028, "H2": 0.038, "N1_pred": 20.0, "Delta_pred": 5870.0},
            ])
            df_mock.to_csv(csv_path, index=False)

            df_comp = f6.resolve_case_study_data(model=None, pareto_csv_path=csv_path)
            self.assertEqual(len(df_comp), 3)
            # Opt 1 has minimum N1 (H2=0.059)
            row_opt1 = df_comp[df_comp["variant"] == "Pareto Opt. 1\n(Zero loss)"].iloc[0]
            self.assertAlmostEqual(row_opt1["H2"], 0.059, places=3)
            self.assertAlmostEqual(row_opt1["N1"], 1.0, places=3)
            self.assertAlmostEqual(row_opt1["Delta"], 5840.0, places=3)

            # Opt 2 has maximum Delta (H2=0.038)
            row_opt2 = df_comp[df_comp["variant"] == "Pareto Opt. 2\n(Max capture)"].iloc[0]
            self.assertAlmostEqual(row_opt2["H2"], 0.038, places=3)
            self.assertAlmostEqual(row_opt2["N1"], 20.0, places=3)
            self.assertAlmostEqual(row_opt2["Delta"], 5870.0, places=3)

    def test_draw_advantage_bars_dynamic_counts(self):
        # Test with 2 rows and 4 rows
        df_2 = pd.DataFrame({
            "variant": ["Base", "Opt1"],
            "Delta": [5400.0, 5800.0],
        })
        fig1, ax1 = plt.subplots()
        f6.draw_advantage_bars(ax=ax1, comparison_df=df_2)
        self.assertEqual(len(ax1.patches), 2)

        df_4 = pd.DataFrame({
            "variant": ["Base", "Opt1", "Opt2", "Opt3"],
            "Delta": [5400.0, 5800.0, 5850.0, 5900.0],
        })
        fig2, ax2 = plt.subplots()
        f6.draw_advantage_bars(ax=ax2, comparison_df=df_4)
        self.assertEqual(len(ax2.patches), 4)

    def test_plot_case_study_efficiency_standalone(self):
        df_comp = f6.resolve_case_study_data(model=None, pareto_csv_path="non_existent.csv")
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f6.plot_case_study_efficiency(
                comparison_df=df_comp,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_case_study_advantage_standalone(self):
        df_comp = f6.resolve_case_study_data(model=None, pareto_csv_path="non_existent.csv")
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f6.plot_case_study_advantage(
                comparison_df=df_comp,
                output_dir=tmp_dir,
                formats=["png"],
            )
            self.assertTrue(Path(res["png"]).exists())

    def test_plot_case_study_comparison_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = f6.plot_case_study_comparison(
                comparison_df=None,
                output_dir=tmp_dir,
                formats=["png"],
                save_individual=True,
                individual_prefix="TestFig6",
            )
            self.assertIn("combined_grid", res)
            self.assertTrue(Path(res["combined_grid"]["png"]).exists())
            self.assertIn("TestFig6a_efficiency_comparison", res)
            self.assertIn("TestFig6b_advantage_comparison", res)


class TestOrchestratorAndFacade(unittest.TestCase):
    """Tests for backward-compatible facade re-exports and orchestrator."""

    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_legacy_figures_facade(self):
        self.assertTrue(hasattr(legacy_figures, "generate_figure_2"))
        self.assertTrue(hasattr(legacy_figures, "generate_figure_3"))
        self.assertTrue(hasattr(legacy_figures, "generate_figure_4"))
        self.assertTrue(hasattr(legacy_figures, "generate_figure_5"))
        self.assertTrue(hasattr(legacy_figures, "generate_figure_6"))
        self.assertTrue(hasattr(legacy_figures, "generate_all_figures"))
        self.assertEqual(legacy_figures.TOTAL_PARTICLES, TOTAL_PARTICLES)
        self.assertEqual(legacy_figures.DEFAULT_PARAM_RANGES, PARAM_BOUNDS)
        self.assertEqual(legacy_figures.DEFAULT_REFERENCE_PARAMS, DEFAULT_REFERENCE_PARAMS)

    def test_publication_plots_facade(self):
        required_symbols = [
            "OKABE_ITO",
            "CVD_PALETTES",
            "LINE_STYLES",
            "MARKERS",
            "FEATURE_LABEL_MAP",
            "PARAM_AXIS_LABEL_MAP",
            "PARAM_NOMINAL_LABEL_MAP",
            "PARAM_BOUNDS",
            "publication_style",
            "set_publication_style",
            "add_panel_label",
            "save_publication_figure",
            "draw_parity_panel",
            "draw_residuals_panel",
            "plot_parity_single",
            "plot_residuals_single",
            "plot_model_diagnostics",
            "draw_importance_bars",
            "plot_feature_importance",
            "draw_sweep_1d_panel",
            "plot_sensitivity_sweeps_1d",
            "plot_sensitivity_heatmaps_2d",
            "draw_convergence",
            "draw_pareto_front",
            "plot_optimization_figure_5",
            "draw_comparison_bars",
            "plot_case_study_comparison",
        ]
        for sym in required_symbols:
            self.assertTrue(hasattr(pub_plots, sym), f"Missing symbol {sym} in publication_plots")

    def test_orchestrator_generate_figure_2_demo(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = orch.generate_figure_2(output_dir=tmp_dir, formats=["png"], use_demo=True)
            self.assertIn("combined_grid", res)
            self.assertTrue(Path(res["combined_grid"]["png"]).exists())

    def test_orchestrator_generate_figure_4_demo(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = orch.generate_figure_4(
                output_dir=tmp_dir,
                formats=["png"],
                n_points_1d=20,
                grid_size_2d=10,
                use_demo=True,
            )
            self.assertIn("sweeps_1d", res)
            self.assertIn("contours_2d", res)

    def test_orchestrator_generate_figure_6_demo(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = orch.generate_figure_6(output_dir=tmp_dir, formats=["png"], use_demo=True)
            self.assertIn("combined_grid", res)
            self.assertTrue(Path(res["combined_grid"]["png"]).exists())


if __name__ == "__main__":
    unittest.main()
