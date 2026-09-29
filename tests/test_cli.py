"""
Comprehensive test suite for Phase 5 refactoring:
- PipelineContext state encapsulation and path resolution
- Universal CLI argument parsing and flags handling
- Single rendering guarantee (no duplicate Figure 2 rendering in --all)
- Demo mode propagation to all pipeline stages
- Thin wrappers in src.__main__ and src.pipeline
"""

from __future__ import annotations

import argparse
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from config.path import (
    config_dir,
    demo_data_file,
    processed_data_dir,
)
import src.cli as cli
from src.cli import (
    PipelineContext,
    build_parser,
    execute_pipeline,
    main as cli_main,
)
import src.__main__ as src_main
import src.pipeline as src_pipeline


class MockSurrogateModel:
    """Mock surrogate model for fast testing without training overhead."""

    def predict(self, X):
        import numpy as np

        n = len(X)
        return np.column_stack([np.full(n, 10.0), np.full(n, 5500.0)])


class TestPipelineContext(unittest.TestCase):
    """Tests for PipelineContext data structure and lifecycle methods."""

    def test_default_initialization(self):
        ctx = PipelineContext()
        self.assertFalse(ctx.demo)
        self.assertEqual(ctx.step, "all")
        self.assertIsNone(ctx.figure)
        self.assertEqual(ctx.pop_size, 100)
        self.assertEqual(ctx.n_gen, 25)
        self.assertEqual(ctx.n_trials, 200)
        self.assertEqual(ctx.n_points, 100)
        self.assertEqual(ctx.formats, ("pdf", "svg", "png"))
        self.assertIsInstance(ctx.results, dict)

    def test_custom_initialization_and_normalization(self):
        ctx = PipelineContext(
            demo=True,
            step="eval",  # should normalize to cv
            figure=2,  # should normalize to "2"
            output_dir="custom_output",
            formats=["PNG", ".SVG", "pdf"],
            pop_size=50,
            n_gen=10,
        )
        self.assertTrue(ctx.demo)
        self.assertEqual(ctx.step, "cv")
        self.assertEqual(ctx.figure, "2")
        self.assertIsInstance(ctx.output_dir, Path)
        self.assertEqual(ctx.formats, ("png", "svg", "pdf"))
        self.assertEqual(ctx.pop_size, 50)
        self.assertEqual(ctx.n_gen, 10)

    def test_step_alias_normalization(self):
        ctx_eval = PipelineContext(step="eval")
        self.assertEqual(ctx_eval.step, "cv")

        ctx_select = PipelineContext(step="select")
        self.assertEqual(ctx_select.step, "pareto")

    def test_resolve_paths_demo_mode(self):
        ctx = PipelineContext(demo=True)
        raw, proc, feat_json = ctx.resolve_paths()

        self.assertEqual(raw.resolve(), demo_data_file.resolve())
        self.assertEqual(proc.name, "df_selected_demo.csv")
        self.assertEqual(feat_json.resolve(), (config_dir / "selected_features.json").resolve())
        self.assertNotEqual(raw.resolve(), proc.resolve())

    def test_ensure_model_cached(self):
        mock_model = MockSurrogateModel()
        ctx = PipelineContext(model=mock_model)
        retrieved = ctx.ensure_model()
        self.assertIs(retrieved, mock_model)

    def test_ensure_model_fallback_train(self):
        mock_model = MockSurrogateModel()
        ctx = PipelineContext()
        with patch("src.cli.load_model", side_effect=FileNotFoundError), patch(
            "src.cli.train_final_model", return_value=(mock_model, Path("dummy.joblib"))
        ) as mock_train:
            model = ctx.ensure_model()
            self.assertIs(model, mock_model)
            self.assertIs(ctx.model, mock_model)
            mock_train.assert_called_once()

    def test_from_args_explicit_step(self):
        parser = build_parser()
        args = parser.parse_args(["--step", "optimize", "--pop-size", "80", "--n-gen", "20"])
        ctx = PipelineContext.from_args(args)
        self.assertEqual(ctx.step, "optimize")
        self.assertEqual(ctx.pop_size, 80)
        self.assertEqual(ctx.n_gen, 20)

    def test_from_args_shortcut_flags(self):
        parser = build_parser()

        # --all
        ctx_all = PipelineContext.from_args(parser.parse_args(["--all"]))
        self.assertEqual(ctx_all.step, "all")

        # --cv
        ctx_cv = PipelineContext.from_args(parser.parse_args(["--cv"]))
        self.assertEqual(ctx_cv.step, "cv")

        # --train
        ctx_tr = PipelineContext.from_args(parser.parse_args(["--train"]))
        self.assertEqual(ctx_tr.step, "train")

        # --optimize
        ctx_opt = PipelineContext.from_args(parser.parse_args(["--optimize"]))
        self.assertEqual(ctx_opt.step, "optimize")

        # --all-figures
        ctx_fig_all = PipelineContext.from_args(parser.parse_args(["--all-figures"]))
        self.assertEqual(ctx_fig_all.step, "figures")
        self.assertEqual(ctx_fig_all.figure, "all")

        # --figure 4
        ctx_fig4 = PipelineContext.from_args(parser.parse_args(["--figure", "4"]))
        self.assertEqual(ctx_fig4.step, "figures")
        self.assertEqual(ctx_fig4.figure, "4")

    def test_from_args_default_step_propagation(self):
        parser = build_parser()
        args = parser.parse_args([])
        ctx = PipelineContext.from_args(args, default_step="all")
        self.assertEqual(ctx.step, "all")

        ctx_none = PipelineContext.from_args(args, default_step=None)
        self.assertIsNone(ctx_none.step)


class TestCLIFunctionality(unittest.TestCase):
    """Tests for argument parsing, command execution, and step routing."""

    def test_build_parser_options(self):
        parser = build_parser()
        # Verify choices and expected arguments
        actions = {a.dest: a for a in parser._actions}
        self.assertIn("step", actions)
        self.assertIn("figure", actions)
        self.assertIn("demo", actions)
        self.assertIn("all", actions)
        self.assertIn("cv", actions)
        self.assertIn("train", actions)
        self.assertIn("optimize", actions)
        self.assertIn("all_figures", actions)
        self.assertIn("pop_size", actions)
        self.assertIn("n_gen", actions)
        self.assertIn("n_trials", actions)
        self.assertIn("formats", actions)

    def test_main_no_arguments_prints_help(self):
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            exit_code = cli_main(argv=[], default_step=None)
            self.assertEqual(exit_code, 0)
            output = fake_out.getvalue()
            self.assertIn("usage:", output.lower())
            self.assertIn("--step", output)

    def test_main_demo_propagation_cv(self):
        mock_folds = pd.DataFrame({"fold": [1]})
        mock_summary = pd.DataFrame({"metric": ["MAE"]})
        mock_oof = pd.DataFrame({"N1_true": [1.0], "N1_pred_oof": [1.0], "N2_true": [2.0], "N2_pred_oof": [2.0], "Delta_true": [1.0], "Delta_pred_oof": [1.0]})

        with patch("src.cli.run_step_cv", return_value=(mock_folds, mock_summary, mock_oof)) as mock_cv:
            exit_code = cli_main(argv=["--demo", "--step", "cv"])
            self.assertEqual(exit_code, 0)
            mock_cv.assert_called_once()
            called_ctx = mock_cv.call_args[0][0]
            self.assertTrue(called_ctx.demo)
            self.assertEqual(called_ctx.step, "cv")

    def test_main_figure_single_delegation(self):
        with patch("src.cli.generate_figure_2", return_value={"status": "ok"}) as mock_f2:
            exit_code = cli_main(argv=["--demo", "--figure", "2"])
            self.assertEqual(exit_code, 0)
            mock_f2.assert_called_once()
            kwargs = mock_f2.call_args[1]
            self.assertTrue(kwargs["use_demo"])

    def test_main_figure_all_delegation(self):
        with patch("src.cli.generate_all_figures", return_value={"Fig2": "ok"}) as mock_all_figs:
            exit_code = cli_main(argv=["--demo", "--figure", "all"])
            self.assertEqual(exit_code, 0)
            mock_all_figs.assert_called_once()
            kwargs = mock_all_figs.call_args[1]
            self.assertTrue(kwargs["use_demo"])

    def test_no_double_rendering_in_step_all(self):
        """
        Critical requirement test:
        When running step 'all', evaluate_cv must be called with save_plot_path=None
        (save_plot=False) so Figure 2 is NOT rendered during CV,
        and Figure 2 is only rendered during run_step_figures.
        """
        mock_folds = pd.DataFrame({"fold": [1]})
        mock_summary = pd.DataFrame({"metric": ["MAE"]})
        mock_oof = pd.DataFrame({
            "N1_true": [10.0], "N1_pred_oof": [10.0],
            "N2_true": [5000.0], "N2_pred_oof": [5000.0],
            "Delta_true": [4990.0], "Delta_pred_oof": [4990.0],
        })

        ctx = PipelineContext(demo=True, step="all")

        with patch("src.cli.run_step_select_features"), \
             patch("src.cli.run_step_tune"), \
             patch("src.cli.run_step_train"), \
             patch("src.cli.evaluate_cv", return_value=(mock_folds, mock_summary, mock_oof)) as mock_eval_cv, \
             patch("src.cli.evaluate_train_test", return_value=(None, {"R2": 0.99}, None, None)), \
             patch("src.cli.run_step_optimize"), \
             patch("src.cli.run_step_pareto"), \
             patch("src.cli.run_step_sensitivity"), \
             patch("src.cli.generate_all_figures", return_value={}) as mock_gen_all, \
             patch("src.models.plot_actual_vs_predicted") as mock_plot_actual:

            execute_pipeline(ctx)

            # 1. evaluate_cv was called with save_plot_path=None
            mock_eval_cv.assert_called_once()
            self.assertIsNone(mock_eval_cv.call_args[1].get("save_plot_path"))

            # 2. plot_actual_vs_predicted was NEVER called during CV
            mock_plot_actual.assert_not_called()

            # 3. generate_all_figures was called exactly once in the figures step
            mock_gen_all.assert_called_once()

    def test_standalone_cv_saves_diagnostic_plot(self):
        """When running standalone --step cv, save_plot=True and diagnostic plot is requested."""
        mock_folds = pd.DataFrame({"fold": [1]})
        mock_summary = pd.DataFrame({"metric": ["MAE"]})
        mock_oof = pd.DataFrame({
            "N1_true": [10.0], "N1_pred_oof": [10.0],
            "N2_true": [5000.0], "N2_pred_oof": [5000.0],
            "Delta_true": [4990.0], "Delta_pred_oof": [4990.0],
        })

        ctx = PipelineContext(demo=True, step="cv")
        with patch("src.cli.evaluate_cv", return_value=(mock_folds, mock_summary, mock_oof)) as mock_eval_cv, \
             patch("src.cli.evaluate_train_test", return_value=(None, {"R2": 0.99}, None, None)):

            execute_pipeline(ctx)
            mock_eval_cv.assert_called_once()
            self.assertIsNotNone(mock_eval_cv.call_args[1].get("save_plot_path"))

    def test_invalid_step_argument(self):
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--step", "non_existent_step"])

    def test_invalid_figure_argument(self):
        with patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--figure", "99"])


    def test_case_insensitive_and_prefixed_inputs(self):
        """Case-insensitivity and prefix tolerance for step and figure arguments."""
        self.assertEqual(PipelineContext(step="CV").step, "cv")
        self.assertEqual(PipelineContext(step="EVAL").step, "cv")
        self.assertEqual(PipelineContext(step="SELECT").step, "pareto")
        self.assertEqual(PipelineContext(step="ALL").step, "all")
        self.assertEqual(PipelineContext(figure="Fig2").figure, "2")
        self.assertEqual(PipelineContext(figure="Figure_4").figure, "4")
        self.assertEqual(PipelineContext(figure="FIGURE 5").figure, "5")

        parser = build_parser()
        self.assertEqual(parser.parse_args(["--step", "CV"]).step, "cv")
        self.assertEqual(parser.parse_args(["--step", "ALL"]).step, "all")
        self.assertEqual(parser.parse_args(["--figure", "Fig3"]).figure, "3")
        self.assertEqual(parser.parse_args(["--figure", "figure_6"]).figure, "6")

    def test_run_step_pareto_fallback_optimize_when_no_pareto_files(self):
        """When no pareto files exist in results, run_step_pareto triggers run_step_optimize."""
        ctx = PipelineContext(demo=True, step="pareto")
        mock_pareto = pd.DataFrame({"Alfa": [50.0], "Beta": [50.0], "H1": [0.03], "H2": [0.03], "N1": [10.0], "Delta": [5000.0], "N2": [5010.0]})

        with patch("src.cli.optimization_results_dir") as mock_dir, \
             patch("src.cli.run_step_optimize", return_value=(None, None, mock_pareto, None)) as mock_opt, \
             patch("src.cli.select_optimal_configurations", return_value=mock_pareto) as mock_sel:

            mock_dir.__truediv__.return_value.exists.return_value = False
            mock_dir.glob.return_value = []

            cli.run_step_pareto(ctx)
            mock_opt.assert_called_once_with(ctx)
            mock_sel.assert_called_once()

    def test_run_step_pareto_custom_output_dir(self):
        """When context.output_dir is set, select_optimal_configurations receives custom path."""
        ctx = PipelineContext(output_dir=Path("custom/out_dir"), step="pareto")
        dummy_df = pd.DataFrame({"Alfa": [50.0]})

        with patch("src.cli.select_optimal_configurations", return_value=dummy_df) as mock_sel:
            cli.run_step_pareto(ctx)
            mock_sel.assert_called_once()
            self.assertEqual(
                mock_sel.call_args[1].get("output_excel_path"),
                Path("custom/out_dir/pareto_optimal_designs.xlsx"),
            )

    def test_run_step_sensitivity_uses_ensure_model_and_output_dir(self):
        """run_step_sensitivity passes cached model, n_points, and output_dir."""
        ctx = PipelineContext(output_dir=Path("custom/sens_out"), n_points=42)
        mock_model = MockSurrogateModel()
        ctx.model = mock_model

        with patch("src.cli.run_sensitivity_analysis", return_value={}) as mock_sens:
            cli.run_step_sensitivity(ctx)
            mock_sens.assert_called_once_with(
                model=mock_model,
                n_points=42,
                output_dir=Path("custom/sens_out"),
            )

    def test_run_step_figures_n_points_forwarding(self):
        """n_points hyperparameter is forwarded to generate_figure_4 and generate_all_figures."""
        ctx_f4 = PipelineContext(step="figures", figure="4", n_points=75)
        with patch("src.cli.generate_figure_4", return_value={}) as mock_f4:
            cli.run_step_figures(ctx_f4)
            mock_f4.assert_called_once()
            self.assertEqual(mock_f4.call_args[1].get("n_points_1d"), 75)

        ctx_all = PipelineContext(step="figures", figure="all", n_points=85)
        with patch("src.cli.generate_all_figures", return_value={}) as mock_all:
            cli.run_step_figures(ctx_all)
            mock_all.assert_called_once()
            self.assertEqual(mock_all.call_args[1].get("n_points_1d"), 85)

    def test_run_step_all_standard_mode_needs_features_if_proc_missing(self):
        """run_step_all executes feature selection if processed_path is missing even with demo=False."""
        ctx = PipelineContext(demo=False, step="all")
        mock_raw = Path("data/raw/inertial_separator_cfd.xlsx")
        mock_proc = MagicMock(spec=Path)
        mock_proc.exists.return_value = False
        mock_feat = MagicMock(spec=Path)
        mock_feat.exists.return_value = True

        with patch.object(ctx, "resolve_paths", return_value=(mock_raw, mock_proc, mock_feat)), \
             patch("src.cli.run_step_select_features") as mock_select, \
             patch("src.cli.load_best_params"), \
             patch("src.cli.run_step_train"), \
             patch("src.cli.run_step_cv"), \
             patch("src.cli.run_step_optimize"), \
             patch("src.cli.run_step_pareto"), \
             patch("src.cli.run_step_sensitivity"), \
             patch("src.cli.run_step_figures"):

            cli.run_step_all(ctx)
            mock_select.assert_called_once()


class TestThinWrappersAndCompatibility(unittest.TestCase):
    """Tests ensuring src.__main__ and src.pipeline maintain 100% backward compatibility."""

    def test_src_main_delegates_to_cli_main(self):
        with patch("src.cli.main", return_value=0) as mock_cli_main:
            # Test calling main in src.__main__
            src_main.main(["--demo", "--all"])
            mock_cli_main.assert_called_once_with(["--demo", "--all"])

    def test_src_pipeline_main_delegates_to_cli_main(self):
        with patch("src.cli.main", return_value=0) as mock_cli_main:
            src_pipeline.main(["--demo", "--step", "train"])
            mock_cli_main.assert_called_once_with(argv=["--demo", "--step", "train"], default_step="all")

    def test_src_pipeline_orchestrator_functions_remain_callable(self):
        self.assertTrue(callable(src_pipeline.resolve_data_paths))
        self.assertTrue(callable(src_pipeline.run_cv_pipeline))
        self.assertTrue(callable(src_pipeline.run_training_pipeline))
        self.assertTrue(callable(src_pipeline.run_optimization_pipeline))

    def test_src_pipeline_run_cv_pipeline_save_plot_flag(self):
        """run_cv_pipeline accepts save_plot keyword argument."""
        mock_folds = pd.DataFrame({"fold": [1]})
        mock_summary = pd.DataFrame({"metric": ["MAE"]})
        mock_oof = pd.DataFrame({"N1_true": [1.0]})

        with patch("src.pipeline.evaluate_cv", return_value=(mock_folds, mock_summary, mock_oof)) as mock_cv:
            src_pipeline.run_cv_pipeline(save_plot=False)
            self.assertIsNone(mock_cv.call_args[1].get("save_plot_path"))

            src_pipeline.run_cv_pipeline(save_plot=True)
            self.assertIsNotNone(mock_cv.call_args[1].get("save_plot_path"))


if __name__ == "__main__":
    unittest.main()

