"""
Tests for pipeline execution in demonstration mode and integrity of raw demo data:
- Invariance of SHA256 checksum of data/demo/inertial_separator_demo.csv before and after execution
- Verification that resolve_data_paths(use_demo=True) directs processed output to df_selected_demo.csv
- Verification that demo_data_file is never overwritten
- Verification that config.path has no directory creation side effects on import
"""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from config.path import (
    demo_data_file,
    processed_data_dir,
    config_dir,
)
from src.pipeline import resolve_data_paths
from src.feature_selection import run_feature_selection


def compute_file_sha256(file_path: Path) -> str:
    """Computes SHA256 checksum for a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


EXPECTED_DEMO_SHA256 = "c0044f6b881bf82cc442cb589a68cf1dcd92e82ab2458b3c70de242384a66bdd"


class TestPipelineDemo(unittest.TestCase):
    """Test suite ensuring data preservation and safe execution in demo mode."""

    def setUp(self):
        self.assertTrue(demo_data_file.exists(), f"Demo file not found at: {demo_data_file}")
        self.initial_demo_hash = compute_file_sha256(demo_data_file)
        self.demo_processed_path = processed_data_dir / "df_selected_demo.csv"
        self.processed_existed_initially = self.demo_processed_path.exists()

    def tearDown(self):
        # Clean up generated demo processed file if created during tests
        if not self.processed_existed_initially and self.demo_processed_path.exists():
            try:
                self.demo_processed_path.unlink()
            except Exception:
                pass

    def test_demo_golden_master_checksum(self):
        """Reference demo file must match its exact repository Golden Master SHA256 checksum."""
        current_hash = compute_file_sha256(demo_data_file)
        self.assertEqual(
            current_hash,
            EXPECTED_DEMO_SHA256,
            "Demo file content has diverged from Golden Master reference!",
        )

    def test_resolve_data_paths_demo_mode_separation(self):
        """resolve_data_paths(use_demo=True) must return separate raw input and processed output paths."""
        raw_path, processed_path, features_json = resolve_data_paths(use_demo=True)

        self.assertEqual(raw_path.resolve(), demo_data_file.resolve())
        self.assertEqual(processed_path.resolve(), self.demo_processed_path.resolve())
        self.assertEqual(features_json.resolve(), (config_dir / "selected_features.json").resolve())
        self.assertNotEqual(
            raw_path.resolve(),
            processed_path.resolve(),
            "Critical defect: processed_path must never point to raw demo_data_file",
        )

    def test_resolve_data_paths_fallback_separation(self):
        """resolve_data_paths fallback when raw CFD files are missing must also preserve demo_data_file."""
        with patch.object(Path, "exists", autospec=True) as mock_exists:
            # Simulate: processed df_selected.csv absent, CFD xlsx absent, demo file present
            def side_effect(path_self):
                p_str = str(path_self)
                if "df_selected.csv" in p_str:
                    return False
                if "cfd.xlsx" in p_str or "Dane_T5.xlsx" in p_str:
                    return False
                if "inertial_separator_demo.csv" in p_str:
                    return True
                return True

            mock_exists.side_effect = side_effect
            raw_path, processed_path, _ = resolve_data_paths(use_demo=False)
            self.assertNotEqual(
                raw_path.resolve(),
                processed_path.resolve(),
                "Fallback mode must not point processed output to raw demo file",
            )
            self.assertEqual(processed_path.name, "df_selected_demo.csv")

    def test_demo_file_hash_integrity_after_feature_selection(self):
        """SHA256 checksum of inertial_separator_demo.csv must remain strictly unchanged after feature selection."""
        raw_path, processed_path, _ = resolve_data_paths(use_demo=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_json = Path(tmp_dir) / "test_features.json"

            # Execute feature selection step
            run_feature_selection(
                raw_path=raw_path,
                output_csv_path=processed_path,
                json_path=temp_json,
            )

            # Assert output was created at destination
            self.assertTrue(processed_path.exists(), "Processed demo CSV was not created")

            # Check SHA256 of demo file post-execution
            post_hash = compute_file_sha256(demo_data_file)
            self.assertEqual(
                self.initial_demo_hash,
                post_hash,
                "CRITICAL: demo dataset was modified or overwritten during pipeline execution!",
            )

    def test_config_path_import_has_no_side_effects(self):
        """Importing config.path must not invoke create_directories()."""
        import ast
        import config.path

        # Parse AST of config/path.py to statically guarantee no top-level call to create_directories exists
        path_file = Path(config.path.__file__)
        tree = ast.parse(path_file.read_text(encoding="utf-8"))
        top_level_calls = []
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                func = node.value.func
                func_name = getattr(func, "id", None) or getattr(func, "attr", None)
                if func_name:
                    top_level_calls.append(func_name)

        self.assertNotIn(
            "create_directories",
            top_level_calls,
            "create_directories() must not be invoked at module scope upon importing config/path.py",
        )

    def test_pipeline_main_demo_select_features(self):
        """Executing pipeline CLI main() with --demo --step select_features must preserve demo data and create demo output."""
        import sys
        import pandas as pd
        from src.pipeline import main as pipeline_main
        from src.constants import TOTAL_PARTICLES

        features_json = config_dir / "selected_features.json"
        saved_json_content = features_json.read_text(encoding="utf-8") if features_json.exists() else None

        test_argv = ["src.pipeline", "--demo", "--step", "select_features"]
        with patch.object(sys, "argv", test_argv):
            try:
                pipeline_main()

                # Verify demo raw file checksum is still preserved
                post_hash = compute_file_sha256(demo_data_file)
                self.assertEqual(
                    self.initial_demo_hash,
                    post_hash,
                    "CRITICAL: demo raw dataset was modified during CLI select_features execution!",
                )

                # Verify demo output exists and preserves particle balance
                self.assertTrue(self.demo_processed_path.exists())
                df_demo_out = pd.read_csv(self.demo_processed_path)
                self.assertEqual(len(df_demo_out), 150)
                sums = df_demo_out["N1"] + df_demo_out["N2"] + df_demo_out["N3"]
                np.testing.assert_array_equal(sums.values, np.full(len(df_demo_out), TOTAL_PARTICLES))

            finally:
                # Restore original selected_features.json to avoid test pollution
                if saved_json_content is not None:
                    features_json.write_text(saved_json_content, encoding="utf-8")

    def test_create_directories_explicit_execution(self):
        """Explicit invocation of create_directories() must successfully ensure all required directories exist."""
        from config.path import create_directories, ALL_DIRS

        create_directories()
        for d in ALL_DIRS:
            self.assertTrue(d.exists(), f"Directory {d} was not created by create_directories()")
            self.assertTrue(d.is_dir(), f"Path {d} is not a directory")


if __name__ == "__main__":
    unittest.main()
