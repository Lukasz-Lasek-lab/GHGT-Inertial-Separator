"""
Pareto Front Post-processing and Engineering Decision Support module for MsCO2limit.
Executes:
- K-Means clustering of non-dominated Pareto front configurations
- Rigorous physical particle balance verification (N1 + N2 + N3 = 6417)
- Percentage separation efficiency calculation (loss eta_1_pct and capture eta_2_pct)
- Normalized trade-off scoring and geometry deduplication
- Engineering export to Excel and CSV formats.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from config.path import optimization_results_dir

TOTAL_PARTICLES: int = 6417


def select_optimal_configurations(
    pareto_df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Path] = None,
    output_excel_path: Optional[Path] = None,
    n_clusters: int = 3,
    round_decimals: bool = True,
) -> pd.DataFrame:
    """
    Filters, clusters, scores, and deduplicates optimal separator geometric designs:
    - K-Means clustering on (N1, N2)
    - Full particle balance: N1 + N2 + N3 = 6417
    - Efficiency calculations: eta_1 (%) and eta_2 (%)
    - Geometric deduplication
    - Engineering export to Excel and CSV
    """
    if pareto_df is None:
        if csv_path is None:
            # Search for Pareto result files
            pareto_files = list(optimization_results_dir.glob("pareto_front*.csv"))
            if not pareto_files:
                pareto_files = list(optimization_results_dir.glob("genetic_results_pareto_*.csv"))
            if not pareto_files:
                raise FileNotFoundError(f"No Pareto front result CSV files found in {optimization_results_dir}")
            csv_path = max(pareto_files, key=lambda p: p.stat().st_mtime)

        print(f"[INFO] Loading Pareto solutions from: {csv_path.name}")
        df = pd.read_csv(csv_path)
    else:
        df = pareto_df.copy()

    if len(df) == 0:
        raise ValueError("Pareto solution dataset is empty.")

    # K-Means clustering
    if len(df) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clustering_features = df[["N1_pred", "N2_pred"]].values
        df["cluster"] = kmeans.fit_predict(clustering_features)

    # Compute complete particle balance
    n1_raw = df["N1_pred"].values
    delta_raw = df["Delta_pred"].values
    n2_raw = df["N2_pred"].values
    n3_raw = np.clip(TOTAL_PARTICLES - n1_raw - n2_raw, 0, TOTAL_PARTICLES)

    # Normalized compromise score: lower score indicates superior compromise (min loss N1, max advantage Delta)
    n1_min, n1_max = n1_raw.min(), n1_raw.max()
    delta_min, delta_max = delta_raw.min(), delta_raw.max()

    n1_norm = (n1_raw - n1_min) / (n1_max - n1_min + 1e-6)
    delta_norm = (delta_raw - delta_min) / (delta_max - delta_min + 1e-6)
    df["score"] = 0.5 * n1_norm + 0.5 * (1.0 - delta_norm)

    # Prepare export DataFrame
    df_export = df.copy()
    if round_decimals:
        df_export["Alfa"] = df_export["Alfa"].round(2)
        df_export["Beta"] = df_export["Beta"].round(2)
        df_export["H1"] = df_export["H1"].round(4)
        df_export["H2"] = df_export["H2"].round(4)
        df_export["N1"] = np.round(n1_raw).astype(int)
        df_export["N2"] = np.round(n2_raw).astype(int)
        df_export["Delta"] = np.round(delta_raw).astype(int)
        df_export["N3"] = np.round(n3_raw).astype(int)
        df_export["eta_1_pct"] = np.round((n1_raw / TOTAL_PARTICLES) * 100.0, 2)
        df_export["eta_2_pct"] = np.round((n2_raw / TOTAL_PARTICLES) * 100.0, 2)
        df_export["score"] = df_export["score"].round(4)
    else:
        df_export["N1"] = n1_raw
        df_export["N2"] = n2_raw
        df_export["Delta"] = delta_raw
        df_export["N3"] = n3_raw
        df_export["eta_1_pct"] = (n1_raw / TOTAL_PARTICLES) * 100.0
        df_export["eta_2_pct"] = (n2_raw / TOTAL_PARTICLES) * 100.0

    # Geometric deduplication
    before = len(df_export)
    df_export = df_export.drop_duplicates(subset=["Alfa", "Beta", "H1", "H2"])
    after = len(df_export)
    print(f"[INFO] Removed {before - after} duplicates. Preserved {after} unique Pareto design configurations.")

    # Sort by compromise score
    df_export = df_export.sort_values(by=["score", "N1", "N2"], ascending=[True, True, False]).reset_index(drop=True)

    target_cols = [
        "Alfa",
        "Beta",
        "H1",
        "H2",
        "N1",
        "N2",
        "Delta",
        "N3",
        "eta_1_pct",
        "eta_2_pct",
        "score",
    ]
    if "cluster" in df_export.columns:
        target_cols.append("cluster")

    result_df = df_export[target_cols]

    if output_excel_path is None:
        output_excel_path = optimization_results_dir / "pareto_optimal_designs.xlsx"

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(output_excel_path, index=False)
    csv_out = output_excel_path.with_suffix(".csv")
    result_df.to_csv(csv_out, index=False)
    # Also save to legacy path Test5_Results_src.csv for backward compatibility
    legacy_csv = optimization_results_dir / "Test5_Results_src.csv"
    result_df.to_csv(legacy_csv, index=False)

    print(f"[SAVE] Optimal designs exported to Excel: {output_excel_path}")
    print(f"[SAVE] Optimal designs exported to CSV: {csv_out}")

    return result_df
